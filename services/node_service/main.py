"""
MeshGuard AI – Node Service
============================
Each bank / fintech node runs this FastAPI service.

Endpoints:
  POST /train                – trigger local training round
  POST /predict              – local fraud prediction
  GET  /weights              – return current model weights (for FL server)
  POST /weights              – receive updated global weights from coordinator
  GET  /health               – liveness probe
  GET  /metrics              – node metrics
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger("node_service")

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
NODE_ID = os.getenv("NODE_ID", "node_1")
DATA_PATH = os.getenv("DATA_PATH", f"data/simulated/{NODE_ID}.csv")
COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://coordinator:8000")
PORT = int(os.getenv("PORT", "8001"))
MALICIOUS = os.getenv("MALICIOUS", "false").lower() == "true"
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "5"))
# Maximum fraud probability reported by the malicious node (prediction suppression attack).
MALICIOUS_MAX_FRAUD_PROB = 0.15

# ---------------------------------------------------------------------------
# Lazy imports so the service starts even without heavy deps installed
# in the coordinator container
# ---------------------------------------------------------------------------
from ml.training.trainer import NodeTrainer, FEATURE_COLS
from ml.models.fraud_model import build_model

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_trainer: Optional[NodeTrainer] = None
_metrics: Dict[str, Any] = {
    "predictions_served": 0,
    "training_rounds": 0,
    "last_train_metrics": {},
}


def get_trainer() -> NodeTrainer:
    global _trainer
    if _trainer is None:
        _trainer = NodeTrainer(node_id=NODE_ID, data_path=DATA_PATH)
    return _trainer


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Node %s starting up (malicious=%s)", NODE_ID, MALICIOUS)
    # Attempt initial training in background
    try:
        trainer = get_trainer()
        trainer.train(epochs=TRAIN_EPOCHS)
        _metrics["training_rounds"] += 1
        logger.info("Node %s initial training complete", NODE_ID)
    except Exception as exc:
        logger.warning("Initial training skipped: %s", exc)
    yield
    logger.info("Node %s shutting down", NODE_ID)


app = FastAPI(
    title=f"MeshGuard Node – {NODE_ID}",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TransactionFeatures(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    amount: float
    time_of_day: float
    day_of_week: float
    merchant_category: float
    distance_from_home: float
    transaction_count_1h: float
    transaction_count_24h: float
    avg_amount_30d: float
    velocity_change: float
    is_international: float
    card_present: float
    device_trust_score: float
    # Optional graph metadata
    account_id: Optional[str] = None
    merchant_id: Optional[str] = None
    device_id: Optional[str] = None


class PredictResponse(BaseModel):
    transaction_id: str
    node_id: str
    fraud_prob: float
    is_fraud: bool
    latency_ms: float


class TrainRequest(BaseModel):
    epochs: int = TRAIN_EPOCHS
    federated_weights: Optional[Dict] = None  # global weights from coordinator


class TrainResponse(BaseModel):
    node_id: str
    epochs: int
    metrics: Dict[str, Any]
    training_rounds: int


class WeightsResponse(BaseModel):
    node_id: str
    weights: Dict[str, Any]
    n_samples: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "node_id": NODE_ID, "malicious": MALICIOUS}


@app.get("/metrics")
async def metrics():
    trust_info: Dict = {}
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{COORDINATOR_URL}/trust/{NODE_ID}")
            if r.status_code == 200:
                trust_info = r.json()
    except Exception:
        pass
    return {**_metrics, "node_id": NODE_ID, "trust": trust_info}


@app.post("/predict", response_model=PredictResponse)
async def predict(tx: TransactionFeatures):
    t0 = time.perf_counter()
    trainer = get_trainer()
    features = {col: getattr(tx, col) for col in FEATURE_COLS}

    if MALICIOUS:
        # Malicious node: always report low fraud probability to evade detection
        fraud_prob = float(np.random.uniform(0.0, MALICIOUS_MAX_FRAUD_PROB))
    else:
        fraud_prob = trainer.predict(features)

    latency_ms = (time.perf_counter() - t0) * 1000
    _metrics["predictions_served"] += 1

    return PredictResponse(
        transaction_id=tx.transaction_id,
        node_id=NODE_ID,
        fraud_prob=fraud_prob,
        is_fraud=fraud_prob >= 0.5,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    trainer = get_trainer()

    # If coordinator sent updated global weights, load them first
    if req.federated_weights:
        try:
            trainer.set_weights(req.federated_weights)
            logger.info("Node %s loaded federated weights", NODE_ID)
        except Exception as exc:
            logger.warning("Failed to load federated weights: %s", exc)

    metrics = trainer.train(epochs=req.epochs)
    _metrics["training_rounds"] += 1
    _metrics["last_train_metrics"] = metrics

    return TrainResponse(
        node_id=NODE_ID,
        epochs=req.epochs,
        metrics=metrics,
        training_rounds=_metrics["training_rounds"],
    )


@app.get("/weights", response_model=WeightsResponse)
async def get_weights():
    trainer = get_trainer()
    weights = trainer.get_weights()

    # Malicious node: inject noise into weights before sending
    if MALICIOUS:
        weights = _inject_noise(weights, scale=5.0)

    # Count samples from the data file
    n_samples = _count_samples()
    return WeightsResponse(node_id=NODE_ID, weights=weights, n_samples=n_samples)


@app.post("/weights")
async def set_weights(body: Dict):
    trainer = get_trainer()
    try:
        trainer.set_weights(body["weights"])
        return {"status": "ok", "node_id": NODE_ID}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inject_noise(weights: Dict, scale: float = 5.0) -> Dict:
    """Malicious node: corrupt weights with large noise."""
    noisy = {}
    for k, v in weights.items():
        arr = np.array(v, dtype=np.float32)
        noise = np.random.randn(*arr.shape).astype(np.float32) * scale
        noisy[k] = (arr + noise).tolist()
    return noisy


def _count_samples() -> int:
    try:
        import pandas as pd
        return len(pd.read_csv(DATA_PATH))
    except Exception:
        return 5000


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("services.node_service.main:app", host="0.0.0.0", port=PORT, reload=False)
