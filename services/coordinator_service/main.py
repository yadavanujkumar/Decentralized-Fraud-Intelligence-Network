"""
MeshGuard AI – Coordinator Service
====================================
Central coordination layer responsible for:
  1. Federated learning aggregation (FedAvg)
  2. Consensus scoring across nodes
  3. Trust score management
  4. Graph intelligence integration
  5. Real-time inference endpoint (/predict)

Endpoints:
  POST /predict          – query all nodes, run consensus, return final score
  POST /federated/round  – trigger a full FL round
  GET  /trust            – get all trust scores
  GET  /trust/{node_id}  – get single node trust score
  GET  /graph/stats      – graph statistics
  GET  /health           – liveness
  GET  /nodes            – registered node info
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ml.federated.fedavg import aggregate, detect_poisoning
from services.consensus_engine import ConsensusEngine, NodePrediction, get_engine
from services.trust_engine import TrustEngine, get_trust_engine
from services.graph_engine import TransactionGraph, Transaction, get_graph

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger("coordinator")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PORT = int(os.getenv("PORT", "8000"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10.0"))

# Weight applied to the graph-intelligence score when blending with the
# consensus score.  Kept intentionally small so the graph signal acts as a
# soft boost rather than dominating the consensus result.
GRAPH_BOOST_FACTOR = 0.15

# Node registry: populated from env var NODE_URLS (comma-separated)
# e.g. NODE_URLS=http://node_1:8001,http://node_2:8002,...
_NODE_URLS_ENV = os.getenv(
    "NODE_URLS",
    "http://node_1:8001,http://node_2:8002,http://node_3:8003,"
    "http://node_4:8004,http://node_5:8005,http://node_6:8006,"
    "http://node_7:8007,http://node_malicious:8008",
)

NODE_REGISTRY: Dict[str, str] = {}  # {node_id: base_url}


def _build_registry() -> None:
    for url in _NODE_URLS_ENV.split(","):
        url = url.strip()
        if not url:
            continue
        # Derive node_id from last path segment or hostname
        node_id = url.rstrip("/").split("//")[-1].split(":")[0]
        NODE_REGISTRY[node_id] = url
    logger.info("Node registry: %s", list(NODE_REGISTRY.keys()))


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
consensus_engine: ConsensusEngine = get_engine()
trust_engine: TrustEngine = get_trust_engine()
tx_graph: TransactionGraph = get_graph()

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _build_registry()
    for node_id in NODE_REGISTRY:
        trust_engine.register_node(node_id)
    logger.info("Coordinator ready – %d nodes registered", len(NODE_REGISTRY))
    yield
    logger.info("Coordinator shutting down")


app = FastAPI(title="MeshGuard Coordinator", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
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
    account_id: Optional[str] = None
    merchant_id: Optional[str] = None
    device_id: Optional[str] = None


class PredictResponse(BaseModel):
    transaction_id: str
    local_scores: Dict[str, float]
    consensus_score: float
    trust_adjusted_score: float
    graph_score: float
    final_score: float
    is_fraud: bool
    disagreement_score: float
    conflict_detected: bool
    resolution: str
    latency_ms: float
    participating_nodes: List[str]


class FederatedRoundResponse(BaseModel):
    round_id: str
    participating_nodes: List[str]
    excluded_nodes: List[str]
    aggregated: bool
    flagged_nodes: List[str]
    round_latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "nodes": list(NODE_REGISTRY.keys())}


@app.get("/nodes")
async def list_nodes():
    result = []
    for node_id, url in NODE_REGISTRY.items():
        result.append({
            "node_id": node_id,
            "url": url,
            "trust_score": trust_engine.get_trust(node_id),
        })
    return result


@app.post("/predict", response_model=PredictResponse)
async def predict(tx: TransactionRequest):
    t0 = time.perf_counter()

    # 1. Query all nodes in parallel
    node_predictions = await _query_all_nodes(tx.dict())

    if not node_predictions:
        raise HTTPException(status_code=503, detail="No nodes available")

    # 2. Build NodePrediction objects (with current trust scores)
    np_list = [
        NodePrediction(
            node_id=np_raw["node_id"],
            fraud_prob=np_raw["fraud_prob"],
            trust_score=trust_engine.get_trust(np_raw["node_id"]),
            latency_ms=np_raw.get("latency_ms", 0.0),
        )
        for np_raw in node_predictions
    ]

    # 3. Consensus
    consensus_result = consensus_engine.compute(tx.transaction_id, np_list)

    # 4. Graph analysis
    tx_obj = Transaction(
        transaction_id=tx.transaction_id,
        account_id=tx.account_id,
        merchant_id=tx.merchant_id,
        device_id=tx.device_id,
        amount=tx.amount,
        timestamp=time.time(),
        fraud_prob=consensus_result.final_score,
    )
    tx_graph.add_transaction(tx_obj)
    graph_result = tx_graph.analyse(tx.transaction_id)

    # 5. Final score – blend consensus + graph signal
    graph_boost = graph_result.graph_fraud_score * GRAPH_BOOST_FACTOR
    final_score = float(min(1.0, consensus_result.final_score + graph_boost))

    # 6. Latency penalty trust update
    for np_raw in node_predictions:
        trust_engine.update_latency(np_raw["node_id"], np_raw.get("latency_ms", 0.0))

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "predict txn=%s  final=%.4f  latency=%.1fms",
        tx.transaction_id,
        final_score,
        latency_ms,
    )

    return PredictResponse(
        transaction_id=tx.transaction_id,
        local_scores={p.node_id: round(p.fraud_prob, 4) for p in np_list},
        consensus_score=consensus_result.weighted_avg,
        trust_adjusted_score=consensus_result.trust_adjusted,
        graph_score=graph_result.graph_fraud_score,
        final_score=final_score,
        is_fraud=final_score >= 0.5,
        disagreement_score=consensus_result.disagreement_score,
        conflict_detected=consensus_result.conflict_detected,
        resolution=consensus_result.resolution,
        latency_ms=round(latency_ms, 2),
        participating_nodes=consensus_result.participating_nodes,
    )


@app.post("/federated/round", response_model=FederatedRoundResponse)
async def federated_round(background_tasks: BackgroundTasks):
    round_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    # 1. Collect weights from all nodes
    weight_responses = await _collect_weights()

    if not weight_responses:
        raise HTTPException(status_code=503, detail="No nodes returned weights")

    node_ids = [r["node_id"] for r in weight_responses]
    weight_dicts = [r["weights"] for r in weight_responses]
    n_samples_list = [r["n_samples"] for r in weight_responses]

    # 2. Poison detection
    flagged = detect_poisoning(weight_dicts, node_ids)
    for node_id in flagged:
        trust_engine.flag_poisoning(node_id)

    # 3. Filter out flagged (highly untrusted) nodes
    trust_scores = trust_engine.get_all_trust()
    excluded = [nid for nid in node_ids if trust_scores.get(nid, 1.0) < 0.15]
    participating = [nid for nid in node_ids if nid not in excluded]

    clean_updates = [
        (w, n)
        for w, n, nid in zip(weight_dicts, n_samples_list, node_ids)
        if nid not in excluded
    ]

    if not clean_updates:
        return FederatedRoundResponse(
            round_id=round_id,
            participating_nodes=[],
            excluded_nodes=excluded,
            aggregated=False,
            flagged_nodes=flagged,
            round_latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # 4. FedAvg aggregation
    aggregated_weights = aggregate(
        clean_updates,
        trust_scores={nid: trust_scores.get(nid, 1.0) for nid in participating},
        node_ids=participating,
    )

    # 5. Redistribute aggregated weights to all nodes
    background_tasks.add_task(_push_weights_to_nodes, aggregated_weights)

    # 6. Update trust based on weight consistency with aggregated model
    for nid, w in zip(node_ids, weight_dicts):
        if nid in excluded:
            continue
        from ml.federated.fedavg import compute_gradient_similarity
        sim = compute_gradient_similarity(w, aggregated_weights)
        trust_engine.update_consistency(nid, sim)

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("FL round=%s  nodes=%s  flagged=%s  latency=%.0fms",
                round_id, participating, flagged, latency_ms)

    return FederatedRoundResponse(
        round_id=round_id,
        participating_nodes=participating,
        excluded_nodes=excluded,
        aggregated=True,
        flagged_nodes=flagged,
        round_latency_ms=round(latency_ms, 2),
    )


@app.get("/trust")
async def get_all_trust():
    return trust_engine.all_records()


@app.get("/trust/{node_id}")
async def get_trust(node_id: str):
    return trust_engine.get_record(node_id)


@app.get("/graph/stats")
async def graph_stats():
    return tx_graph.get_stats()


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def _query_node(
    client: httpx.AsyncClient,
    node_id: str,
    base_url: str,
    tx_data: Dict,
) -> Optional[Dict]:
    try:
        t0 = time.perf_counter()
        r = await client.post(f"{base_url}/predict", json=tx_data, timeout=REQUEST_TIMEOUT)
        latency_ms = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            data = r.json()
            data["latency_ms"] = latency_ms
            return data
    except Exception as exc:
        logger.warning("Node %s unreachable: %s", node_id, exc)
    return None


async def _query_all_nodes(tx_data: Dict) -> List[Dict]:
    async with httpx.AsyncClient() as client:
        tasks = [
            _query_node(client, nid, url, tx_data)
            for nid, url in NODE_REGISTRY.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    return [r for r in results if r is not None]


async def _collect_node_weights(
    client: httpx.AsyncClient,
    node_id: str,
    base_url: str,
) -> Optional[Dict]:
    try:
        r = await client.get(f"{base_url}/weights", timeout=REQUEST_TIMEOUT * 2)
        if r.status_code == 200:
            return r.json()
    except Exception as exc:
        logger.warning("Cannot collect weights from %s: %s", node_id, exc)
    return None


async def _collect_weights() -> List[Dict]:
    async with httpx.AsyncClient() as client:
        tasks = [
            _collect_node_weights(client, nid, url)
            for nid, url in NODE_REGISTRY.items()
        ]
        results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def _push_weights_to_node(
    client: httpx.AsyncClient,
    node_id: str,
    base_url: str,
    weights: Dict,
) -> None:
    try:
        await client.post(
            f"{base_url}/weights",
            json={"weights": weights},
            timeout=REQUEST_TIMEOUT,
        )
    except Exception as exc:
        logger.warning("Failed to push weights to %s: %s", node_id, exc)


async def _push_weights_to_nodes(weights: Dict) -> None:
    async with httpx.AsyncClient() as client:
        tasks = [
            _push_weights_to_node(client, nid, url, weights)
            for nid, url in NODE_REGISTRY.items()
        ]
        await asyncio.gather(*tasks)
    logger.info("Pushed aggregated weights to %d nodes", len(NODE_REGISTRY))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("services.coordinator_service.main:app", host="0.0.0.0", port=PORT, reload=False)
