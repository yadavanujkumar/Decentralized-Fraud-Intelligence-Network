"""
Local training pipeline for a single federated node.

Usage (standalone):
    python -m ml.training.trainer \
        --data-path data/simulated/node_1.csv \
        --node-id node_1 \
        --epochs 10 \
        --output-dir /tmp/models
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ml.models.fraud_model import FEATURE_DIM, FraudMLP, build_model

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "amount",
    "time_of_day",
    "day_of_week",
    "merchant_category",
    "distance_from_home",
    "transaction_count_1h",
    "transaction_count_24h",
    "avg_amount_30d",
    "velocity_change",
    "is_international",
    "card_present",
    "device_trust_score",
]
LABEL_COL = "is_fraud"


class NodeTrainer:
    """Encapsulates local training for a single federated node."""

    def __init__(
        self,
        node_id: str,
        data_path: str,
        model: Optional[FraudMLP] = None,
        lr: float = 1e-3,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> None:
        self.node_id = node_id
        self.data_path = data_path
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.scaler = StandardScaler()
        self.model: FraudMLP = model if model else build_model()
        self.model.to(self.device)
        self._data_loaded = False

    # ------------------------------------------------------------------ #
    # Data helpers                                                        #
    # ------------------------------------------------------------------ #

    def _load_data(self) -> Tuple[DataLoader, DataLoader]:
        df = pd.read_csv(self.data_path)
        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df[LABEL_COL].values.astype(np.float32)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_tr = self.scaler.fit_transform(X_tr).astype(np.float32)
        X_val = self.scaler.transform(X_val).astype(np.float32)

        tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        tr_loader = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        self._data_loaded = True
        return tr_loader, val_loader

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def train(self, epochs: int = 5) -> Dict[str, float]:
        """Train the model for `epochs` epochs; return final metrics."""
        tr_loader, val_loader = self._load_data()

        # Use weighted loss to handle class imbalance
        pos_count = sum(y.sum().item() for _, y in tr_loader)
        total = sum(len(y) for _, y in tr_loader)
        neg_count = total - pos_count
        pos_weight = torch.tensor(
            [neg_count / max(pos_count, 1)], dtype=torch.float32, device=self.device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # We run forward through net directly (bypassing sigmoid) for BCEWithLogits
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            steps_per_epoch=len(tr_loader),
            epochs=epochs,
        )

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for X_batch, y_batch in tr_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.model.net(X_batch).squeeze(-1)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(tr_loader)
            logger.debug("[%s] epoch %d  loss=%.4f", self.node_id, epoch, avg_loss)

        metrics = self.evaluate(val_loader)
        logger.info("[%s] training done  %s", self.node_id, metrics)
        return metrics

    # ------------------------------------------------------------------ #
    # Evaluation                                                          #
    # ------------------------------------------------------------------ #

    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        if loader is None:
            _, loader = self._load_data()

        self.model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                probs = self.model(X_batch).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(y_batch.numpy().tolist())

        probs_arr = np.array(all_probs)
        labels_arr = np.array(all_labels)
        preds = (probs_arr >= 0.5).astype(int)

        tp = int(((preds == 1) & (labels_arr == 1)).sum())
        fp = int(((preds == 1) & (labels_arr == 0)).sum())
        fn = int(((preds == 0) & (labels_arr == 1)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support_fraud": tp + fn,
        }

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, features: Dict[str, float]) -> float:
        """Return fraud probability for a single transaction dict."""
        x = np.array([[features[col] for col in FEATURE_COLS]], dtype=np.float32)
        if self._data_loaded:
            x = self.scaler.transform(x).astype(np.float32)
        x_t = torch.from_numpy(x).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return float(self.model(x_t).item())

    def get_weights(self) -> Dict:
        return self.model.get_weights()

    def set_weights(self, weights: Dict) -> None:
        self.model.set_weights(weights)


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    trainer = NodeTrainer(node_id=args.node_id, data_path=args.data_path)
    metrics = trainer.train(epochs=args.epochs)
    print(f"Metrics: {metrics}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save(str(out / f"{args.node_id}.pt"))
    print(f"Model saved to {out / args.node_id}.pt")
