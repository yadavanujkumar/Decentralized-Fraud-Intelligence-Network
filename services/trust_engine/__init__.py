"""
Trust / Reputation Engine
=========================
Maintains a per-node trust score in [0, 1].

Trust updates are driven by:
  1. Prediction accuracy signal – did this node's prediction agree with the
     eventual consensus label?
  2. Consistency – cosine-similarity of the node's weight update vs. the
     median of all updates (poison detection signal).
  3. Latency – nodes that respond too slowly are penalised slightly.

The trust score is an exponentially-weighted moving average (EWMA) so that
recent behaviour matters more.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TRUST = 0.80    # initial trust for new nodes
ALPHA = 0.20            # EWMA smoothing for trust updates
MIN_TRUST = 0.05        # floor – never fully exclude a node
MAX_TRUST = 1.00
LATENCY_PENALTY_MS = 5000.0  # ms above which latency starts penalising


@dataclass
class TrustRecord:
    node_id: str
    trust_score: float = DEFAULT_TRUST
    total_predictions: int = 0
    correct_predictions: int = 0
    poison_flags: int = 0
    last_update: float = 0.0
    history: List[float] = field(default_factory=list)


class TrustEngine:
    """Thread-safe node trust/reputation registry."""

    def __init__(
        self,
        alpha: float = ALPHA,
        min_trust: float = MIN_TRUST,
        max_trust: float = MAX_TRUST,
        initial_trust: float = DEFAULT_TRUST,
    ) -> None:
        self.alpha = alpha
        self.min_trust = min_trust
        self.max_trust = max_trust
        self.initial_trust = initial_trust
        self._records: Dict[str, TrustRecord] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_trust(self, node_id: str) -> float:
        with self._lock:
            return self._get_or_create(node_id).trust_score

    def get_all_trust(self) -> Dict[str, float]:
        with self._lock:
            return {nid: rec.trust_score for nid, rec in self._records.items()}

    def register_node(self, node_id: str) -> None:
        with self._lock:
            self._get_or_create(node_id)

    def update_prediction_accuracy(
        self,
        node_id: str,
        predicted_fraud: bool,
        ground_truth_fraud: bool,
    ) -> float:
        """
        Update trust based on prediction accuracy.
        Returns the new trust score.
        """
        with self._lock:
            rec = self._get_or_create(node_id)
            rec.total_predictions += 1
            correct = predicted_fraud == ground_truth_fraud
            if correct:
                rec.correct_predictions += 1

            reward = 1.0 if correct else 0.0
            new_trust = self._ewma_update(rec.trust_score, reward)
            rec.trust_score = new_trust
            rec.history.append(new_trust)
            logger.debug(
                "[TrustEngine] node=%s  correct=%s  trust→%.4f",
                node_id,
                correct,
                new_trust,
            )
            return new_trust

    def update_consistency(
        self,
        node_id: str,
        cosine_similarity: float,
    ) -> float:
        """
        Update trust based on weight consistency with the global update.
        cosine_similarity in [-1, 1]; negative values = strongly divergent.
        Returns new trust score.
        """
        with self._lock:
            rec = self._get_or_create(node_id)
            # Map cosine_similarity from [-1,1] to [0,1]
            reward = (cosine_similarity + 1.0) / 2.0
            # Penalise heavily if similarity < 0
            if cosine_similarity < 0:
                reward = reward * 0.5

            new_trust = self._ewma_update(rec.trust_score, reward)
            rec.trust_score = new_trust
            rec.history.append(new_trust)
            logger.debug(
                "[TrustEngine] node=%s  cos_sim=%.4f  trust→%.4f",
                node_id,
                cosine_similarity,
                new_trust,
            )
            return new_trust

    def flag_poisoning(self, node_id: str, penalty: float = 0.40) -> float:
        """
        Hard-penalise a node for suspected model poisoning.
        Returns new trust score.
        """
        with self._lock:
            rec = self._get_or_create(node_id)
            rec.poison_flags += 1
            new_trust = max(self.min_trust, rec.trust_score * (1.0 - penalty))
            rec.trust_score = new_trust
            rec.history.append(new_trust)
            logger.warning(
                "[TrustEngine] POISON FLAG node=%s  trust→%.4f  (flags=%d)",
                node_id,
                new_trust,
                rec.poison_flags,
            )
            return new_trust

    def update_latency(self, node_id: str, latency_ms: float) -> float:
        """Slight trust penalty for very slow nodes."""
        if latency_ms <= LATENCY_PENALTY_MS:
            return self.get_trust(node_id)
        with self._lock:
            rec = self._get_or_create(node_id)
            # Penalty proportional to how much over the threshold
            excess = (latency_ms - LATENCY_PENALTY_MS) / LATENCY_PENALTY_MS
            penalty = min(0.05 * excess, 0.15)
            new_trust = max(self.min_trust, rec.trust_score - penalty)
            rec.trust_score = new_trust
            return new_trust

    def get_record(self, node_id: str) -> Dict:
        with self._lock:
            rec = self._get_or_create(node_id)
            return {
                "node_id": rec.node_id,
                "trust_score": round(rec.trust_score, 4),
                "total_predictions": rec.total_predictions,
                "correct_predictions": rec.correct_predictions,
                "poison_flags": rec.poison_flags,
                "accuracy": (
                    round(rec.correct_predictions / rec.total_predictions, 4)
                    if rec.total_predictions > 0
                    else None
                ),
                "history": [round(h, 4) for h in rec.history[-20:]],  # last 20
            }

    def all_records(self) -> List[Dict]:
        with self._lock:
            return [self.get_record(nid) for nid in self._records]

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _get_or_create(self, node_id: str) -> TrustRecord:
        if node_id not in self._records:
            self._records[node_id] = TrustRecord(
                node_id=node_id, trust_score=self.initial_trust
            )
        return self._records[node_id]

    def _ewma_update(self, current: float, reward: float) -> float:
        new_val = (1.0 - self.alpha) * current + self.alpha * reward
        return float(np.clip(new_val, self.min_trust, self.max_trust))


# ------------------------------------------------------------------ #
# Singleton                                                           #
# ------------------------------------------------------------------ #

_trust_engine: Optional[TrustEngine] = None


def get_trust_engine() -> TrustEngine:
    global _trust_engine
    if _trust_engine is None:
        _trust_engine = TrustEngine()
    return _trust_engine
