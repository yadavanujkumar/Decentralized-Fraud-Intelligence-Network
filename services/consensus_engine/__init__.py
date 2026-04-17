"""
Consensus Engine
================
Aggregates fraud-probability predictions from multiple nodes into a single
consensus score. Supports:
  - weighted averaging (weighted by trust scores)
  - disagreement detection (via variance / entropy analysis)
  - conflict resolution when high disagreement is detected
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DISAGREEMENT_THRESHOLD = 0.20   # std-dev above which we flag disagreement
HIGH_CONFIDENCE_THRESHOLD = 0.80  # unweighted vote above which a node is "certain"


@dataclass
class NodePrediction:
    node_id: str
    fraud_prob: float   # in [0, 1]
    trust_score: float  # in [0, 1]
    latency_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class ConsensusResult:
    transaction_id: str
    simple_avg: float
    weighted_avg: float
    trust_adjusted: float
    disagreement_score: float
    conflict_detected: bool
    resolution: str
    participating_nodes: List[str]
    node_scores: Dict[str, float]
    final_score: float

    def to_dict(self) -> Dict:
        return {
            "transaction_id": self.transaction_id,
            "simple_avg": round(self.simple_avg, 4),
            "weighted_avg": round(self.weighted_avg, 4),
            "trust_adjusted": round(self.trust_adjusted, 4),
            "disagreement_score": round(self.disagreement_score, 4),
            "conflict_detected": self.conflict_detected,
            "resolution": self.resolution,
            "participating_nodes": self.participating_nodes,
            "node_scores": {k: round(v, 4) for k, v in self.node_scores.items()},
            "final_score": round(self.final_score, 4),
        }


class ConsensusEngine:
    """
    Aggregates node predictions into a consensus fraud score.
    """

    def __init__(
        self,
        disagreement_threshold: float = DISAGREEMENT_THRESHOLD,
        min_nodes: int = 2,
    ) -> None:
        self.disagreement_threshold = disagreement_threshold
        self.min_nodes = min_nodes

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def compute(
        self,
        transaction_id: str,
        predictions: List[NodePrediction],
    ) -> ConsensusResult:
        """
        Compute consensus fraud score from a list of node predictions.
        """
        if not predictions:
            raise ValueError("No predictions supplied to consensus engine")

        node_ids = [p.node_id for p in predictions]
        probs = np.array([p.fraud_prob for p in predictions], dtype=np.float64)
        trusts = np.array([p.trust_score for p in predictions], dtype=np.float64)

        # Clamp
        probs = np.clip(probs, 0.0, 1.0)
        trusts = np.clip(trusts, 0.0, 1.0)

        simple_avg = float(probs.mean())
        weighted_avg = self._weighted_average(probs, trusts)
        disagreement = self._disagreement_score(probs)

        conflict = bool(disagreement > self.disagreement_threshold)
        resolution = "normal"
        if conflict:
            resolution, resolved_score = self._resolve_conflict(probs, trusts, predictions)
        else:
            resolved_score = weighted_avg

        # Final trust-adjusted score (only high-trust nodes)
        trust_adjusted = self._trust_adjusted(probs, trusts)
        final_score = resolved_score

        result = ConsensusResult(
            transaction_id=transaction_id,
            simple_avg=simple_avg,
            weighted_avg=weighted_avg,
            trust_adjusted=trust_adjusted,
            disagreement_score=disagreement,
            conflict_detected=conflict,
            resolution=resolution,
            participating_nodes=node_ids,
            node_scores=dict(zip(node_ids, probs.tolist())),
            final_score=final_score,
        )

        logger.info(
            "Consensus txn=%s  final=%.4f  conflict=%s  resolution=%s",
            transaction_id,
            final_score,
            conflict,
            resolution,
        )
        return result

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _weighted_average(probs: np.ndarray, trusts: np.ndarray) -> float:
        w_sum = trusts.sum()
        if w_sum <= 0:
            return float(probs.mean())
        return float((probs * trusts).sum() / w_sum)

    @staticmethod
    def _disagreement_score(probs: np.ndarray) -> float:
        """Standard deviation of node predictions (range 0–0.5)."""
        if len(probs) < 2:
            return 0.0
        return float(probs.std())

    @staticmethod
    def _trust_adjusted(probs: np.ndarray, trusts: np.ndarray) -> float:
        """Only use nodes whose trust > mean trust."""
        mean_trust = trusts.mean()
        mask = trusts >= mean_trust
        if mask.sum() == 0:
            return float(probs.mean())
        return float(probs[mask].mean())

    def _resolve_conflict(
        self,
        probs: np.ndarray,
        trusts: np.ndarray,
        predictions: List[NodePrediction],
    ) -> Tuple[str, float]:
        """
        Conflict-resolution strategies:
        1. If a clear majority votes fraud (> 0.5), trust the majority.
        2. If there are high-trust nodes that agree, use their average.
        3. Fall back to median (more robust to outliers).
        """
        majority_fraud = float((probs >= 0.5).mean())
        majority_legit = 1.0 - majority_fraud

        # Strategy 1 – majority vote
        if majority_fraud >= 0.6:
            score = float(probs[probs >= 0.5].mean())
            return "majority_fraud", score
        if majority_legit >= 0.6:
            score = float(probs[probs < 0.5].mean())
            return "majority_legit", score

        # Strategy 2 – high-trust agreement
        high_trust = trusts >= 0.7
        if high_trust.sum() >= 2:
            score = float(self._weighted_average(probs[high_trust], trusts[high_trust]))
            return "high_trust_nodes", score

        # Strategy 3 – median (robust)
        return "median_fallback", float(np.median(probs))


# ------------------------------------------------------------------ #
# Module-level singleton (shared by node & coordinator services)      #
# ------------------------------------------------------------------ #

_engine: Optional[ConsensusEngine] = None


def get_engine() -> ConsensusEngine:
    global _engine
    if _engine is None:
        _engine = ConsensusEngine()
    return _engine
