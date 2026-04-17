"""
Tests for the consensus engine.
"""
import pytest
from services.consensus_engine import ConsensusEngine, NodePrediction


def _make_preds(scores_trusts):
    return [
        NodePrediction(
            node_id=f"node_{i}",
            fraud_prob=score,
            trust_score=trust,
        )
        for i, (score, trust) in enumerate(scores_trusts)
    ]


class TestConsensusEngine:
    def setup_method(self):
        self.engine = ConsensusEngine(disagreement_threshold=0.20)

    def test_simple_unanimous_fraud(self):
        preds = _make_preds([(0.9, 0.8), (0.85, 0.9), (0.92, 0.75)])
        result = self.engine.compute("txn_001", preds)
        assert result.final_score > 0.5
        assert not result.conflict_detected

    def test_simple_unanimous_legit(self):
        preds = _make_preds([(0.05, 0.8), (0.08, 0.9), (0.03, 0.75)])
        result = self.engine.compute("txn_002", preds)
        assert result.final_score < 0.5
        assert not result.conflict_detected

    def test_disagreement_detected(self):
        # Nodes split 50/50
        preds = _make_preds([(0.95, 0.8), (0.90, 0.7), (0.05, 0.8), (0.03, 0.7)])
        result = self.engine.compute("txn_003", preds)
        assert result.conflict_detected

    def test_trust_down_weights_outlier(self):
        """Low-trust node reporting fraud should not dominate."""
        preds = _make_preds([
            (0.05, 0.9),  # high trust, legit
            (0.04, 0.85), # high trust, legit
            (0.99, 0.05), # very low trust, fraud claim
        ])
        result = self.engine.compute("txn_004", preds)
        # Trust-adjusted should be < 0.5 (high-trust nodes dominate)
        assert result.trust_adjusted < 0.5

    def test_single_node(self):
        preds = _make_preds([(0.75, 0.8)])
        result = self.engine.compute("txn_005", preds)
        assert abs(result.simple_avg - 0.75) < 1e-6

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            self.engine.compute("txn_000", [])

    def test_result_serialisable(self):
        preds = _make_preds([(0.6, 0.7), (0.4, 0.8)])
        result = self.engine.compute("txn_006", preds)
        d = result.to_dict()
        assert "final_score" in d
        assert isinstance(d["final_score"], float)
        assert "node_scores" in d
