"""
Tests for the trust/reputation engine.
"""
import pytest
from services.trust_engine import TrustEngine, DEFAULT_TRUST


class TestTrustEngine:
    def setup_method(self):
        self.engine = TrustEngine(alpha=0.20, min_trust=0.05, max_trust=1.0)

    def test_initial_trust(self):
        assert abs(self.engine.get_trust("node_1") - DEFAULT_TRUST) < 1e-6

    def test_correct_prediction_increases_trust(self):
        t0 = self.engine.get_trust("node_a")
        t1 = self.engine.update_prediction_accuracy("node_a", True, True)
        assert t1 >= t0

    def test_wrong_prediction_decreases_trust(self):
        # Start from DEFAULT_TRUST
        t0 = self.engine.get_trust("node_b")
        t1 = self.engine.update_prediction_accuracy("node_b", False, True)
        assert t1 < t0

    def test_poison_flag_penalises(self):
        t0 = self.engine.get_trust("malicious_node")
        t1 = self.engine.flag_poisoning("malicious_node", penalty=0.40)
        assert t1 < t0
        assert t1 >= self.engine.min_trust

    def test_repeated_poison_capped_at_min(self):
        for _ in range(20):
            self.engine.flag_poisoning("bad_node", penalty=0.50)
        assert self.engine.get_trust("bad_node") >= self.engine.min_trust

    def test_consistency_high_cosine_improves_trust(self):
        t0 = self.engine.get_trust("node_c")
        # Perfect cosine similarity (1.0) → reward = 1.0 → trust increases
        for _ in range(5):
            t0 = self.engine.update_consistency("node_c", cosine_similarity=1.0)
        assert t0 > DEFAULT_TRUST * 0.9  # should not have dropped

    def test_consistency_negative_cosine_drops_trust(self):
        t0 = DEFAULT_TRUST
        for _ in range(10):
            t0 = self.engine.update_consistency("node_d", cosine_similarity=-0.9)
        assert t0 < DEFAULT_TRUST

    def test_get_all_trust_returns_dict(self):
        self.engine.register_node("n1")
        self.engine.register_node("n2")
        all_trust = self.engine.get_all_trust()
        assert "n1" in all_trust
        assert "n2" in all_trust

    def test_record_tracks_predictions(self):
        self.engine.update_prediction_accuracy("node_e", True, True)
        self.engine.update_prediction_accuracy("node_e", True, True)
        self.engine.update_prediction_accuracy("node_e", False, True)
        rec = self.engine.get_record("node_e")
        assert rec["total_predictions"] == 3
        assert rec["correct_predictions"] == 2
