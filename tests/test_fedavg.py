"""
Tests for the FedAvg federated learning module.
"""
import numpy as np
import pytest
from ml.federated.fedavg import aggregate, detect_poisoning, compute_gradient_similarity


def _make_weights(values: list, keys=None) -> dict:
    """Helper: create a weight dict from flat values."""
    if keys is None:
        keys = ["layer_a", "layer_b"]
    n = len(values)
    half = n // 2
    return {
        keys[0]: values[:half],
        keys[1]: values[half:],
    }


class TestFedAvg:
    def test_simple_average(self):
        """Two equal nodes → result should be the average."""
        w1 = _make_weights([1.0, 1.0, 1.0, 1.0])
        w2 = _make_weights([3.0, 3.0, 3.0, 3.0])
        updates = [(w1, 100), (w2, 100)]
        result = aggregate(updates)
        for k in result:
            for v in result[k]:
                assert abs(v - 2.0) < 1e-6

    def test_weighted_by_samples(self):
        """More samples → higher contribution."""
        w1 = _make_weights([0.0, 0.0, 0.0, 0.0])
        w2 = _make_weights([4.0, 4.0, 4.0, 4.0])
        updates = [(w1, 100), (w2, 300)]  # w2 has 3× more data
        result = aggregate(updates)
        # Expected: (0*100 + 4*300) / 400 = 3.0
        for k in result:
            for v in result[k]:
                assert abs(v - 3.0) < 1e-6

    def test_trust_weighted(self):
        """Low-trust node is down-weighted."""
        w1 = _make_weights([0.0, 0.0, 0.0, 0.0])  # bad node
        w2 = _make_weights([2.0, 2.0, 2.0, 2.0])  # good node
        updates = [(w1, 100), (w2, 100)]
        trust = {"node_a": 0.1, "node_b": 0.9}
        result = aggregate(updates, trust_scores=trust, node_ids=["node_a", "node_b"])
        # Good node should dominate → result closer to 2.0
        for k in result:
            for v in result[k]:
                assert v > 1.5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate([])

    def test_single_node(self):
        w = _make_weights([5.0, 5.0, 5.0, 5.0])
        result = aggregate([(w, 200)])
        for k in result:
            for v in result[k]:
                assert abs(v - 5.0) < 1e-6


class TestPoisonDetection:
    def test_detects_malicious_node(self):
        """Node with very large weights should be flagged."""
        node_ids = ["n1", "n2", "n3", "malicious"]
        updates = [
            _make_weights([0.1, 0.2, 0.1, 0.2]),
            _make_weights([0.15, 0.18, 0.12, 0.22]),
            _make_weights([0.09, 0.21, 0.11, 0.19]),
            _make_weights([100.0, 200.0, 150.0, 180.0]),  # malicious
        ]
        flagged = detect_poisoning(updates, node_ids, threshold=2.0)
        assert "malicious" in flagged

    def test_clean_nodes_not_flagged(self):
        node_ids = ["n1", "n2", "n3"]
        updates = [
            _make_weights([0.1, 0.2, 0.1, 0.2]),
            _make_weights([0.15, 0.18, 0.12, 0.22]),
            _make_weights([0.09, 0.21, 0.11, 0.19]),
        ]
        flagged = detect_poisoning(updates, node_ids)
        assert flagged == []

    def test_insufficient_nodes_returns_empty(self):
        node_ids = ["n1", "n2"]
        updates = [_make_weights([1.0, 1.0, 1.0, 1.0])] * 2
        assert detect_poisoning(updates, node_ids) == []


class TestGradientSimilarity:
    def test_identical_weights_similarity_one(self):
        w = _make_weights([1.0, 2.0, 3.0, 4.0])
        sim = compute_gradient_similarity(w, w)
        assert abs(sim - 1.0) < 1e-6

    def test_opposite_weights_similarity_negative(self):
        wa = _make_weights([1.0, 1.0, 1.0, 1.0])
        wb = _make_weights([-1.0, -1.0, -1.0, -1.0])
        sim = compute_gradient_similarity(wa, wb)
        assert sim < 0

    def test_orthogonal_zero(self):
        wa = {"a": [1.0, 0.0]}
        wb = {"a": [0.0, 1.0]}
        sim = compute_gradient_similarity(wa, wb)
        assert abs(sim) < 1e-6
