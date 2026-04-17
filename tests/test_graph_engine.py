"""
Tests for the graph intelligence engine.
"""
import time
import pytest
from services.graph_engine import TransactionGraph, Transaction


def _make_tx(tx_id, account_id=None, merchant_id=None, device_id=None,
             amount=100.0, fraud_prob=0.1, ts=None):
    return Transaction(
        transaction_id=tx_id,
        account_id=account_id,
        merchant_id=merchant_id,
        device_id=device_id,
        amount=amount,
        timestamp=ts or time.time(),
        fraud_prob=fraud_prob,
    )


class TestTransactionGraph:
    def setup_method(self):
        self.graph = TransactionGraph(max_nodes=500)

    def test_add_transaction(self):
        tx = _make_tx("t1")
        self.graph.add_transaction(tx)
        assert "t1" in self.graph.G

    def test_same_account_creates_edge(self):
        t0 = time.time()
        tx1 = _make_tx("t1", account_id="acc_1", ts=t0)
        tx2 = _make_tx("t2", account_id="acc_1", ts=t0 + 10)
        self.graph.add_transaction(tx1)
        self.graph.add_transaction(tx2)
        assert self.graph.G.has_edge("t1", "t2")

    def test_different_accounts_no_edge(self):
        t0 = time.time()
        tx1 = _make_tx("t3", account_id="acc_10", merchant_id="m_x", ts=t0)
        tx2 = _make_tx("t4", account_id="acc_20", merchant_id="m_y", ts=t0 + 10000)
        self.graph.add_transaction(tx1)
        self.graph.add_transaction(tx2)
        assert not self.graph.G.has_edge("t3", "t4")

    def test_analyse_unknown_transaction(self):
        result = self.graph.analyse("unknown_tx")
        assert result.graph_fraud_score == 0.0
        assert result.connected_component_size == 1

    def test_suspicious_cluster_detected(self):
        t0 = time.time()
        for i in range(5):
            tx = _make_tx(f"cluster_{i}", account_id="acc_sus", ts=t0 + i * 5, fraud_prob=0.9)
            self.graph.add_transaction(tx)
        result = self.graph.analyse("cluster_0")
        assert result.suspicious_cluster

    def test_graph_stats(self):
        tx = _make_tx("stats_tx")
        self.graph.add_transaction(tx)
        stats = self.graph.get_stats()
        assert "total_nodes" in stats
        assert stats["total_nodes"] >= 1
