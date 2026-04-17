"""
Graph Intelligence Engine
=========================
Builds a transaction graph and detects suspicious clusters / connected
components that exhibit shared fraud patterns.

Nodes  = transaction IDs (or account IDs when available)
Edges  = shared features: same merchant, same device, close amounts & timing

Uses networkx for graph construction + community detection.
Optionally exposes PyTorch Geometric integration hooks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Edge weight thresholds
AMOUNT_SIMILARITY_PCT = 0.05   # within 5% => link
TIME_WINDOW_SECONDS = 300       # within 5 min => link
# Minimum combined edge weight required to create a relationship between two
# transactions.  Prevents spurious links from amount/time coincidence alone.
MIN_EDGE_WEIGHT = 0.2


@dataclass
class Transaction:
    transaction_id: str
    account_id: Optional[str]
    merchant_id: Optional[str]
    device_id: Optional[str]
    amount: float
    timestamp: float       # unix epoch seconds
    fraud_prob: float = 0.0
    is_fraud: Optional[bool] = None
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class GraphAnalysisResult:
    transaction_id: str
    connected_component_size: int
    cluster_fraud_ratio: float
    graph_fraud_score: float       # [0,1] – graph-based risk boost
    suspicious_cluster: bool
    neighbors: List[str]

    def to_dict(self) -> Dict:
        return {
            "transaction_id": self.transaction_id,
            "component_size": self.connected_component_size,
            "cluster_fraud_ratio": round(self.cluster_fraud_ratio, 4),
            "graph_fraud_score": round(self.graph_fraud_score, 4),
            "suspicious_cluster": self.suspicious_cluster,
            "neighbors": self.neighbors,
        }


class TransactionGraph:
    """In-memory transaction relationship graph."""

    def __init__(self, max_nodes: int = 10_000) -> None:
        self.G = nx.Graph()
        self.max_nodes = max_nodes
        self._transactions: Dict[str, Transaction] = {}

    # ------------------------------------------------------------------ #
    # Graph construction                                                  #
    # ------------------------------------------------------------------ #

    def add_transaction(self, tx: Transaction) -> None:
        """Add a transaction node and create edges to similar transactions."""
        # Prune if too large
        if len(self.G) >= self.max_nodes:
            oldest = sorted(
                self._transactions.keys(),
                key=lambda k: self._transactions[k].timestamp,
            )[:100]
            for tid in oldest:
                self.G.remove_node(tid)
                del self._transactions[tid]

        self.G.add_node(
            tx.transaction_id,
            fraud_prob=tx.fraud_prob,
            amount=tx.amount,
            timestamp=tx.timestamp,
            account_id=tx.account_id or "",
            merchant_id=tx.merchant_id or "",
            device_id=tx.device_id or "",
        )
        self._transactions[tx.transaction_id] = tx
        self._create_edges(tx)

    def _create_edges(self, tx: Transaction) -> None:
        for other_id, other in self._transactions.items():
            if other_id == tx.transaction_id:
                continue
            weight = self._edge_weight(tx, other)
            # Only create an edge when there is at least one strong shared
            # feature (account / device / merchant hit).  Pure time/amount
            # proximity alone is insufficient to establish a relationship.
            if weight >= MIN_EDGE_WEIGHT:
                self.G.add_edge(tx.transaction_id, other_id, weight=weight)

    @staticmethod
    def _edge_weight(tx_a: Transaction, tx_b: Transaction) -> float:
        score = 0.0

        # Same account
        if tx_a.account_id and tx_a.account_id == tx_b.account_id:
            score += 0.4

        # Same merchant
        if tx_a.merchant_id and tx_a.merchant_id == tx_b.merchant_id:
            score += 0.2

        # Same device
        if tx_a.device_id and tx_a.device_id == tx_b.device_id:
            score += 0.3

        # Similar amount (within AMOUNT_SIMILARITY_PCT %)
        avg_amount = (tx_a.amount + tx_b.amount) / 2.0
        if avg_amount > 0:
            pct_diff = abs(tx_a.amount - tx_b.amount) / avg_amount
            if pct_diff <= AMOUNT_SIMILARITY_PCT:
                score += 0.1

        # Time proximity
        time_diff = abs(tx_a.timestamp - tx_b.timestamp)
        if time_diff <= TIME_WINDOW_SECONDS:
            score += 0.1 * (1.0 - time_diff / TIME_WINDOW_SECONDS)

        return round(score, 4)

    # ------------------------------------------------------------------ #
    # Analysis                                                            #
    # ------------------------------------------------------------------ #

    def analyse(self, transaction_id: str) -> GraphAnalysisResult:
        """Return graph-based risk analysis for a given transaction."""
        if transaction_id not in self.G:
            return GraphAnalysisResult(
                transaction_id=transaction_id,
                connected_component_size=1,
                cluster_fraud_ratio=0.0,
                graph_fraud_score=0.0,
                suspicious_cluster=False,
                neighbors=[],
            )

        # Connected component
        component: Set[str] = nx.node_connected_component(self.G, transaction_id)
        comp_size = len(component)

        # Fraud ratio in component
        fraud_probs = [
            self.G.nodes[n].get("fraud_prob", 0.0) for n in component
        ]
        cluster_fraud_ratio = float(np.mean(fraud_probs)) if fraud_probs else 0.0

        # Graph fraud score – amplify if cluster is large and fraudulent
        size_factor = min(1.0, comp_size / 10.0)
        graph_fraud_score = float(np.clip(cluster_fraud_ratio * (0.5 + 0.5 * size_factor), 0, 1))

        suspicious = cluster_fraud_ratio > 0.4 and comp_size >= 3

        neighbors = [n for n in self.G.neighbors(transaction_id)]

        if suspicious:
            logger.warning(
                "Suspicious cluster: txn=%s  cluster_size=%d  fraud_ratio=%.2f",
                transaction_id,
                comp_size,
                cluster_fraud_ratio,
            )

        return GraphAnalysisResult(
            transaction_id=transaction_id,
            connected_component_size=comp_size,
            cluster_fraud_ratio=cluster_fraud_ratio,
            graph_fraud_score=graph_fraud_score,
            suspicious_cluster=suspicious,
            neighbors=neighbors[:10],  # cap for API responses
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "connected_components": nx.number_connected_components(self.G),
            "density": round(nx.density(self.G), 6),
        }


# ------------------------------------------------------------------ #
# Singleton                                                           #
# ------------------------------------------------------------------ #

_graph: Optional[TransactionGraph] = None


def get_graph() -> TransactionGraph:
    global _graph
    if _graph is None:
        _graph = TransactionGraph()
    return _graph
