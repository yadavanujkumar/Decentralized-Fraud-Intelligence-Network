"""
Federated Averaging (FedAvg) — manual implementation.

Paper: McMahan et al., 2017 – "Communication-Efficient Learning of Deep
Networks from Decentralized Data".

The coordinator calls `aggregate()` with a list of (weights, n_samples)
tuples from participating nodes and returns the globally aggregated weights.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

WeightDict = Dict[str, List]


def aggregate(
    node_updates: List[Tuple[WeightDict, int]],
    trust_scores: Optional[Dict[str, float]] = None,
    node_ids: Optional[List[str]] = None,
) -> WeightDict:
    """
    Weighted FedAvg over node weight dicts.

    Args:
        node_updates: list of (weights_dict, n_samples) per node
        trust_scores:  optional {node_id: trust_score}; combined with
                       n_samples to compute final weights
        node_ids:      list of node IDs matching node_updates order

    Returns:
        aggregated_weights: dict matching the weight structure of a single node
    """
    if not node_updates:
        raise ValueError("No node updates provided")

    total_samples = sum(n for _, n in node_updates)
    n_nodes = len(node_updates)

    # Build per-node scalar weight (n_samples-proportional × trust)
    scalar_weights: List[float] = []
    for i, (_, n) in enumerate(node_updates):
        sample_w = n / max(total_samples, 1)
        trust_w = 1.0
        if trust_scores and node_ids and i < len(node_ids):
            trust_w = trust_scores.get(node_ids[i], 1.0)
        scalar_weights.append(sample_w * trust_w)

    # Re-normalise so weights sum to 1
    w_sum = sum(scalar_weights)
    if w_sum <= 0:
        # Fallback to uniform
        scalar_weights = [1.0 / n_nodes] * n_nodes
    else:
        scalar_weights = [w / w_sum for w in scalar_weights]

    logger.debug("FedAvg scalar weights: %s", scalar_weights)

    # Aggregate layer-by-layer
    reference_weights, _ = node_updates[0]
    aggregated: WeightDict = {}

    for key in reference_weights:
        arrays = [np.array(update[key], dtype=np.float64) for update, _ in node_updates]
        agg = sum(w * a for w, a in zip(scalar_weights, arrays))
        aggregated[key] = agg.tolist()

    return aggregated


def detect_poisoning(
    node_updates: List[WeightDict],
    node_ids: List[str],
    threshold: float = 3.0,
) -> List[str]:
    """
    Detect potential gradient-poisoning / model-poisoning attacks.

    Strategy: compute per-layer L2 norms, then flag nodes whose norm
    is > `threshold` standard deviations above the median.

    Returns list of suspected malicious node IDs.
    """
    if len(node_updates) < 3:
        return []

    # Compute overall L2 norm for each node's weights
    norms: List[float] = []
    for weights in node_updates:
        total = 0.0
        for v in weights.values():
            arr = np.array(v, dtype=np.float64).ravel()
            total += float(np.dot(arr, arr))
        norms.append(float(np.sqrt(total)))

    norms_arr = np.array(norms)
    median = float(np.median(norms_arr))
    # Ensure MAD is at least 10% of the median to avoid false positives on
    # tightly clustered norms where tiny absolute differences appear extreme.
    raw_mad = float(np.median(np.abs(norms_arr - median)))
    mad = max(raw_mad, median * 0.10, 1e-9)

    flagged = []
    for i, (node_id, norm) in enumerate(zip(node_ids, norms)):
        # 1.4826 scales MAD to be consistent with the standard deviation for a
        # normal distribution (MAD ≈ σ/1.4826), giving a robust z-score that
        # is comparable to the classical z-score but resistant to outliers.
        z = abs(norm - median) / (mad * 1.4826)  # robust z-score
        if z > threshold:
            flagged.append(node_id)
            logger.warning(
                "Potential poisoning detected: node=%s  norm=%.4f  z=%.2f",
                node_id,
                norm,
                z,
            )

    return flagged


def compute_gradient_similarity(
    weights_a: WeightDict,
    weights_b: WeightDict,
) -> float:
    """
    Cosine similarity between two flat weight vectors.
    Used by the trust engine to assess consistency.
    """
    vec_a = np.concatenate([np.array(v, dtype=np.float64).ravel() for v in weights_a.values()])
    vec_b = np.concatenate([np.array(v, dtype=np.float64).ravel() for v in weights_b.values()])

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
