"""
Synthetic fraud-detection dataset generator.
Each node gets its own CSV with different fraud ratios and feature distributions
to simulate realistic non-IID federated learning conditions.
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

FEATURE_NAMES = [
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

# Per-node configuration: (fraud_ratio, amount_mean_multiplier, velocity_multiplier)
NODE_CONFIGS = {
    "node_1": {"fraud_ratio": 0.03, "amount_scale": 1.0, "velocity_scale": 1.0},
    "node_2": {"fraud_ratio": 0.05, "amount_scale": 1.5, "velocity_scale": 0.8},
    "node_3": {"fraud_ratio": 0.08, "amount_scale": 0.7, "velocity_scale": 1.3},
    "node_4": {"fraud_ratio": 0.10, "amount_scale": 2.0, "velocity_scale": 0.5},
    "node_5": {"fraud_ratio": 0.04, "amount_scale": 1.2, "velocity_scale": 1.1},
    "node_6": {"fraud_ratio": 0.06, "amount_scale": 0.9, "velocity_scale": 1.4},
    "node_7": {"fraud_ratio": 0.12, "amount_scale": 1.8, "velocity_scale": 0.6},
    # malicious node – skewed distribution
    "node_malicious": {"fraud_ratio": 0.50, "amount_scale": 5.0, "velocity_scale": 3.0},
}


def _generate_node_data(
    node_id: str,
    n_samples: int,
    fraud_ratio: float,
    amount_scale: float,
    velocity_scale: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    def base_row(label: int) -> dict:
        if label == 0:
            amount = rng.lognormal(mean=4.5, sigma=1.0) * amount_scale
            velocity = rng.poisson(2) * velocity_scale
            dist = rng.exponential(10)
            device_trust = rng.beta(5, 2)
        else:
            amount = rng.lognormal(mean=6.0, sigma=1.5) * amount_scale
            velocity = rng.poisson(8) * velocity_scale
            dist = rng.exponential(100)
            device_trust = rng.beta(2, 5)

        return {
            "amount": round(float(amount), 2),
            "time_of_day": float(rng.randint(0, 24)),
            "day_of_week": float(rng.randint(0, 7)),
            "merchant_category": float(rng.randint(0, 20)),
            "distance_from_home": round(float(dist), 2),
            "transaction_count_1h": float(max(0, int(velocity))),
            "transaction_count_24h": float(max(0, int(velocity * rng.uniform(2, 12)))),
            "avg_amount_30d": round(float(rng.lognormal(mean=4.0, sigma=0.8) * amount_scale), 2),
            "velocity_change": round(float(rng.normal(0, 1) * velocity_scale), 4),
            "is_international": float(rng.binomial(1, 0.15 if label == 0 else 0.45)),
            "card_present": float(rng.binomial(1, 0.80 if label == 0 else 0.30)),
            "device_trust_score": round(float(np.clip(device_trust, 0, 1)), 4),
            "is_fraud": label,
        }

    rows = [base_row(0) for _ in range(n_legit)] + [base_row(1) for _ in range(n_fraud)]
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["node_id"] = node_id
    return df


def generate_all(output_dir: str, n_samples: int = 5000) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for idx, (node_id, cfg) in enumerate(NODE_CONFIGS.items()):
        df = _generate_node_data(
            node_id=node_id,
            n_samples=n_samples,
            fraud_ratio=cfg["fraud_ratio"],
            amount_scale=cfg["amount_scale"],
            velocity_scale=cfg["velocity_scale"],
            seed=idx * 42,
        )
        path = out / f"{node_id}.csv"
        df.to_csv(path, index=False)
        print(f"Generated {len(df)} rows for {node_id} → {path}  (fraud={cfg['fraud_ratio']*100:.0f}%)")

    print(f"\nAll datasets written to {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic fraud datasets")
    parser.add_argument("--output-dir", default="data/simulated", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=5000, help="Samples per node")
    args = parser.parse_args()
    generate_all(args.output_dir, args.n_samples)
