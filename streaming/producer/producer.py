"""
Transaction Producer
====================
Simulates a real-time transaction stream.  In production this would publish
to Apache Kafka; here we provide a lightweight implementation that works with
both Kafka (via confluent-kafka) and a simple HTTP fallback for local dev.

Usage:
    python -m streaming.producer.producer --mode http --target http://coordinator:8000 --tps 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
import uuid
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("producer")

KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://coordinator:8000")


def _random_transaction() -> Dict[str, Any]:
    """Generate a random synthetic transaction."""
    is_fraud = random.random() < 0.05  # 5% fraud rate in stream
    rng = np.random.default_rng()

    if is_fraud:
        amount = float(rng.lognormal(6.0, 1.5))
        distance = float(rng.exponential(100))
        velocity_1h = int(rng.poisson(8))
        device_trust = float(rng.beta(2, 5))
        is_international = int(rng.binomial(1, 0.45))
        card_present = int(rng.binomial(1, 0.30))
    else:
        amount = float(rng.lognormal(4.5, 1.0))
        distance = float(rng.exponential(10))
        velocity_1h = int(rng.poisson(2))
        device_trust = float(rng.beta(5, 2))
        is_international = int(rng.binomial(1, 0.15))
        card_present = int(rng.binomial(1, 0.80))

    return {
        "transaction_id": str(uuid.uuid4()),
        "amount": round(amount, 2),
        "time_of_day": float(random.randint(0, 23)),
        "day_of_week": float(random.randint(0, 6)),
        "merchant_category": float(random.randint(0, 19)),
        "distance_from_home": round(distance, 2),
        "transaction_count_1h": float(velocity_1h),
        "transaction_count_24h": float(velocity_1h * random.randint(2, 12)),
        "avg_amount_30d": round(float(rng.lognormal(4.0, 0.8)), 2),
        "velocity_change": round(float(rng.normal(0, 1)), 4),
        "is_international": float(is_international),
        "card_present": float(card_present),
        "device_trust_score": round(float(np.clip(device_trust, 0, 1)), 4),
        "account_id": f"acc_{random.randint(1, 1000):04d}",
        "merchant_id": f"mer_{random.randint(1, 200):03d}",
        "device_id": f"dev_{random.randint(1, 500):04d}",
        "_ground_truth_fraud": is_fraud,
    }


# ---------------------------------------------------------------------------
# Kafka producer
# ---------------------------------------------------------------------------

class KafkaTransactionProducer:
    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP, topic: str = KAFKA_TOPIC):
        from confluent_kafka import Producer  # type: ignore
        self.topic = topic
        self._producer = Producer({"bootstrap.servers": bootstrap_servers})

    def send(self, transaction: Dict) -> None:
        self._producer.produce(
            self.topic,
            key=transaction["transaction_id"],
            value=json.dumps(transaction).encode(),
        )
        self._producer.poll(0)

    def flush(self) -> None:
        self._producer.flush()


# ---------------------------------------------------------------------------
# HTTP producer (fallback / dev mode)
# ---------------------------------------------------------------------------

class HttpTransactionProducer:
    def __init__(self, target_url: str = COORDINATOR_URL):
        import httpx
        self.target_url = target_url.rstrip("/")
        self._client = httpx.Client(timeout=10.0)

    def send(self, transaction: Dict) -> Dict:
        payload = {k: v for k, v in transaction.items() if not k.startswith("_")}
        r = self._client.post(f"{self.target_url}/predict", json=payload)
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Stream loop
# ---------------------------------------------------------------------------

def run_stream(
    mode: str = "http",
    tps: float = 5.0,
    duration_s: Optional[float] = None,
    target: str = COORDINATOR_URL,
) -> None:
    """Continuously emit transactions at `tps` transactions per second."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    interval = 1.0 / max(tps, 0.001)
    start = time.time()
    count = 0

    if mode == "kafka":
        producer: Any = KafkaTransactionProducer()
    else:
        producer = HttpTransactionProducer(target_url=target)

    logger.info("Starting transaction stream  mode=%s  tps=%.1f", mode, tps)

    try:
        while True:
            tx = _random_transaction()
            try:
                if mode == "kafka":
                    producer.send(tx)
                    result_str = "(sent to kafka)"
                else:
                    result = producer.send(tx)
                    result_str = f"final_score={result.get('final_score', '?'):.3f}"
                count += 1
                logger.info("tx #%d  id=%s  %s", count, tx["transaction_id"][:8], result_str)
            except Exception as exc:
                logger.warning("Failed to send tx: %s", exc)

            elapsed = time.time() - start
            if duration_s and elapsed >= duration_s:
                break
            time.sleep(interval)
    finally:
        if hasattr(producer, "flush"):
            producer.flush()
        if hasattr(producer, "close"):
            producer.close()
        logger.info("Stream ended after %d transactions", count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MeshGuard transaction stream producer")
    parser.add_argument("--mode", choices=["http", "kafka"], default="http")
    parser.add_argument("--target", default=COORDINATOR_URL)
    parser.add_argument("--tps", type=float, default=5.0, help="Transactions per second")
    parser.add_argument("--duration", type=float, default=None, help="Stop after N seconds")
    args = parser.parse_args()
    run_stream(mode=args.mode, tps=args.tps, duration_s=args.duration, target=args.target)
