"""
Transaction Consumer
====================
Consumes transactions from Kafka (or an HTTP queue) and forwards them
to the coordinator for consensus-based fraud prediction.

In development mode it calls the coordinator REST endpoint directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("consumer")

KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
KAFKA_GROUP = os.getenv("KAFKA_GROUP", "meshguard-consumers")
COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://coordinator:8000")


# ---------------------------------------------------------------------------
# Kafka consumer
# ---------------------------------------------------------------------------

class KafkaTransactionConsumer:
    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP,
        topic: str = KAFKA_TOPIC,
        group_id: str = KAFKA_GROUP,
    ):
        from confluent_kafka import Consumer, KafkaError  # type: ignore
        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
            }
        )
        self._consumer.subscribe([topic])
        self.KafkaError = KafkaError

    def poll(self, timeout: float = 1.0) -> Optional[Dict]:
        msg = self._consumer.poll(timeout)
        if msg is None:
            return None
        if msg.error():
            if msg.error().code() == self.KafkaError._PARTITION_EOF:
                return None
            raise RuntimeError(f"Kafka error: {msg.error()}")
        return json.loads(msg.value().decode())

    def close(self) -> None:
        self._consumer.close()


# ---------------------------------------------------------------------------
# Processing logic
# ---------------------------------------------------------------------------

def _process_transaction(tx: Dict, coordinator_url: str) -> None:
    import httpx
    payload = {k: v for k, v in tx.items() if not k.startswith("_")}
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(f"{coordinator_url}/predict", json=payload)
            result = r.json()
        final_score = result.get("final_score", "?")
        is_fraud = result.get("is_fraud", "?")
        logger.info(
            "Processed txn=%s  final_score=%.4f  is_fraud=%s",
            payload.get("transaction_id", "?")[:8],
            float(final_score) if isinstance(final_score, (int, float)) else 0,
            is_fraud,
        )
    except Exception as exc:
        logger.warning("Failed to process transaction: %s", exc)


# ---------------------------------------------------------------------------
# Consumer loop
# ---------------------------------------------------------------------------

def run_consumer(
    mode: str = "kafka",
    coordinator_url: str = COORDINATOR_URL,
    max_messages: Optional[int] = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger.info("Starting consumer  mode=%s", mode)

    if mode == "kafka":
        consumer: Any = KafkaTransactionConsumer()
        count = 0
        try:
            while True:
                msg = consumer.poll(timeout=1.0)
                if msg:
                    _process_transaction(msg, coordinator_url)
                    count += 1
                if max_messages and count >= max_messages:
                    break
        finally:
            consumer.close()
    else:
        # Dev mode – consumer is just a stub (production uses Kafka)
        logger.info("Running in dev mode – no Kafka needed; use the HTTP producer instead")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MeshGuard transaction consumer")
    parser.add_argument("--mode", choices=["kafka", "dev"], default="kafka")
    parser.add_argument("--coordinator", default=COORDINATOR_URL)
    parser.add_argument("--max-messages", type=int, default=None)
    args = parser.parse_args()
    run_consumer(mode=args.mode, coordinator_url=args.coordinator, max_messages=args.max_messages)
