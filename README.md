# MeshGuard AI – Decentralized Fraud Intelligence Network

**A production-grade decentralized fraud detection network using federated learning, consensus scoring, and trust-based reputation.**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MeshGuard AI                                │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │  Node 1  │  │  Node 2  │  │  Node N  │  │  Malicious Node   │  │
│  │ (Bank A) │  │ (Bank B) │  │ (Fintech)│  │  (Adversarial)    │  │
│  │          │  │          │  │          │  │                   │  │
│  │ PyTorch  │  │ PyTorch  │  │ PyTorch  │  │  Noisy weights    │  │
│  │  FraudML │  │  FraudML │  │  FraudML │  │  + low fraud prob │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬──────────┘  │
│       └──────────────┴──────────────┴──────────────────┘           │
│                                    │                                │
│                         ┌──────────▼──────────┐                    │
│                         │    Coordinator       │                    │
│                         │  ┌───────────────┐  │                    │
│                         │  │  FedAvg (FL)  │  │                    │
│                         │  └───────────────┘  │                    │
│                         │  ┌───────────────┐  │                    │
│                         │  │Consensus Eng. │  │                    │
│                         │  └───────────────┘  │                    │
│                         │  ┌───────────────┐  │                    │
│                         │  │ Trust Engine  │  │                    │
│                         │  └───────────────┘  │                    │
│                         │  ┌───────────────┐  │                    │
│                         │  │ Graph Engine  │  │                    │
│                         │  └───────────────┘  │                    │
│                         └─────────────────────┘                    │
│                                                                     │
│           ┌────────────────────────────────────────┐               │
│           │         Streaming (Kafka / HTTP)        │               │
│           │  Producer → Coordinator → Consumer      │               │
│           └────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
meshguard-ai/
│
├── data/
│   ├── generate_data.py          # Synthetic fraud dataset generator
│   └── simulated/                # Per-node CSVs (gitignored, generated at runtime)
│
├── ml/
│   ├── models/
│   │   └── fraud_model.py        # PyTorch FraudMLP – fraud probability model
│   ├── training/
│   │   └── trainer.py            # NodeTrainer – local training + inference
│   └── federated/
│       └── fedavg.py             # Manual FedAvg + poisoning detection
│
├── services/
│   ├── node_service/
│   │   └── main.py               # FastAPI node (train / predict / weights)
│   ├── coordinator_service/
│   │   └── main.py               # FastAPI coordinator (FL + consensus + graph)
│   ├── consensus_engine/
│   │   └── __init__.py           # Weighted consensus + conflict resolution
│   ├── trust_engine/
│   │   └── __init__.py           # EWMA trust / reputation scores
│   └── graph_engine/
│       └── __init__.py           # Transaction graph + cluster detection
│
├── streaming/
│   ├── producer/
│   │   └── producer.py           # Transaction stream (HTTP / Kafka)
│   └── consumer/
│       └── consumer.py           # Kafka consumer → coordinator
│
├── infra/
│   ├── docker/
│   │   ├── Dockerfile.node
│   │   ├── Dockerfile.coordinator
│   │   └── Dockerfile.producer
│   ├── docker-compose.yml        # Full stack (7 nodes + malicious + coordinator + producer)
│   └── prometheus.yml            # Prometheus scrape config
│
├── tests/
│   ├── test_fedavg.py
│   ├── test_consensus.py
│   ├── test_trust_engine.py
│   ├── test_graph_engine.py
│   ├── test_model.py
│   └── test_data_generator.py
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Core Components

### 1. Fraud Detection Model (`ml/models/fraud_model.py`)

PyTorch feed-forward MLP with batch-normalisation and dropout:
- **Input:** 12 transaction features
- **Hidden:** `[64, 32, 16]` with BatchNorm + ReLU + Dropout(0.3)
- **Output:** sigmoid fraud probability in [0, 1]
- Weight serialisation helpers for federated exchange

### 2. Local Training (`ml/training/trainer.py`)

`NodeTrainer` class per node:
- Loads node-specific CSV
- Handles class imbalance via `BCEWithLogitsLoss(pos_weight=...)`
- `OneCycleLR` scheduler + gradient clipping
- `predict()` for single-transaction inference
- Incremental update: `set_weights()` + re-train

### 3. Federated Learning (`ml/federated/fedavg.py`)

Manual FedAvg implementation (no FL library dependency):
- **`aggregate()`** – sample-proportional + trust-weighted averaging of weight dicts
- **`detect_poisoning()`** – robust Z-score on L2 weight norms (MAD-based, 10% floor)
- **`compute_gradient_similarity()`** – cosine similarity for trust consistency updates

### 4. Consensus Engine (`services/consensus_engine/`)

Aggregates per-node fraud probabilities into a final score:
- **Simple average** – unweighted baseline
- **Weighted average** – proportional to trust score
- **Disagreement detection** – stddev threshold (default 0.20)
- **Conflict resolution** – majority vote → high-trust node subset → median fallback

### 5. Trust / Reputation Engine (`services/trust_engine/`)

EWMA-based per-node reputation (α=0.20):
- Updated on prediction accuracy vs consensus label
- Updated on weight consistency (cosine similarity vs global update)
- Hard penalty on poison detection
- Latency penalty for slow nodes
- Floor `MIN_TRUST=0.05` – nodes are never fully excluded

### 6. Graph Intelligence Engine (`services/graph_engine/`)

NetworkX transaction relationship graph:
- Edges on shared account / device / merchant + time proximity (≥0.2 weight threshold)
- Connected component analysis for cluster detection
- `graph_fraud_score` amplifies final risk when clustered transactions are flagged

### 7. Node Service (`services/node_service/main.py`)

FastAPI microservice per node (ports 8001–8008):

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/metrics` | GET | Node statistics |
| `/predict` | POST | Local fraud inference |
| `/weights` | GET | Export model weights (FL) |
| `/weights` | POST | Receive global weights from coordinator |
| `/train` | POST | Trigger local training round |

### 8. Coordinator Service (`services/coordinator_service/main.py`)

Central FastAPI service (port 8000):

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness + node list |
| `/nodes` | GET | Node registry + trust scores |
| `/predict` | POST | Full inference pipeline (fan-out → consensus → graph) |
| `/federated/round` | POST | Federated learning round |
| `/trust` | GET | All trust records |
| `/trust/{node_id}` | GET | Single node trust |
| `/graph/stats` | GET | Transaction graph statistics |

---

## Quick Start

### Prerequisites
- Docker ≥ 24 and Docker Compose v2

### Run the full stack

```bash
# Generate datasets
python data/generate_data.py --output-dir data/simulated --n-samples 5000

# Start all services
docker compose -f infra/docker-compose.yml up --build
```

Services:
| Service | URL |
|---|---|
| Coordinator | http://localhost:8000 |
| Node 1 | http://localhost:8001 |
| … (nodes 2-7) | http://localhost:8002–8007 |
| Malicious Node | http://localhost:8008 |

### Submit a transaction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 9500.00,
    "time_of_day": 2,
    "day_of_week": 6,
    "merchant_category": 15,
    "distance_from_home": 850.5,
    "transaction_count_1h": 12,
    "transaction_count_24h": 45,
    "avg_amount_30d": 200.0,
    "velocity_change": 3.5,
    "is_international": 1,
    "card_present": 0,
    "device_trust_score": 0.15
  }'
```

Example response:
```json
{
  "transaction_id": "a1b2c3d4-...",
  "local_scores": {
    "node_1": 0.87,
    "node_2": 0.91,
    "node_malicious": 0.03
  },
  "consensus_score": 0.82,
  "trust_adjusted_score": 0.89,
  "graph_score": 0.41,
  "final_score": 0.88,
  "is_fraud": true,
  "disagreement_score": 0.31,
  "conflict_detected": true,
  "resolution": "majority_fraud",
  "latency_ms": 42.1,
  "participating_nodes": ["node_1", "node_2", "node_3", "node_4", "node_5", "node_6", "node_7", "node_malicious"]
}
```

### Trigger a federated learning round

```bash
curl -X POST http://localhost:8000/federated/round
```

### View trust scores

```bash
curl http://localhost:8000/trust
```

### Start transaction stream

```bash
# HTTP mode (no Kafka required)
python -m streaming.producer.producer --mode http --tps 10 --target http://localhost:8000

# Kafka mode
docker compose -f infra/docker-compose.yml --profile kafka up -d
python -m streaming.producer.producer --mode kafka --tps 50
```

### Enable Prometheus monitoring

```bash
docker compose -f infra/docker-compose.yml --profile monitoring up -d
# Prometheus at http://localhost:9090
```

---

## Running Tests

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -v
```

Tests cover (39 test cases):
- **FedAvg** – weighted aggregation, sample-proportional weighting, trust weighting, poison detection
- **Consensus engine** – weighted average, disagreement detection, conflict resolution
- **Trust engine** – EWMA updates, poison flags, consistency tracking, floor enforcement
- **Graph engine** – edge creation, cluster detection, suspicious cluster flagging
- **Data generator** – fraud ratios, column completeness, null checks, malicious node distribution
- **Fraud model** – forward pass, output range, weight serialisation, save/load *(requires PyTorch)*

---

## Adversarial Resilience

The malicious node (`node_malicious`) implements two attack vectors:

1. **Model poisoning** – injects large Gaussian noise into weights before sending to coordinator
2. **Prediction suppression** – always returns near-zero fraud probability to avoid detection

The system defends via:
1. **Poison detection** (FedAvg layer) – MAD-based robust Z-score flags anomalous weight norms
2. **Trust degradation** – prediction divergence reduces trust via EWMA; down-weights in consensus
3. **Conflict resolution** – majority vote / high-trust subset overrides low-trust outlier

---

## System Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| ML framework | PyTorch MLP | Direct weight access for FL; no library constraints |
| FL algorithm | Manual FedAvg | Transparent, auditable, easily extended to FedProx |
| Consensus | Weighted avg + conflict resolution | Robust to single-node failures and adversaries |
| Trust | EWMA (α=0.20) | Smoothly decays old behaviour; recent actions matter more |
| Graph | NetworkX | Production-ready; drop-in PyTorch Geometric if GPU available |
| API | FastAPI + async | Native asyncio, Pydantic validation, built-in OpenAPI docs |
| Streaming | confluent-kafka + HTTP fallback | Full dev experience without Kafka dependency |
| Infra | Docker Compose | Single-command deployment of 10 services |

---

## Feature Set

| # | Requirement | Status |
|---|---|---|
| 1 | Multi-node simulation (5–10 nodes) | ✅ 7 normal + 1 malicious |
| 2 | Local PyTorch model with training | ✅ FraudMLP + NodeTrainer |
| 3 | Manual FedAvg implementation | ✅ ml/federated/fedavg.py |
| 4 | Consensus engine (weighted + conflict) | ✅ services/consensus_engine |
| 5 | Trust / reputation system | ✅ services/trust_engine |
| 6 | Adversarial node simulation | ✅ node_malicious + detection |
| 7 | Graph intelligence (NetworkX) | ✅ services/graph_engine |
| 8 | Real-time inference API `/predict` | ✅ coordinator + nodes |
| 9 | FastAPI microservices + REST | ✅ all services |
| 10 | Streaming (Kafka + HTTP fallback) | ✅ streaming/ |
| 11 | Docker + docker-compose | ✅ infra/ |
| 12 | Observability / Prometheus | ✅ infra/prometheus.yml |
| 13 | Tests | ✅ 39 test cases |
| 14 | Synthetic data generator | ✅ data/generate_data.py |

---

## License

Apache 2.0