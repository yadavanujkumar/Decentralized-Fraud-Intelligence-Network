"""
Fraud detection neural network model (PyTorch).
Architecture: Feed-forward MLP with batch-norm and dropout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_DIM = 12  # must match generate_data.py FEATURE_NAMES length


class FraudMLP(nn.Module):
    """
    Multi-layer perceptron for binary fraud classification.
    Returns a single sigmoid probability in [0, 1].
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, input_dim) → (B,)
        return torch.sigmoid(self.net(x)).squeeze(-1)

    # ------------------------------------------------------------------ #
    # Weight serialisation helpers (used by federated layer)              #
    # ------------------------------------------------------------------ #

    def get_weights(self) -> Dict[str, List]:
        """Return model weights as JSON-serialisable dict of lists."""
        return {
            k: v.cpu().numpy().tolist()
            for k, v in self.state_dict().items()
        }

    def set_weights(self, weights: Dict[str, List]) -> None:
        """Load weights from a dict of lists."""
        new_state = {
            k: torch.tensor(np.array(v), dtype=torch.float32)
            for k, v in weights.items()
        }
        self.load_state_dict(new_state, strict=True)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "FraudMLP":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model


def build_model(
    input_dim: int = FEATURE_DIM,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.3,
) -> FraudMLP:
    return FraudMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
