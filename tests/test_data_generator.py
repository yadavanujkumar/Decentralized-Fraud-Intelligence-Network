"""
Tests for the synthetic data generator.
"""
import os
import tempfile
import pandas as pd
import pytest
from data.generate_data import generate_all, NODE_CONFIGS, FEATURE_NAMES


class TestDataGeneration:
    def test_generates_all_nodes(self, tmp_path):
        generate_all(str(tmp_path), n_samples=200)
        for node_id in NODE_CONFIGS:
            path = tmp_path / f"{node_id}.csv"
            assert path.exists(), f"Missing CSV for {node_id}"

    def test_correct_columns(self, tmp_path):
        generate_all(str(tmp_path), n_samples=100)
        df = pd.read_csv(tmp_path / "node_1.csv")
        for col in FEATURE_NAMES:
            assert col in df.columns, f"Missing column: {col}"
        assert "is_fraud" in df.columns

    def test_fraud_ratio_approximately_correct(self, tmp_path):
        generate_all(str(tmp_path), n_samples=1000)
        for node_id, cfg in NODE_CONFIGS.items():
            df = pd.read_csv(tmp_path / f"{node_id}.csv")
            actual_ratio = df["is_fraud"].mean()
            expected = cfg["fraud_ratio"]
            # Allow ±5% tolerance
            assert abs(actual_ratio - expected) < 0.05, (
                f"{node_id}: expected fraud_ratio={expected:.2f}, got {actual_ratio:.2f}"
            )

    def test_sample_count(self, tmp_path):
        n = 300
        generate_all(str(tmp_path), n_samples=n)
        df = pd.read_csv(tmp_path / "node_1.csv")
        assert len(df) == n

    def test_no_nulls(self, tmp_path):
        generate_all(str(tmp_path), n_samples=200)
        df = pd.read_csv(tmp_path / "node_1.csv")
        assert df.isnull().sum().sum() == 0

    def test_malicious_node_high_fraud(self, tmp_path):
        generate_all(str(tmp_path), n_samples=500)
        df = pd.read_csv(tmp_path / "node_malicious.csv")
        assert df["is_fraud"].mean() > 0.3  # malicious node has ~50% fraud
