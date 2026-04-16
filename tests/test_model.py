"""
Tests for the ML model and local trainer.
"""
import os
import tempfile
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed – skipping model tests")
from ml.models.fraud_model import FraudMLP, FEATURE_DIM, build_model


class TestFraudMLP:
    def test_forward_shape(self):
        model = build_model()
        model.eval()
        x = torch.randn(8, FEATURE_DIM)
        out = model(x)
        assert out.shape == (8,)

    def test_output_in_zero_one(self):
        model = build_model()
        model.eval()
        x = torch.randn(100, FEATURE_DIM)
        out = model(x)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_get_set_weights_roundtrip(self):
        model = build_model()
        original_weights = model.get_weights()
        # Modify
        model2 = build_model()
        model2.set_weights(original_weights)
        # Both should produce same output
        model.eval()
        model2.eval()
        x = torch.randn(4, FEATURE_DIM)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_save_load(self):
        model = build_model()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = FraudMLP.load(path)
            model.eval()
            loaded.eval()
            x = torch.randn(4, FEATURE_DIM)
            with torch.no_grad():
                out1 = model(x)
                out2 = loaded(x)
            assert torch.allclose(out1, out2, atol=1e-6)
        finally:
            os.unlink(path)

    def test_custom_hidden_dims(self):
        model = FraudMLP(input_dim=FEATURE_DIM, hidden_dims=[128, 64, 32])
        x = torch.randn(2, FEATURE_DIM)
        out = model(x)
        assert out.shape == (2,)


class TestNodeTrainer:
    """Integration-light tests that use the trainer with a tiny in-memory dataset."""

    @pytest.fixture
    def csv_file(self, tmp_path):
        import pandas as pd
        from ml.training.trainer import FEATURE_COLS, LABEL_COL
        rows = []
        np.random.seed(0)
        for i in range(200):
            row = {col: float(np.random.randn()) for col in FEATURE_COLS}
            row[LABEL_COL] = int(i < 20)  # 10% fraud
            rows.append(row)
        df = pd.DataFrame(rows)
        path = str(tmp_path / "test_data.csv")
        df.to_csv(path, index=False)
        return path

    def test_train_returns_metrics(self, csv_file):
        from ml.training.trainer import NodeTrainer
        trainer = NodeTrainer(node_id="test_node", data_path=csv_file)
        metrics = trainer.train(epochs=2)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_predict_returns_float_in_range(self, csv_file):
        from ml.training.trainer import NodeTrainer, FEATURE_COLS
        trainer = NodeTrainer(node_id="test_node", data_path=csv_file)
        trainer.train(epochs=1)
        features = {col: 0.0 for col in FEATURE_COLS}
        prob = trainer.predict(features)
        assert 0.0 <= prob <= 1.0

    def test_get_set_weights(self, csv_file):
        from ml.training.trainer import NodeTrainer
        t1 = NodeTrainer(node_id="n1", data_path=csv_file)
        t2 = NodeTrainer(node_id="n2", data_path=csv_file)
        t1.train(epochs=1)
        weights = t1.get_weights()
        t2.set_weights(weights)
        assert set(t1.get_weights().keys()) == set(t2.get_weights().keys())
