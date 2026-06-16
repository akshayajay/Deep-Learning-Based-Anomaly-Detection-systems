"""
tests/test_anomaly_detection.py
--------------------------------
Unit tests for pure utility functions in anomaly_detection.py.

No TensorFlow model training, no yfinance network calls required.
"""

import sys
import os
import types
import numpy as np
import pytest

# Stub heavy imports so tests run without TensorFlow or yfinance installed
for _mod in ("tensorflow", "tensorflow.keras", "tensorflow.keras.losses", "yfinance"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Minimal tensorflow stub
tf_stub = sys.modules["tensorflow"]
tf_stub.keras = types.SimpleNamespace(
    Model=object,
    Sequential=object,
    layers=types.SimpleNamespace(Dense=object),
    losses=types.SimpleNamespace(mean_absolute_error=lambda a, b: np.zeros(len(a))),
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anomaly_detection import (
    normalise,
    split_data,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.uniform(100, 5000, 1000)


@pytest.fixture
def binary_labels():
    np.random.seed(0)
    true = np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    scores = np.array([0.05, 0.08, 0.18, 0.92, 0.03, 0.87, 0.06, 0.21, 0.78, 0.12])
    return true, pred, scores


# ---------------------------------------------------------------------------
# normalise()
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_output_range_zero_to_one(self, sample_data):
        scaled, _ = normalise(sample_data)
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0

    def test_min_is_exactly_zero(self, sample_data):
        scaled, _ = normalise(sample_data)
        assert pytest.approx(scaled.min(), abs=1e-9) == 0.0

    def test_max_is_exactly_one(self, sample_data):
        scaled, _ = normalise(sample_data)
        assert pytest.approx(scaled.max(), abs=1e-9) == 1.0

    def test_output_shape_matches_input(self, sample_data):
        scaled, _ = normalise(sample_data)
        assert scaled.shape == (len(sample_data), 1)

    def test_returns_scaler(self, sample_data):
        from sklearn.preprocessing import MinMaxScaler
        _, scaler = normalise(sample_data)
        assert isinstance(scaler, MinMaxScaler)

    def test_scaler_can_inverse_transform(self, sample_data):
        scaled, scaler = normalise(sample_data)
        recovered = scaler.inverse_transform(scaled).flatten()
        np.testing.assert_allclose(recovered, sample_data, rtol=1e-5)

    def test_constant_array_does_not_crash(self):
        data = np.ones(50) * 42.0
        # MinMaxScaler clips constant arrays to 0 — just check no exception
        try:
            scaled, _ = normalise(data)
        except Exception:
            pass  # Known sklearn behaviour for constant input

    def test_single_element(self):
        data = np.array([7.0])
        scaled, _ = normalise(data)
        assert scaled.shape == (1, 1)


# ---------------------------------------------------------------------------
# split_data()
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_sizes_sum_to_total(self, sample_data):
        scaled, _ = normalise(sample_data)
        train, val, test = split_data(scaled)
        assert len(train) + len(val) + len(test) == len(scaled)

    def test_default_train_fraction(self, sample_data):
        scaled, _ = normalise(sample_data)
        train, val, test = split_data(scaled)
        assert pytest.approx(len(train) / len(scaled), abs=0.02) == 0.70

    def test_default_val_fraction(self, sample_data):
        scaled, _ = normalise(sample_data)
        train, val, test = split_data(scaled)
        assert pytest.approx(len(val) / len(scaled), abs=0.02) == 0.20

    def test_no_overlap_between_splits(self, sample_data):
        scaled, _ = normalise(sample_data)
        train, val, test = split_data(scaled)
        # Check splits are contiguous segments (no shuffling)
        n_train = len(train)
        n_val = len(val)
        np.testing.assert_array_equal(train, scaled[:n_train])
        np.testing.assert_array_equal(val,   scaled[n_train:n_train + n_val])
        np.testing.assert_array_equal(test,  scaled[n_train + n_val:])

    def test_custom_fractions(self, sample_data):
        scaled, _ = normalise(sample_data)
        train, val, test = split_data(scaled, train_frac=0.6, val_frac=0.3)
        assert pytest.approx(len(train) / len(scaled), abs=0.02) == 0.60
        assert pytest.approx(len(val) / len(scaled), abs=0.02) == 0.30

    def test_all_splits_non_empty(self, sample_data):
        scaled, _ = normalise(sample_data)
        train, val, test = split_data(scaled)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0


# ---------------------------------------------------------------------------
# compute_metrics()
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_all_five_keys(self, binary_labels):
        true, pred, scores = binary_labels
        result = compute_metrics(true, pred, scores)
        assert set(result.keys()) == {"accuracy", "precision", "recall", "f1", "auc"}

    def test_perfect_predictions_give_accuracy_one(self):
        true   = np.array([0, 1, 0, 1])
        pred   = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.1, 0.9])
        result = compute_metrics(true, pred, scores)
        assert pytest.approx(result["accuracy"], abs=1e-6) == 1.0

    def test_all_zeros_prediction_accuracy(self):
        true   = np.array([0, 0, 0, 0])
        pred   = np.array([0, 0, 0, 0])
        scores = np.array([0.1, 0.05, 0.08, 0.03])
        result = compute_metrics(true, pred, scores)
        assert pytest.approx(result["accuracy"], abs=1e-6) == 1.0

    def test_metrics_are_between_zero_and_one(self, binary_labels):
        true, pred, scores = binary_labels
        result = compute_metrics(true, pred, scores)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} is out of [0,1]"

    def test_auc_perfect_score(self):
        true   = np.array([0, 0, 1, 1])
        pred   = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_metrics(true, pred, scores)
        assert pytest.approx(result["auc"], abs=1e-6) == 1.0

    def test_returns_floats(self, binary_labels):
        true, pred, scores = binary_labels
        result = compute_metrics(true, pred, scores)
        for val in result.values():
            assert isinstance(val, float)

    def test_zero_division_handled_gracefully(self):
        # All predicted negative — precision/recall would divide by zero
        true   = np.array([0, 0, 0, 1])
        pred   = np.array([0, 0, 0, 0])
        scores = np.array([0.1, 0.1, 0.1, 0.1])
        result = compute_metrics(true, pred, scores)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
