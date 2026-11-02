"""Tests for the protected clean model-comparison workflow."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.taxi_lts.run_analysis import original_unit_parameters, prediction_metrics


def test_original_unit_parameters_preserve_predictions() -> None:
    """Back-transformed parameters must reproduce standardized predictions."""
    X = np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    scaler = StandardScaler().fit(X)
    standardized_coefficients = np.array([2.5, -1.25])
    standardized_intercept = 7.0
    coefficients, intercept = original_unit_parameters(
        scaler, standardized_coefficients, standardized_intercept
    )
    standardized_prediction = (
        scaler.transform(X) @ standardized_coefficients + standardized_intercept
    )
    original_prediction = X @ coefficients + intercept
    np.testing.assert_allclose(standardized_prediction, original_prediction)


def test_prediction_metrics_are_exact_for_perfect_predictions() -> None:
    """Every exported metric should recognize a perfect prediction."""
    truth = np.array([2.0, 4.0, 8.0])
    metrics = prediction_metrics(truth, truth.copy())
    assert metrics == {
        "mae": 0.0,
        "rmse": 0.0,
        "median_absolute_error": 0.0,
        "r_squared": 1.0,
    }
