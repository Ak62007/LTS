"""Synthetic validation for raw and reweighted FAST-LTS."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from src.taxi_lts.robust_lts import fit_reweighted_fast_lts


def contaminated_linear_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a known linear relationship with 10% response contamination."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(600, 3))
    true_coefficients = np.array([2.0, -1.5, 0.75])
    y = 5.0 + X @ true_coefficients + rng.normal(scale=0.2, size=len(X))
    contaminated = rng.choice(len(X), size=60, replace=False)
    y[contaminated] += 50.0
    return X, y, true_coefficients, contaminated


def test_lts_recovers_known_coefficients_better_than_ols() -> None:
    X, y, truth, _ = contaminated_linear_data()
    lts = fit_reweighted_fast_lts(X, y, n_initial_subsets=100)
    ols = LinearRegression().fit(X, y)

    lts_error = np.linalg.norm(lts.reweighted_model.coef_ - truth)
    ols_error = np.linalg.norm(ols.coef_ - truth)
    assert lts_error < 0.1
    assert lts_error < ols_error


def test_lts_downweights_planted_response_contamination() -> None:
    X, y, _, contaminated = contaminated_linear_data()
    lts = fit_reweighted_fast_lts(X, y, n_initial_subsets=100)

    assert (~lts.reweighting_weights[contaminated]).sum() >= 58
    assert len(lts.raw_estimator.best_h_subset) == 450


def test_lts_is_deterministic_for_fixed_seed() -> None:
    X, y, _, _ = contaminated_linear_data()
    first = fit_reweighted_fast_lts(X, y, n_initial_subsets=30)
    second = fit_reweighted_fast_lts(X, y, n_initial_subsets=30)

    np.testing.assert_allclose(
        first.reweighted_model.coef_, second.reweighted_model.coef_
    )
    np.testing.assert_array_equal(
        first.raw_estimator.best_h_subset, second.raw_estimator.best_h_subset
    )
