"""Transparent raw and reweighted FAST-LTS regression helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from robpy.regression import FastLTSRegression
from scipy.stats import chi2, median_abs_deviation
from sklearn.linear_model import LinearRegression

from src.taxi_lts import RANDOM_SEED


@dataclass
class ReweightedLTSResult:
    """Fitted raw FAST-LTS model and its transparent reweighted estimate."""

    raw_estimator: FastLTSRegression
    reweighted_model: LinearRegression
    raw_residuals: np.ndarray
    raw_robust_scale: float
    raw_standardized_residuals: np.ndarray
    reweighting_weights: np.ndarray
    reweighting_cutoff: float


def fit_reweighted_fast_lts(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 0.75,
    n_initial_subsets: int = 500,
    random_state: int = RANDOM_SEED,
    reweighting_cutoff: float | None = None,
) -> ReweightedLTSResult:
    """Fit raw FAST-LTS and reweight using its MAD-standardized residuals.

    Built-in reweighting is disabled in favour of MAD scaling. The default
    cutoff is the square root of the chi-squared 0.975 quantile with one degree
    of freedom, approximately 2.24.
    """
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=float).reshape(-1)
    if X_array.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if len(X_array) != len(y_array):
        raise ValueError("X and y must contain the same number of observations.")

    cutoff = (
        float(np.sqrt(chi2.ppf(0.975, df=1)))
        if reweighting_cutoff is None
        else float(reweighting_cutoff)
    )
    raw_estimator = FastLTSRegression(
        alpha=alpha,
        n_initial_subsets=n_initial_subsets,
        reweighting=False,
        random_state=random_state,
    )
    raw_estimator.fit(X_array, y_array, verbosity=logging.WARNING)
    raw_predictions = raw_estimator.predict(X_array).reshape(-1)
    raw_residuals = y_array - raw_predictions
    raw_robust_scale = float(median_abs_deviation(raw_residuals, scale="normal"))
    if not np.isfinite(raw_robust_scale) or raw_robust_scale <= 0:
        raise ValueError("A positive finite robust residual scale is required.")

    standardized = raw_residuals / raw_robust_scale
    weights = np.abs(standardized) <= cutoff
    if int(weights.sum()) <= X_array.shape[1]:
        raise ValueError("Too few observations remain for reweighted regression.")
    reweighted_model = LinearRegression().fit(X_array[weights], y_array[weights])

    return ReweightedLTSResult(
        raw_estimator=raw_estimator,
        reweighted_model=reweighted_model,
        raw_residuals=raw_residuals,
        raw_robust_scale=raw_robust_scale,
        raw_standardized_residuals=standardized,
        reweighting_weights=weights,
        reweighting_cutoff=cutoff,
    )
