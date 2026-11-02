"""Validate FAST-LTS against a known contaminated linear relationship."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.taxi_lts import RANDOM_SEED
from src.taxi_lts.prepare_data import PROJECT_ROOT
from src.taxi_lts.robust_lts import fit_reweighted_fast_lts


def run_validation() -> dict[str, int | float]:
    """Return coefficient recovery and detection evidence with known truth."""
    rng = np.random.default_rng(RANDOM_SEED)
    n_observations = 1_000
    true_intercept = 5.0
    true_coefficients = np.array([2.0, -1.5, 0.75])
    X = rng.normal(size=(n_observations, len(true_coefficients)))
    clean_y = (
        true_intercept
        + X @ true_coefficients
        + rng.normal(scale=0.2, size=n_observations)
    )
    contaminated_indices = rng.choice(n_observations, size=100, replace=False)
    y = clean_y.copy()
    y[contaminated_indices] += 50.0

    lts = fit_reweighted_fast_lts(X, y, n_initial_subsets=500)
    ols = LinearRegression().fit(X, y)
    flagged = ~lts.reweighting_weights
    known_contamination = np.zeros(n_observations, dtype=bool)
    known_contamination[contaminated_indices] = True
    true_positive = int(np.sum(flagged & known_contamination))
    false_positive = int(np.sum(flagged & ~known_contamination))
    false_negative = int(np.sum(~flagged & known_contamination))

    row: dict[str, int | float] = {
        "n_observations": n_observations,
        "n_predictors": len(true_coefficients),
        "n_contaminated": len(contaminated_indices),
        "contamination_added": 50.0,
        "alpha": 0.75,
        "h": len(lts.raw_estimator.best_h_subset),
        "n_initial_subsets": 500,
        "true_intercept": true_intercept,
        "ols_intercept": float(ols.intercept_),
        "raw_lts_intercept": float(lts.raw_estimator.model.intercept_[0]),
        "reweighted_lts_intercept": float(lts.reweighted_model.intercept_),
        "ols_coefficient_l2_error": float(
            np.linalg.norm(ols.coef_ - true_coefficients)
        ),
        "raw_lts_coefficient_l2_error": float(
            np.linalg.norm(
                lts.raw_estimator.model.coef_.reshape(-1) - true_coefficients
            )
        ),
        "reweighted_lts_coefficient_l2_error": float(
            np.linalg.norm(lts.reweighted_model.coef_ - true_coefficients)
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "detection_recall": true_positive / len(contaminated_indices),
        "detection_precision": true_positive / (true_positive + false_positive),
        "raw_mad_scale": lts.raw_robust_scale,
        "reweighting_cutoff": lts.reweighting_cutoff,
    }
    for index, truth in enumerate(true_coefficients, start=1):
        row[f"true_beta_{index}"] = float(truth)
        row[f"ols_beta_{index}"] = float(ols.coef_[index - 1])
        row[f"raw_lts_beta_{index}"] = float(
            lts.raw_estimator.model.coef_.reshape(-1)[index - 1]
        )
        row[f"reweighted_lts_beta_{index}"] = float(
            lts.reweighted_model.coef_[index - 1]
        )
    return row


def parse_args() -> argparse.Namespace:
    """Parse the validation output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tables" / "lts_synthetic_validation.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Run the validation and write its machine-readable evidence."""
    args = parse_args()
    result = run_validation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(
        args.output, index=False, float_format="%.10f", lineterminator="\n"
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
