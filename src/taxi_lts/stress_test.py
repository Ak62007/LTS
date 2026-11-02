"""Run the controlled 5% +$100 training-response contamination experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.taxi_lts import RANDOM_SEED
from src.taxi_lts.explore_training import PREDICTORS, TARGET
from src.taxi_lts.prepare_data import PROJECT_ROOT, sha256_file, write_csv
from src.taxi_lts.robust_lts import ReweightedLTSResult, fit_reweighted_fast_lts
from src.taxi_lts.run_analysis import (
    AUDIT_STANDARDIZED_RESIDUAL_CUTOFF,
    N_INITIAL_SUBSETS,
    PRIMARY_ALPHA,
    load_protected_split,
    original_unit_parameters,
    prediction_metrics,
)

CONTAMINATION_FRACTION = 0.05
CONTAMINATION_ADDITION = 100.0


def contamination_positions(
    n_observations: int,
    *,
    fraction: float = CONTAMINATION_FRACTION,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Select a deterministic response-contamination subset."""
    count = int(n_observations * fraction)
    if count <= 0 or count >= n_observations:
        raise ValueError("Contamination count must be between zero and n.")
    return np.sort(
        np.random.default_rng(seed).choice(n_observations, size=count, replace=False)
    )


def contaminate_training_response(
    training: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return contaminated fares and a known-contamination mask."""
    original = training[TARGET].to_numpy(dtype=float)
    contaminated = original.copy()
    positions = contamination_positions(len(training))
    contaminated[positions] += CONTAMINATION_ADDITION
    mask = np.zeros(len(training), dtype=bool)
    mask[positions] = True
    return contaminated, mask


def fit_pair(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[LinearRegression, ReweightedLTSResult]:
    """Fit OLS and primary LTS to one training response vector."""
    ols = LinearRegression().fit(X_train, y_train)
    lts = fit_reweighted_fast_lts(
        X_train,
        y_train,
        alpha=PRIMARY_ALPHA,
        n_initial_subsets=N_INITIAL_SUBSETS,
        random_state=RANDOM_SEED,
    )
    return ols, lts


def parameter_rows(
    scaler: StandardScaler,
    scenario: str,
    ols: LinearRegression,
    lts: ReweightedLTSResult,
) -> list[dict[str, float | str]]:
    """Return original-unit OLS and reweighted-LTS coefficients."""
    rows: list[dict[str, float | str]] = []
    models = [
        ("OLS", ols.coef_, float(ols.intercept_)),
        (
            "Reweighted LTS",
            lts.reweighted_model.coef_,
            float(lts.reweighted_model.intercept_),
        ),
    ]
    for estimator, coefficients, intercept in models:
        original_coef, original_intercept = original_unit_parameters(
            scaler, np.asarray(coefficients), intercept
        )
        for term, value in zip(
            ["const", *PREDICTORS],
            [original_intercept, *original_coef],
            strict=True,
        ):
            rows.append(
                {
                    "scenario": scenario,
                    "estimator": estimator,
                    "term": term,
                    "coefficient_original_units": float(value),
                }
            )
    return rows


def comparison_tables(
    scaler: StandardScaler,
    test: pd.DataFrame,
    clean_ols: LinearRegression,
    clean_lts: ReweightedLTSResult,
    contaminated_ols: LinearRegression,
    contaminated_lts: ReweightedLTSResult,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare clean and contaminated fits on the same clean test set."""
    X_test = scaler.transform(test[PREDICTORS].to_numpy(dtype=float))
    y_test = test[TARGET].to_numpy(dtype=float)
    models = [
        ("clean", "OLS", clean_ols.predict(X_test)),
        ("clean", "Reweighted LTS", clean_lts.reweighted_model.predict(X_test)),
        ("contaminated", "OLS", contaminated_ols.predict(X_test)),
        (
            "contaminated",
            "Reweighted LTS",
            contaminated_lts.reweighted_model.predict(X_test),
        ),
    ]
    performance = pd.DataFrame(
        [
            {
                "training_scenario": scenario,
                "estimator": estimator,
                **prediction_metrics(y_test, prediction),
            }
            for scenario, estimator, prediction in models
        ]
    )

    coefficient_rows = parameter_rows(
        scaler, "clean", clean_ols, clean_lts
    ) + parameter_rows(scaler, "contaminated", contaminated_ols, contaminated_lts)
    coefficients = pd.DataFrame(coefficient_rows)
    pivot = coefficients.pivot(
        index=["estimator", "term"],
        columns="scenario",
        values="coefficient_original_units",
    ).reset_index()
    pivot["absolute_change"] = pivot["contaminated"] - pivot["clean"]
    pivot["absolute_change_magnitude"] = pivot["absolute_change"].abs()
    pivot["percent_change_from_clean"] = np.where(
        pivot["clean"].abs().gt(1e-8),
        100 * pivot["absolute_change"] / pivot["clean"].abs(),
        np.nan,
    )

    performance_pivot = performance.pivot(
        index="estimator", columns="training_scenario"
    )
    degradation_rows: list[dict[str, float | str]] = []
    for estimator in performance["estimator"].unique():
        row: dict[str, float | str] = {"estimator": estimator}
        for metric in ["mae", "rmse", "median_absolute_error", "r_squared"]:
            clean = float(performance_pivot.loc[estimator, (metric, "clean")])
            contaminated = float(
                performance_pivot.loc[estimator, (metric, "contaminated")]
            )
            row[f"clean_{metric}"] = clean
            row[f"contaminated_{metric}"] = contaminated
            row[f"change_{metric}"] = contaminated - clean
        degradation_rows.append(row)
    return performance, pivot, pd.DataFrame(degradation_rows)


def detection_tables(
    training: pd.DataFrame,
    contaminated_y: np.ndarray,
    known_mask: np.ndarray,
    contaminated_lts: ReweightedLTSResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure LTS audit screening against known artificial contamination."""
    robust_residual = contaminated_lts.raw_standardized_residuals
    zero_weight = ~contaminated_lts.reweighting_weights
    above_2_5 = np.abs(robust_residual) > AUDIT_STANDARDIZED_RESIDUAL_CUTOFF
    flagged = zero_weight | above_2_5
    true_positive = int(np.sum(flagged & known_mask))
    false_positive = int(np.sum(flagged & ~known_mask))
    false_negative = int(np.sum(~flagged & known_mask))
    true_negative = int(np.sum(~flagged & ~known_mask))
    summary = pd.DataFrame(
        [
            {
                "training_observations": len(training),
                "known_contaminated": int(known_mask.sum()),
                "flagged_total": int(flagged.sum()),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "detection_recall": true_positive / int(known_mask.sum()),
                "detection_precision": true_positive / (true_positive + false_positive),
                "false_positive_rate": false_positive / int((~known_mask).sum()),
                "raw_mad_scale": contaminated_lts.raw_robust_scale,
                "reweighting_cutoff": contaminated_lts.reweighting_cutoff,
                "audit_residual_cutoff": AUDIT_STANDARDIZED_RESIDUAL_CUTOFF,
            }
        ]
    )
    detail = training[
        [
            "modeling_row_id",
            "source_row_id",
            "pickup_datetime",
            "dropoff_datetime",
            TARGET,
            *PREDICTORS,
        ]
    ].copy()
    detail = detail.rename(columns={TARGET: "original_fare_amount"})
    detail["experimental_fare_amount"] = contaminated_y
    detail["known_artificial_contamination"] = known_mask
    detail["raw_lts_standardized_robust_residual"] = robust_residual
    detail["zero_reweight_flag"] = zero_weight
    detail["absolute_standardized_residual_above_2_5"] = above_2_5
    detail["audit_candidate"] = flagged
    detail["detection_class"] = np.select(
        [flagged & known_mask, flagged & ~known_mask, ~flagged & known_mask],
        ["true_positive", "false_positive", "false_negative"],
        default="true_negative",
    )
    detail = detail.loc[known_mask | flagged].copy()
    detail["absolute_standardized_residual"] = detail[
        "raw_lts_standardized_robust_residual"
    ].abs()
    detail = detail.sort_values(
        ["known_artificial_contamination", "absolute_standardized_residual"],
        ascending=[False, False],
    )
    return summary, detail


def int64_checksum(values: np.ndarray) -> str:
    """Hash an ordered int64 identity vector."""
    return hashlib.sha256(np.asarray(values, dtype="int64").tobytes()).hexdigest()


def save_figures(
    performance: pd.DataFrame,
    coefficient_change: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Save stress-test performance and coefficient-stability figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, metric in zip(axes, ["mae", "rmse"], strict=True):
        pivot = performance.pivot(
            index="estimator", columns="training_scenario", values=metric
        )
        pivot.plot.bar(ax=axis, color=["#315C7C", "#9B2226"])
        axis.set(title=f"Clean-test {metric.upper()}", ylabel="USD", xlabel="")
        axis.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(figures_dir / "contamination_performance.png", dpi=220)
    plt.close(fig)

    slopes = coefficient_change.loc[coefficient_change["term"].ne("const")]
    pivot = slopes.pivot(
        index="term", columns="estimator", values="absolute_change_magnitude"
    )
    axis = pivot.plot.bar(figsize=(9, 4.5), color=["#9B2226", "#315C7C"])
    axis.set(title="Absolute coefficient change after +$100 contamination")
    axis.set(ylabel="Absolute change in original units", xlabel="")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "contamination_coefficient_change.png", dpi=220)
    plt.close()


def run_stress_test(
    input_path: Path,
    tables_dir: Path,
    figures_dir: Path,
    metadata_path: Path,
) -> dict[str, int | float | str | bool]:
    """Execute and export the protected controlled-contamination experiment."""
    training, test = load_protected_split(input_path)
    test_ids_before = int64_checksum(test["source_row_id"].to_numpy())
    test_y_before = hashlib.sha256(
        test[TARGET].to_numpy(dtype="float64").tobytes()
    ).hexdigest()
    X_train_raw = training[PREDICTORS].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    clean_y = training[TARGET].to_numpy(dtype=float)
    contaminated_y, known_mask = contaminate_training_response(training)

    with threadpool_limits(limits=2):
        clean_ols, clean_lts = fit_pair(X_train, clean_y)
        contaminated_ols, contaminated_lts = fit_pair(X_train, contaminated_y)

    performance, coefficient_change, degradation = comparison_tables(
        scaler,
        test,
        clean_ols,
        clean_lts,
        contaminated_ols,
        contaminated_lts,
    )
    detection, detection_detail = detection_tables(
        training, contaminated_y, known_mask, contaminated_lts
    )
    test_unchanged = (
        test_ids_before == int64_checksum(test["source_row_id"].to_numpy())
        and test_y_before
        == hashlib.sha256(test[TARGET].to_numpy(dtype="float64").tobytes()).hexdigest()
    )
    if not test_unchanged:
        raise AssertionError("The clean test set changed during contamination.")

    tables_dir.mkdir(parents=True, exist_ok=True)
    write_csv(performance, tables_dir / "contamination_model_performance.csv")
    write_csv(coefficient_change, tables_dir / "contamination_coefficient_change.csv")
    write_csv(degradation, tables_dir / "contamination_prediction_degradation.csv")
    write_csv(detection, tables_dir / "contamination_detection_summary.csv")
    write_csv(detection_detail, tables_dir / "contamination_detection_detail.csv")
    save_figures(performance, coefficient_change, figures_dir)

    summary = detection.iloc[0]
    metadata: dict[str, int | float | str | bool] = {
        "scope": "controlled training-response contamination",
        "training_observations": len(training),
        "test_observations": len(test),
        "contaminated_training_observations": int(known_mask.sum()),
        "contamination_fraction": CONTAMINATION_FRACTION,
        "fare_addition_usd": CONTAMINATION_ADDITION,
        "seed": RANDOM_SEED,
        "alpha": PRIMARY_ALPHA,
        "initial_subsets": N_INITIAL_SUBSETS,
        "test_set_unchanged": test_unchanged,
        "input_csv_sha256": sha256_file(input_path),
        "contaminated_source_id_sha256": int64_checksum(
            training.loc[known_mask, "source_row_id"].to_numpy()
        ),
        "detection_recall": float(summary["detection_recall"]),
        "detection_precision": float(summary["detection_precision"]),
        "false_positive_rate": float(summary["false_positive_rate"]),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse reproducible path overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/processed/taxi_modeling_50000.csv",
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=PROJECT_ROOT / "outputs/tables"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=PROJECT_ROOT / "outputs/figures"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "outputs/contamination_metadata.json",
    )
    return parser.parse_args()


def main() -> None:
    """Run the controlled contamination stress test."""
    args = parse_args()
    print(
        json.dumps(
            run_stress_test(
                args.input, args.tables_dir, args.figures_dir, args.metadata
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
