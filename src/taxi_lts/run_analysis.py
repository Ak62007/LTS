"""Fit clean OLS and prespecified FAST-LTS models on the protected split."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.taxi_lts import RANDOM_SEED
from src.taxi_lts.explore_training import PREDICTORS, TARGET
from src.taxi_lts.prepare_data import PROJECT_ROOT, sha256_file, write_csv
from src.taxi_lts.robust_lts import ReweightedLTSResult, fit_reweighted_fast_lts

PRIMARY_ALPHA = 0.75
SENSITIVITY_ALPHAS = [0.50, 0.75, 0.90]
N_INITIAL_SUBSETS = 500
AUDIT_STANDARDIZED_RESIDUAL_CUTOFF = 2.5


@dataclass
class CleanFits:
    """Store training-fitted preprocessing and estimators before evaluation."""

    scaler: StandardScaler
    ols: LinearRegression
    lts_by_alpha: dict[float, ReweightedLTSResult]


def load_protected_split(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the fixed sample and return disjoint training and test frames."""
    data = pd.read_csv(path, parse_dates=["pickup_datetime", "dropoff_datetime"])
    training = data.loc[data["split"].eq("train")].copy()
    test = data.loc[data["split"].eq("test")].copy()
    if len(training) != 40_000 or len(test) != 10_000:
        raise ValueError("Expected a 40,000/10,000 protected split.")
    if set(training["source_row_id"]) & set(test["source_row_id"]):
        raise ValueError("Training and test source IDs must be disjoint.")
    return training, test


def fit_clean_models(training: pd.DataFrame) -> CleanFits:
    """Fit preprocessing and all prespecified models using training rows only."""
    X_train_raw = training[PREDICTORS].to_numpy(dtype=float)
    y_train = training[TARGET].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    ols = LinearRegression().fit(X_train, y_train)
    lts_by_alpha: dict[float, ReweightedLTSResult] = {}
    with threadpool_limits(limits=2):
        for alpha in SENSITIVITY_ALPHAS:
            lts_by_alpha[alpha] = fit_reweighted_fast_lts(
                X_train,
                y_train,
                alpha=alpha,
                n_initial_subsets=N_INITIAL_SUBSETS,
                random_state=RANDOM_SEED,
            )
    return CleanFits(scaler=scaler, ols=ols, lts_by_alpha=lts_by_alpha)


def prediction_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Calculate prespecified prediction measures."""
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "median_absolute_error": float(median_absolute_error(y_true, prediction)),
        "r_squared": float(r2_score(y_true, prediction)),
    }


def performance_table(
    fits: CleanFits, training: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """Evaluate frozen clean models on train and then the untouched test set."""
    rows: list[dict[str, float | str]] = []
    for split_name, frame in [("train", training), ("test", test)]:
        X = fits.scaler.transform(frame[PREDICTORS].to_numpy(dtype=float))
        y = frame[TARGET].to_numpy(dtype=float)
        estimators: list[tuple[str, float | None, np.ndarray]] = [
            ("OLS", None, fits.ols.predict(X))
        ]
        for alpha, result in fits.lts_by_alpha.items():
            estimators.extend(
                [
                    ("Raw LTS", alpha, result.raw_estimator.predict(X).reshape(-1)),
                    ("Reweighted LTS", alpha, result.reweighted_model.predict(X)),
                ]
            )
        for estimator, alpha, prediction in estimators:
            rows.append(
                {
                    "evaluation_split": split_name,
                    "estimator": estimator,
                    "alpha": alpha,
                    **prediction_metrics(y, prediction),
                }
            )
    return pd.DataFrame(rows)


def original_unit_parameters(
    scaler: StandardScaler, coefficients: np.ndarray, intercept: float
) -> tuple[np.ndarray, float]:
    """Back-transform parameters from standardized to original predictors."""
    original_coefficients = np.asarray(coefficients) / scaler.scale_
    original_intercept = float(
        intercept - np.sum(np.asarray(coefficients) * scaler.mean_ / scaler.scale_)
    )
    return original_coefficients, original_intercept


def coefficient_table(fits: CleanFits, training: pd.DataFrame) -> pd.DataFrame:
    """Compare original-unit OLS and LTS parameters with HC3 OLS uncertainty."""
    raw_design = sm.add_constant(training[PREDICTORS], has_constant="add")
    ols_hc3 = (
        sm.OLS(training[TARGET], raw_design).fit().get_robustcov_results(cov_type="HC3")
    )
    hc3_by_term = {
        term: (float(coef), float(se))
        for term, coef, se in zip(
            raw_design.columns, ols_hc3.params, ols_hc3.bse, strict=True
        )
    }
    rows: list[dict[str, float | str]] = []

    def append_model(
        estimator: str, alpha: float | None, coefficients: np.ndarray, intercept: float
    ) -> None:
        original_coef, original_intercept = original_unit_parameters(
            fits.scaler, coefficients, intercept
        )
        for term, value in zip(
            ["const", *PREDICTORS],
            [original_intercept, *original_coef],
            strict=True,
        ):
            hc3 = hc3_by_term.get(term)
            rows.append(
                {
                    "estimator": estimator,
                    "alpha": alpha,
                    "term": term,
                    "coefficient_original_units": float(value),
                    "ols_hc3_standard_error": hc3[1]
                    if estimator == "OLS" and hc3
                    else np.nan,
                }
            )

    append_model("OLS", None, fits.ols.coef_, float(fits.ols.intercept_))
    for alpha, result in fits.lts_by_alpha.items():
        raw = result.raw_estimator.model
        append_model(
            "Raw LTS",
            alpha,
            np.asarray(raw.coef_).reshape(-1),
            float(np.asarray(raw.intercept_).reshape(-1)[0]),
        )
        append_model(
            "Reweighted LTS",
            alpha,
            np.asarray(result.reweighted_model.coef_),
            float(result.reweighted_model.intercept_),
        )
    return pd.DataFrame(rows)


def scaler_table(scaler: StandardScaler) -> pd.DataFrame:
    """Export training-fitted scaling parameters for leakage auditing."""
    return pd.DataFrame(
        {
            "predictor": PREDICTORS,
            "training_mean": scaler.mean_,
            "training_scale": scaler.scale_,
        }
    )


def audit_candidate_table(
    fits: CleanFits, training: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create transparent primary-LTS flags without alleging fraud."""
    result = fits.lts_by_alpha[PRIMARY_ALPHA]
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
    detail["raw_lts_residual"] = result.raw_residuals
    detail["raw_lts_standardized_robust_residual"] = result.raw_standardized_residuals
    detail["zero_reweight_flag"] = ~result.reweighting_weights
    detail["absolute_standardized_residual_above_2_5"] = (
        np.abs(result.raw_standardized_residuals) > AUDIT_STANDARDIZED_RESIDUAL_CUTOFF
    )
    detail["audit_candidate"] = (
        detail["zero_reweight_flag"]
        | detail["absolute_standardized_residual_above_2_5"]
    )
    candidates = detail.loc[detail["audit_candidate"]].copy()
    candidates["absolute_standardized_residual"] = candidates[
        "raw_lts_standardized_robust_residual"
    ].abs()
    candidates = candidates.sort_values(
        "absolute_standardized_residual", ascending=False
    )
    summary = pd.DataFrame(
        [
            {
                "training_observations": len(training),
                "alpha": PRIMARY_ALPHA,
                "raw_h_subset_size": len(result.raw_estimator.best_h_subset),
                "raw_mad_scale": result.raw_robust_scale,
                "reweighting_cutoff": result.reweighting_cutoff,
                "zero_reweight_count": int((~result.reweighting_weights).sum()),
                "absolute_standardized_residual_above_2_5_count": int(
                    detail["absolute_standardized_residual_above_2_5"].sum()
                ),
                "audit_candidate_count": len(candidates),
                "audit_candidate_fraction": len(candidates) / len(training),
            }
        ]
    )
    return candidates, summary


def save_clean_figures(
    performance: pd.DataFrame, coefficients: pd.DataFrame, figures_dir: Path
) -> None:
    """Save compact clean-model comparison figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    test = performance.loc[performance["evaluation_split"].eq("test")].copy()
    test["label"] = test.apply(
        lambda row: (
            row["estimator"]
            if pd.isna(row["alpha"])
            else f"{row['estimator']} a={row['alpha']:.2f}"
        ),
        axis=1,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].barh(test["label"], test["mae"], color="#315C7C")
    axes[0].set(title="Clean test MAE", xlabel="USD")
    axes[1].barh(test["label"], test["rmse"], color="#4F772D")
    axes[1].set(title="Clean test RMSE", xlabel="USD")
    fig.tight_layout()
    fig.savefig(figures_dir / "clean_model_performance.png", dpi=220)
    plt.close(fig)

    primary = coefficients.loc[
        coefficients["term"].isin(PREDICTORS)
        & (
            coefficients["estimator"].eq("OLS")
            | coefficients["alpha"].eq(PRIMARY_ALPHA)
        )
    ].copy()
    pivot = primary.pivot(
        index="term", columns="estimator", values="coefficient_original_units"
    )
    axis = pivot.plot.bar(figsize=(10, 4.8))
    axis.set(title="Original-unit coefficients", ylabel="Fare change in USD")
    axis.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "clean_model_coefficients.png", dpi=220)
    plt.close()


def run_clean_analysis(
    input_path: Path, tables_dir: Path, figures_dir: Path, metadata_path: Path
) -> dict[str, int | float | str]:
    """Fit, evaluate, export, and fingerprint the clean model comparison."""
    training, test = load_protected_split(input_path)
    fits = fit_clean_models(training)
    performance = performance_table(fits, training, test)
    coefficients = coefficient_table(fits, training)
    candidates, candidate_summary = audit_candidate_table(fits, training)
    scaling = scaler_table(fits.scaler)

    tables_dir.mkdir(parents=True, exist_ok=True)
    write_csv(performance, tables_dir / "clean_model_performance.csv")
    write_csv(coefficients, tables_dir / "clean_model_coefficients.csv")
    write_csv(scaling, tables_dir / "training_scaler_parameters.csv")
    write_csv(candidates, tables_dir / "lts_audit_candidates.csv")
    write_csv(candidate_summary, tables_dir / "lts_audit_candidate_summary.csv")
    save_clean_figures(performance, coefficients, figures_dir)

    primary_test = performance.loc[
        performance["evaluation_split"].eq("test")
        & performance["estimator"].eq("Reweighted LTS")
        & performance["alpha"].eq(PRIMARY_ALPHA)
    ].iloc[0]
    ols_test = performance.loc[
        performance["evaluation_split"].eq("test") & performance["estimator"].eq("OLS")
    ].iloc[0]
    metadata: dict[str, int | float | str] = {
        "scope": "clean models; contamination not yet applied",
        "training_observations": len(training),
        "test_observations": len(test),
        "seed": RANDOM_SEED,
        "initial_subsets": N_INITIAL_SUBSETS,
        "primary_alpha": PRIMARY_ALPHA,
        "input_csv_sha256": sha256_file(input_path),
        "ols_test_mae": float(ols_test["mae"]),
        "ols_test_rmse": float(ols_test["rmse"]),
        "primary_reweighted_lts_test_mae": float(primary_test["mae"]),
        "primary_reweighted_lts_test_rmse": float(primary_test["rmse"]),
        "primary_audit_candidate_count": int(
            candidate_summary.iloc[0]["audit_candidate_count"]
        ),
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
        default=PROJECT_ROOT / "outputs/clean_model_metadata.json",
    )
    return parser.parse_args()


def main() -> None:
    """Run the clean protected model comparison."""
    args = parse_args()
    print(
        json.dumps(
            run_clean_analysis(
                args.input, args.tables_dir, args.figures_dir, args.metadata
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
