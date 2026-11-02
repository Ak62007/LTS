"""Explore training data and assess the proposed linear model specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.taxi_lts import RANDOM_SEED
from src.taxi_lts.prepare_data import PROJECT_ROOT, sha256_file, write_csv

TARGET = "fare_amount"
PREDICTORS = [
    "trip_distance",
    "duration_min",
    "passenger_count",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "is_weekend",
]
DESCRIPTIVE_VARIABLES = [
    TARGET,
    "trip_distance",
    "duration_min",
    "passenger_count",
    "pickup_hour",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "is_weekend",
]
DISPLAY_LABELS = {
    "fare_amount": "Fare amount USD",
    "trip_distance": "Trip distance miles",
    "duration_min": "Duration minutes",
    "passenger_count": "Passenger count",
    "pickup_hour": "Pickup hour",
    "pickup_hour_sin": "Pickup hour sine",
    "pickup_hour_cos": "Pickup hour cosine",
    "is_weekend": "Weekend indicator",
}


def load_training_data(path: Path) -> pd.DataFrame:
    """Load the modelling file and immediately isolate training observations."""
    data = pd.read_csv(
        path,
        parse_dates=["pickup_datetime", "dropoff_datetime"],
    )
    training = data.loc[data["split"].eq("train")].copy()
    if len(training) != 40_000:
        raise ValueError(f"Expected 40,000 training rows, found {len(training):,}.")
    if training["source_row_id"].nunique() != len(training):
        raise ValueError("Training source IDs must be unique.")
    return training


def descriptive_statistics(training: pd.DataFrame) -> pd.DataFrame:
    """Return detailed distribution summaries including tail behavior."""
    rows: list[dict[str, int | float | str]] = []
    for variable in DESCRIPTIVE_VARIABLES:
        values = training[variable]
        quantiles = values.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        rows.append(
            {
                "variable": variable,
                "label": DISPLAY_LABELS[variable],
                "count": int(values.count()),
                "missing": int(values.isna().sum()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "minimum": float(values.min()),
                "p01": float(quantiles.loc[0.01]),
                "p05": float(quantiles.loc[0.05]),
                "p25": float(quantiles.loc[0.25]),
                "median": float(quantiles.loc[0.5]),
                "p75": float(quantiles.loc[0.75]),
                "p95": float(quantiles.loc[0.95]),
                "p99": float(quantiles.loc[0.99]),
                "maximum": float(values.max()),
                "skewness": float(values.skew()),
                "excess_kurtosis": float(values.kurt()),
            }
        )
    return pd.DataFrame(rows)


def correlation_tables(training: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate linear and rank correlations for target and predictors."""
    columns = [TARGET, *PREDICTORS]
    return (
        training[columns].corr(method="pearson"),
        training[columns].corr(method="spearman"),
    )


def vif_table(training: pd.DataFrame) -> pd.DataFrame:
    """Calculate variance inflation factors for the proposed predictors."""
    design = sm.add_constant(training[PREDICTORS], has_constant="add")
    return pd.DataFrame(
        {
            "term": design.columns,
            "vif": [
                float(variance_inflation_factor(design.to_numpy(), index))
                for index in range(design.shape[1])
            ],
        }
    )


def fit_diagnostic_ols(training: pd.DataFrame) -> tuple[object, object]:
    """Fit training-only OLS and HC3 covariance for specification diagnostics."""
    design = sm.add_constant(training[PREDICTORS], has_constant="add")
    model = sm.OLS(training[TARGET], design).fit()
    return model, model.get_robustcov_results(cov_type="HC3")


def coefficient_table(model: object, hc3_model: object) -> pd.DataFrame:
    """Export conventional and HC3 uncertainty alongside raw-unit coefficients."""
    confidence = np.asarray(hc3_model.conf_int(alpha=0.05))
    return pd.DataFrame(
        {
            "term": model.model.exog_names,
            "coefficient": np.asarray(model.params),
            "standard_error_conventional": np.asarray(model.bse),
            "standard_error_hc3": np.asarray(hc3_model.bse),
            "hc3_ci_lower_95": confidence[:, 0],
            "hc3_ci_upper_95": confidence[:, 1],
            "hc3_p_value": np.asarray(hc3_model.pvalues),
        }
    )


def diagnostic_tables(
    training: pd.DataFrame, model: object
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize fit, assumptions, influence, and binned residual patterns."""
    residuals = np.asarray(model.resid)
    fitted = np.asarray(model.fittedvalues)
    influence = model.get_influence()
    leverage = np.asarray(influence.hat_matrix_diag)
    cooks_distance = np.asarray(influence.cooks_distance[0])
    studentized = np.asarray(influence.resid_studentized_internal)
    n = len(training)
    p = len(model.params)

    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(
        residuals, np.asarray(model.model.exog)
    )
    reset = linear_reset(model, power=2, use_f=True)
    jb_stat, jb_p, residual_skew, residual_kurtosis = sm.stats.jarque_bera(residuals)
    shapiro_sample = np.random.default_rng(RANDOM_SEED).choice(
        residuals, size=min(5_000, n), replace=False
    )
    shapiro_stat, shapiro_p = stats.shapiro(shapiro_sample)

    fit_rows = [
        ("training_observations", n),
        ("estimated_parameters", p),
        ("training_r_squared", float(model.rsquared)),
        ("training_adjusted_r_squared", float(model.rsquared_adj)),
        ("training_mae", mean_absolute_error(training[TARGET], fitted)),
        ("training_rmse", float(np.sqrt(mean_squared_error(training[TARGET], fitted)))),
        ("training_median_absolute_error", float(np.median(np.abs(residuals)))),
        ("residual_mean", float(np.mean(residuals))),
        ("residual_std", float(np.std(residuals, ddof=p))),
        ("residual_skewness", float(residual_skew)),
        ("residual_kurtosis", float(residual_kurtosis)),
        ("breusch_pagan_lm", float(bp_lm)),
        ("breusch_pagan_lm_p_value", float(bp_lm_p)),
        ("breusch_pagan_f", float(bp_f)),
        ("breusch_pagan_f_p_value", float(bp_f_p)),
        ("ramsey_reset_f", float(reset.fvalue)),
        ("ramsey_reset_p_value", float(reset.pvalue)),
        ("jarque_bera", float(jb_stat)),
        ("jarque_bera_p_value", float(jb_p)),
        ("shapiro_wilk_sample_5000", float(shapiro_stat)),
        ("shapiro_wilk_p_value", float(shapiro_p)),
        ("durbin_watson", float(sm.stats.durbin_watson(residuals))),
    ]
    diagnostics = pd.DataFrame(fit_rows, columns=["diagnostic", "value"])

    leverage_threshold = 2 * p / n
    cook_threshold = 4 / n
    influence_summary = pd.DataFrame(
        [
            {
                "training_observations": n,
                "parameter_count": p,
                "leverage_threshold_2p_over_n": leverage_threshold,
                "high_leverage_count": int(np.sum(leverage > leverage_threshold)),
                "high_leverage_fraction": float(np.mean(leverage > leverage_threshold)),
                "cook_threshold_4_over_n": cook_threshold,
                "cook_above_threshold_count": int(
                    np.sum(cooks_distance > cook_threshold)
                ),
                "cook_above_threshold_fraction": float(
                    np.mean(cooks_distance > cook_threshold)
                ),
                "absolute_studentized_residual_above_3_count": int(
                    np.sum(np.abs(studentized) > 3)
                ),
                "absolute_studentized_residual_above_3_fraction": float(
                    np.mean(np.abs(studentized) > 3)
                ),
                "maximum_leverage": float(leverage.max()),
                "maximum_cooks_distance": float(cooks_distance.max()),
                "maximum_absolute_studentized_residual": float(
                    np.abs(studentized).max()
                ),
            }
        ]
    )

    residual_frame = training[["trip_distance", "duration_min"]].copy()
    residual_frame["residual"] = residuals
    binned_rows: list[dict[str, int | float | str]] = []
    for variable in ["trip_distance", "duration_min"]:
        bins = pd.qcut(residual_frame[variable], q=20, duplicates="drop")
        grouped = residual_frame.groupby(bins, observed=True)
        for bin_number, (_, group) in enumerate(grouped, start=1):
            standard_error = group["residual"].std() / np.sqrt(len(group))
            binned_rows.append(
                {
                    "variable": variable,
                    "bin_number": bin_number,
                    "count": len(group),
                    "predictor_mean": float(group[variable].mean()),
                    "residual_mean": float(group["residual"].mean()),
                    "residual_mean_ci_lower_95": float(
                        group["residual"].mean() - 1.96 * standard_error
                    ),
                    "residual_mean_ci_upper_95": float(
                        group["residual"].mean() + 1.96 * standard_error
                    ),
                }
            )

    scale_frame = pd.DataFrame({"fitted": fitted, "residual": residuals})
    scale_frame["fitted_decile"] = pd.qcut(
        scale_frame["fitted"], q=10, duplicates="drop"
    )
    scale_rows: list[dict[str, int | float]] = []
    for decile, (_, group) in enumerate(
        scale_frame.groupby("fitted_decile", observed=True), start=1
    ):
        scale_rows.append(
            {
                "fitted_decile": decile,
                "count": len(group),
                "fitted_mean": float(group["fitted"].mean()),
                "mean_absolute_residual": float(group["residual"].abs().mean()),
                "median_absolute_residual": float(group["residual"].abs().median()),
                "residual_rmse": float(np.sqrt(np.mean(group["residual"] ** 2))),
            }
        )

    influential = training[
        [
            "modeling_row_id",
            "source_row_id",
            TARGET,
            "trip_distance",
            "duration_min",
            "passenger_count",
            "pickup_hour",
            "is_weekend",
        ]
    ].copy()
    influential["fitted_fare"] = fitted
    influential["residual"] = residuals
    influential["studentized_residual_internal"] = studentized
    influential["leverage"] = leverage
    influential["cooks_distance"] = cooks_distance
    influential = influential.sort_values("cooks_distance", ascending=False).head(20)
    return (
        diagnostics,
        influence_summary,
        pd.DataFrame(binned_rows),
        pd.DataFrame(scale_rows),
        influential,
    )


def set_plot_style() -> None:
    """Apply a restrained report-ready plotting style."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )


def save_distribution_figure(training: pd.DataFrame, path: Path) -> None:
    """Plot central distributions while documenting the full observed maxima."""
    variables = [TARGET, "trip_distance", "duration_min", "passenger_count"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    for axis, variable in zip(axes.flat, variables, strict=True):
        upper = float(training[variable].quantile(0.995))
        visible = training.loc[training[variable].le(upper), variable]
        sns.histplot(visible, bins=45, ax=axis, color="#315C7C", edgecolor=None)
        axis.set_title(DISPLAY_LABELS[variable])
        axis.set_xlabel(
            f"Displayed through p99.5 = {upper:.2f}; full max = "
            f"{training[variable].max():.2f}"
        )
        axis.set_ylabel("Training trips")
    fig.suptitle("Training distributions with tails retained in analysis", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_relationship_figure(training: pd.DataFrame, path: Path) -> None:
    """Plot dense fare relationships in a central visual window."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fare_limit = float(training[TARGET].quantile(0.995))
    for axis, variable in zip(axes, ["trip_distance", "duration_min"], strict=True):
        x_limit = float(training[variable].quantile(0.995))
        visible = training.loc[
            training[variable].le(x_limit) & training[TARGET].le(fare_limit)
        ]
        plot = axis.hexbin(
            visible[variable],
            visible[TARGET],
            gridsize=48,
            mincnt=1,
            bins="log",
            cmap="Blues",
        )
        axis.set_title(f"Fare and {DISPLAY_LABELS[variable].lower()}")
        axis.set_xlabel(DISPLAY_LABELS[variable])
        axis.set_ylabel("Fare amount USD")
        fig.colorbar(plot, ax=axis, label="Log trip density")
    fig.suptitle(
        "Training fare relationships within the p99.5 visual window", fontsize=14
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_correlation_figure(pearson: pd.DataFrame, path: Path) -> None:
    """Plot Pearson correlations with readable variable labels."""
    labels = [DISPLAY_LABELS[column] for column in pearson.columns]
    fig, axis = plt.subplots(figsize=(8.2, 6.7))
    sns.heatmap(
        pearson,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_title("Pearson correlations in the training data", pad=12)
    axis.tick_params(axis="x", rotation=40)
    axis.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_residual_figure(training: pd.DataFrame, model: object, path: Path) -> None:
    """Plot training-only residual shape, scale, and normality evidence."""
    fitted = np.asarray(model.fittedvalues)
    residuals = np.asarray(model.resid)
    rng = np.random.default_rng(RANDOM_SEED)
    qq_residuals = rng.choice(residuals, size=min(5_000, len(residuals)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    plot = axes[0].hexbin(
        fitted,
        residuals,
        gridsize=55,
        mincnt=1,
        bins="log",
        cmap="Blues",
    )
    axes[0].axhline(0, color="#8B1E1E", linewidth=1)
    axes[0].set_xlabel("OLS fitted fare USD")
    axes[0].set_ylabel("Residual USD")
    axes[0].set_title("Residuals against fitted values")
    fig.colorbar(plot, ax=axes[0], label="Log trip density")

    stats.probplot(qq_residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q Q plot from 5,000 residuals")
    axes[1].get_lines()[0].set_markerfacecolor("#315C7C")
    axes[1].get_lines()[0].set_markeredgecolor("#315C7C")
    axes[1].get_lines()[0].set_markersize(2.5)
    axes[1].get_lines()[1].set_color("#8B1E1E")
    fig.suptitle("Training-only OLS residual diagnostics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_binned_residual_figure(binned: pd.DataFrame, path: Path) -> None:
    """Plot mean residuals by predictor quantile to assess functional form."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, variable in zip(axes, ["trip_distance", "duration_min"], strict=True):
        subset = binned.loc[binned["variable"].eq(variable)]
        axis.plot(
            subset["predictor_mean"],
            subset["residual_mean"],
            marker="o",
            color="#315C7C",
        )
        axis.fill_between(
            subset["predictor_mean"],
            subset["residual_mean_ci_lower_95"],
            subset["residual_mean_ci_upper_95"],
            color="#315C7C",
            alpha=0.18,
        )
        axis.axhline(0, color="#8B1E1E", linewidth=1)
        axis.set_title(DISPLAY_LABELS[variable])
        axis.set_xlabel("Mean within quantile bin")
        axis.set_ylabel("Mean OLS residual USD")
    fig.suptitle("Binned residual evidence for linear model form", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_exploration(
    input_path: Path, tables_dir: Path, figures_dir: Path, summary_path: Path
) -> dict[str, int | float | str]:
    """Run all training-only diagnostics and write traceable artifacts."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    training = load_training_data(input_path)

    descriptive = descriptive_statistics(training)
    pearson, spearman = correlation_tables(training)
    vif = vif_table(training)
    model, hc3_model = fit_diagnostic_ols(training)
    coefficients = coefficient_table(model, hc3_model)
    diagnostics, influence, binned, residual_scale, influential = diagnostic_tables(
        training, model
    )

    write_csv(descriptive, tables_dir / "training_descriptive_statistics.csv")
    write_csv(
        pearson.reset_index(names="variable"),
        tables_dir / "training_correlation_pearson.csv",
    )
    write_csv(
        spearman.reset_index(names="variable"),
        tables_dir / "training_correlation_spearman.csv",
    )
    write_csv(vif, tables_dir / "training_vif.csv")
    write_csv(coefficients, tables_dir / "training_ols_coefficients.csv")
    write_csv(diagnostics, tables_dir / "training_ols_diagnostics.csv")
    write_csv(influence, tables_dir / "training_influence_summary.csv")
    write_csv(binned, tables_dir / "training_binned_residuals.csv")
    write_csv(
        residual_scale, tables_dir / "training_residual_scale_by_fitted_decile.csv"
    )
    write_csv(influential, tables_dir / "training_top_influential_records.csv")

    set_plot_style()
    save_distribution_figure(training, figures_dir / "training_distributions.png")
    save_relationship_figure(training, figures_dir / "training_fare_relationships.png")
    save_correlation_figure(pearson, figures_dir / "training_correlation_heatmap.png")
    save_residual_figure(
        training, model, figures_dir / "training_residual_diagnostics.png"
    )
    save_binned_residual_figure(binned, figures_dir / "training_binned_residuals.png")

    diagnostic_values = diagnostics.set_index("diagnostic")["value"]
    influence_row = influence.iloc[0]
    summary: dict[str, int | float | str] = {
        "scope": "training records only",
        "training_observations": len(training),
        "predictor_count": len(PREDICTORS),
        "training_source_id_sha256": hashlib_sha256_int64(
            training["source_row_id"].to_numpy(dtype="int64")
        ),
        "input_csv_sha256": sha256_file(input_path),
        "training_r_squared": float(diagnostic_values["training_r_squared"]),
        "training_mae": float(diagnostic_values["training_mae"]),
        "training_rmse": float(diagnostic_values["training_rmse"]),
        "breusch_pagan_p_value": float(diagnostic_values["breusch_pagan_lm_p_value"]),
        "ramsey_reset_p_value": float(diagnostic_values["ramsey_reset_p_value"]),
        "jarque_bera_p_value": float(diagnostic_values["jarque_bera_p_value"]),
        "high_leverage_count": int(influence_row["high_leverage_count"]),
        "cook_above_threshold_count": int(influence_row["cook_above_threshold_count"]),
        "absolute_studentized_residual_above_3_count": int(
            influence_row["absolute_studentized_residual_above_3_count"]
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def hashlib_sha256_int64(values: np.ndarray) -> str:
    """Hash an ordered int64 array without text-format ambiguity."""
    import hashlib

    return hashlib.sha256(np.asarray(values, dtype="int64").tobytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse reproducible input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "taxi_modeling_50000.csv",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tables",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "training_eda_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line training exploration workflow."""
    args = parse_args()
    summary = run_exploration(
        args.input, args.tables_dir, args.figures_dir, args.summary
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
