"""Build and execute the analysis notebooks for all completed project stages."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def code(source: str):
    """Return a code notebook cell."""
    return nbformat.v4.new_code_cell(source.strip())


def notebook(cells: list) -> nbformat.NotebookNode:
    """Return a Python notebook with stable kernel metadata."""
    document = nbformat.v4.new_notebook(cells=cells)
    document.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    document.metadata.language_info = {"name": "python", "version": "3.13"}
    return document


def build_data_notebook() -> nbformat.NotebookNode:
    """Create the data preparation, filtering, sampling, and split notebook."""
    return notebook(
        [
            code(
                """
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd()
assert (PROJECT_ROOT / "pyproject.toml").exists(), "Run from the repository root."
pd.set_option("display.max_columns", 30)
"""
            ),
            code(
                """
source_metadata = json.loads((PROJECT_ROOT / "data/source_metadata.json").read_text())
display(pd.Series(source_metadata, name="Recorded value").to_frame())

raw_path = PROJECT_ROOT / "data/raw/yellow_tripdata_2026-01.parquet"
digest = hashlib.sha256()
with raw_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)

actual_sha256 = digest.hexdigest()
expected_sha256 = source_metadata["raw_sha256"]
print("Checksum matches preserved metadata:", actual_sha256 == expected_sha256)
print("SHA-256:", actual_sha256)
"""
            ),
            code(
                """
flow = pd.read_csv(PROJECT_ROOT / "outputs/tables/data_quality_flow.csv")
flow["retained_percent_of_original"] = 100 * flow["records_after"] / flow.loc[0, "records_after"]
display(flow.style.format({"retained_percent_of_original": "{:.2f}%"}))

plot_flow = flow.loc[~flow["stage"].isin(["original_records", "eligible_records", "deterministic_sample"])]
ax = plot_flow.plot.barh(x="stage", y="records_removed", legend=False, figsize=(8, 4), color="#315C7C")
ax.set_xlabel("Records removed at this sequential step")
ax.set_ylabel("")
ax.set_title("Sequential filtering audit")
plt.tight_layout()
plt.show()
"""
            ),
            code(
                """
ranges = pd.read_csv(PROJECT_ROOT / "outputs/tables/data_quality_ranges.csv")
missingness = pd.read_csv(PROJECT_ROOT / "outputs/tables/data_quality_missingness.csv")
print("Observed ranges after filtering and sampling")
display(ranges)
print("Missingness evidence")
display(missingness.loc[missingness["missing_count"].gt(0)])
print("No rows above means the reported final sample has no missing values in its exported fields.")
"""
            ),
            code(
                """
model_data = pd.read_csv(
    PROJECT_ROOT / "data/processed/taxi_modeling_50000.csv",
    parse_dates=["pickup_datetime", "dropoff_datetime"],
)

columns_to_show = [
    "source_row_id", "pickup_datetime", "dropoff_datetime", "fare_amount",
    "trip_distance", "duration_min", "passenger_count", "pickup_hour",
    "pickup_hour_sin", "pickup_hour_cos", "is_weekend", "split",
]
display(model_data[columns_to_show].head())
"""
            ),
            code(
                """
metadata = json.loads((PROJECT_ROOT / "data/modeling_metadata.json").read_text())
display(pd.Series(metadata, name="Value").to_frame())
"""
            ),
            code(
                """
split_summary = pd.read_csv(PROJECT_ROOT / "outputs/tables/model_split_summary.csv")
display(split_summary)

train_ids = set(model_data.loc[model_data["split"].eq("train"), "source_row_id"])
test_ids = set(model_data.loc[model_data["split"].eq("test"), "source_row_id"])

assert len(model_data) == 50_000
assert len(train_ids) == 40_000
assert len(test_ids) == 10_000
assert train_ids.isdisjoint(test_ids)
assert len(train_ids | test_ids) == 50_000
print("All split checks passed; train and test identities are disjoint and exhaustive.")
"""
            ),
        ]
    )


def build_lts_notebook() -> nbformat.NotebookNode:
    """Create the LTS validation and scalability notebook."""
    return notebook(
        [
            code(
                """
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from src.taxi_lts.validate_lts import run_validation

PROJECT_ROOT = Path.cwd()
pd.set_option("display.max_columns", 50)
"""
            ),
            code(
                """
validation = pd.Series(run_validation(), name="Value")
display(validation.to_frame())
"""
            ),
            code(
                """
errors = validation[[
    "ols_coefficient_l2_error",
    "raw_lts_coefficient_l2_error",
    "reweighted_lts_coefficient_l2_error",
]].rename(index={
    "ols_coefficient_l2_error": "OLS",
    "raw_lts_coefficient_l2_error": "Raw LTS",
    "reweighted_lts_coefficient_l2_error": "Reweighted LTS",
})

ax = errors.plot.bar(figsize=(7, 4), color=["#9B2226", "#315C7C", "#4F772D"], legend=False)
ax.set_ylabel("Euclidean coefficient error")
ax.set_title("Recovery of the known slope vector")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()

print(f"Detection recall: {validation['detection_recall']:.1%}")
print(f"Detection precision: {validation['detection_precision']:.1%}")
"""
            ),
            code(
                """
benchmark = pd.read_csv(PROJECT_ROOT / "outputs/tables/lts_scalability_benchmark.csv")
benchmark_view = benchmark[[
    "sample_size", "alpha", "h", "n_initial_subsets", "elapsed_seconds",
    "fit_peak_increase_mib", "reweighted_observations", "zero_weight_observations",
    "reweighted_fraction", "sample_source_id_sha256",
]].copy()
display(benchmark_view.style.format({
    "elapsed_seconds": "{:.2f}",
    "fit_peak_increase_mib": "{:.1f}",
    "reweighted_fraction": "{:.2%}",
}))

assert (benchmark["h"] == (0.75 * benchmark["sample_size"]).astype(int)).all()
assert benchmark["n_initial_subsets"].eq(500).all()
print("Benchmark configuration checks passed.")
"""
            ),
            code(
                """
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(benchmark["sample_size"], benchmark["elapsed_seconds"], marker="o", color="#315C7C")
axes[0].set(xlabel="Sample size", ylabel="Elapsed seconds", title="FAST-LTS elapsed time")
axes[1].plot(benchmark["sample_size"], benchmark["fit_peak_increase_mib"], marker="o", color="#4F772D")
axes[1].set(xlabel="Sample size", ylabel="Peak RSS increase MiB", title="Additional memory during fit")
plt.tight_layout()
plt.show()
"""
            ),
        ]
    )


def build_eda_notebook() -> nbformat.NotebookNode:
    """Create the complete training-only exploration notebook."""
    return notebook(
        [
            code(
                """
from pathlib import Path
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import stats

from src.taxi_lts.explore_training import (
    PREDICTORS, TARGET, coefficient_table, correlation_tables,
    descriptive_statistics, diagnostic_tables, fit_diagnostic_ols,
    load_training_data, vif_table,
)

PROJECT_ROOT = Path.cwd()
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 40)
"""
            ),
            code(
                """
input_path = PROJECT_ROOT / "data/processed/taxi_modeling_50000.csv"
training = load_training_data(input_path)

assert len(training) == 40_000
assert training["split"].eq("train").all()
assert training["source_row_id"].is_unique

fingerprint = hashlib.sha256(training["source_row_id"].to_numpy(dtype="int64").tobytes()).hexdigest()
print("Training observations:", f"{len(training):,}")
print("Training source-ID fingerprint:", fingerprint)
print("Predictors:", PREDICTORS)
"""
            ),
            code(
                """
descriptive = descriptive_statistics(training)
display(descriptive[[
    "label", "count", "missing", "mean", "median", "p95", "p99",
    "maximum", "skewness", "excess_kurtosis",
]].style.format(precision=3))
"""
            ),
            code(
                """
variables = ["fare_amount", "trip_distance", "duration_min", "passenger_count"]
labels = ["Fare amount USD", "Trip distance miles", "Duration minutes", "Passenger count"]
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for axis, variable, label in zip(axes.flat, variables, labels, strict=True):
    upper = training[variable].quantile(0.995)
    sns.histplot(training.loc[training[variable].le(upper), variable], bins=45, ax=axis, color="#315C7C")
    axis.set_title(label)
    axis.set_xlabel(f"Shown to p99.5={upper:.2f}; full maximum={training[variable].max():.2f}")
    axis.set_ylabel("Training trips")
plt.tight_layout()
plt.show()
"""
            ),
            code(
                """
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fare_limit = training[TARGET].quantile(0.995)
for axis, variable, label in zip(
    axes, ["trip_distance", "duration_min"], ["Trip distance miles", "Duration minutes"], strict=True
):
    x_limit = training[variable].quantile(0.995)
    visible = training.loc[training[variable].le(x_limit) & training[TARGET].le(fare_limit)]
    density = axis.hexbin(visible[variable], visible[TARGET], gridsize=48, mincnt=1, bins="log", cmap="Blues")
    axis.set(xlabel=label, ylabel="Fare amount USD", title=f"Fare versus {label.lower()}")
    fig.colorbar(density, ax=axis, label="Log trip density")
plt.tight_layout()
plt.show()
"""
            ),
            code(
                """
pearson, spearman = correlation_tables(training)
print("Pearson correlations")
display(pearson.round(3))
print("Spearman rank correlations")
display(spearman.round(3))

fig, axis = plt.subplots(figsize=(8, 6))
sns.heatmap(pearson, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=axis)
plt.title("Pearson correlations in the training data")
plt.tight_layout()
plt.show()
"""
            ),
            code(
                """
vif = vif_table(training)
display(vif.style.format({"vif": "{:.3f}"}))
"""
            ),
            code(
                """
ols_model, ols_hc3 = fit_diagnostic_ols(training)
coefficients = coefficient_table(ols_model, ols_hc3)
display(coefficients.style.format({
    "coefficient": "{:.4f}",
    "standard_error_conventional": "{:.4f}",
    "standard_error_hc3": "{:.4f}",
    "hc3_ci_lower_95": "{:.4f}",
    "hc3_ci_upper_95": "{:.4f}",
    "hc3_p_value": "{:.3g}",
}))
"""
            ),
            code(
                """
diagnostics, influence_summary, binned, residual_scale, influential = diagnostic_tables(training, ols_model)
diagnostic_values = diagnostics.set_index("diagnostic")["value"]

key_diagnostics = [
    "training_r_squared", "training_adjusted_r_squared", "training_mae", "training_rmse",
    "training_median_absolute_error", "residual_skewness", "residual_kurtosis",
    "breusch_pagan_lm", "breusch_pagan_lm_p_value", "ramsey_reset_f",
    "ramsey_reset_p_value", "jarque_bera", "jarque_bera_p_value",
    "shapiro_wilk_sample_5000", "shapiro_wilk_p_value", "durbin_watson",
]
display(diagnostics.loc[diagnostics["diagnostic"].isin(key_diagnostics)].style.format({"value": "{:.6g}"}))
"""
            ),
            code(
                """
fitted = np.asarray(ols_model.fittedvalues)
residuals = np.asarray(ols_model.resid)
rng = np.random.default_rng(42)
qq_residuals = rng.choice(residuals, size=5_000, replace=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
density = axes[0].hexbin(fitted, residuals, gridsize=55, mincnt=1, bins="log", cmap="Blues")
axes[0].axhline(0, color="#8B1E1E")
axes[0].set(xlabel="OLS fitted fare USD", ylabel="Residual USD", title="Residuals versus fitted values")
fig.colorbar(density, ax=axes[0], label="Log trip density")
stats.probplot(qq_residuals, dist="norm", plot=axes[1])
axes[1].set_title("Normal Q-Q plot from 5,000 residuals")
plt.tight_layout()
plt.show()
"""
            ),
            code(
                """
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for axis, variable, label in zip(
    axes, ["trip_distance", "duration_min"], ["Trip distance miles", "Duration minutes"], strict=True
):
    subset = binned.loc[binned["variable"].eq(variable)]
    axis.plot(subset["predictor_mean"], subset["residual_mean"], marker="o", color="#315C7C")
    axis.fill_between(
        subset["predictor_mean"], subset["residual_mean_ci_lower_95"],
        subset["residual_mean_ci_upper_95"], color="#315C7C", alpha=0.18,
    )
    axis.axhline(0, color="#8B1E1E")
    axis.set(xlabel="Mean within quantile bin", ylabel="Mean OLS residual USD", title=label)
plt.tight_layout()
plt.show()

display(residual_scale.style.format(precision=3))
"""
            ),
            code(
                """
display(influence_summary.T.rename(columns={0: "Value"}).style.format(precision=6))
display(influential.head(10).style.format(precision=3))
"""
            ),
        ]
    )


def build_clean_model_notebook() -> nbformat.NotebookNode:
    """Create the clean OLS and LTS comparison notebook."""
    return notebook(
        [
            code(
                """
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd()
TABLES = PROJECT_ROOT / "outputs/tables"
metadata = json.loads((PROJECT_ROOT / "outputs/clean_model_metadata.json").read_text())
display(pd.Series(metadata, name="Value").to_frame())
"""
            ),
            code(
                """
scaler = pd.read_csv(TABLES / "training_scaler_parameters.csv")
display(scaler.style.format({"training_mean": "{:.5f}", "training_scale": "{:.5f}"}))
"""
            ),
            code(
                """
performance = pd.read_csv(TABLES / "clean_model_performance.csv")
test_performance = performance.loc[performance["evaluation_split"].eq("test")].copy()
display(test_performance.style.format({
    "alpha": lambda value: "" if pd.isna(value) else f"{value:.2f}",
    "mae": "{:.3f}", "rmse": "{:.3f}",
    "median_absolute_error": "{:.3f}", "r_squared": "{:.3f}",
}))

test_performance["label"] = test_performance.apply(
    lambda row: row["estimator"] if pd.isna(row["alpha"])
    else f"{row['estimator']} a={row['alpha']:.2f}", axis=1,
)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].barh(test_performance["label"], test_performance["mae"], color="#315C7C")
axes[0].set(title="Clean test MAE", xlabel="USD")
axes[1].barh(test_performance["label"], test_performance["rmse"], color="#4F772D")
axes[1].set(title="Clean test RMSE", xlabel="USD")
plt.tight_layout()
plt.show()
"""
            ),
            code(
                """
coefficients = pd.read_csv(TABLES / "clean_model_coefficients.csv")
primary = coefficients.loc[
    coefficients["estimator"].eq("OLS") | coefficients["alpha"].eq(0.75)
]
display(primary.style.format({
    "alpha": lambda value: "" if pd.isna(value) else f"{value:.2f}",
    "coefficient_original_units": "{:.4f}",
    "ols_hc3_standard_error": lambda value: "" if pd.isna(value) else f"{value:.4f}",
}))
"""
            ),
            code(
                """
display(test_performance[[
    "estimator", "alpha", "mae", "rmse", "median_absolute_error", "r_squared"
]].style.format(precision=3))
"""
            ),
            code(
                """
audit_summary = pd.read_csv(TABLES / "lts_audit_candidate_summary.csv")
candidates = pd.read_csv(TABLES / "lts_audit_candidates.csv")
display(audit_summary.style.format(precision=4))
display(candidates.head(15).style.format(precision=3))
"""
            ),
        ]
    )


def build_contamination_notebook() -> nbformat.NotebookNode:
    """Create the controlled-contamination analysis notebook."""
    return notebook(
        [
            code(
                """
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd()
TABLES = PROJECT_ROOT / "outputs/tables"
metadata = json.loads((PROJECT_ROOT / "outputs/contamination_metadata.json").read_text())
display(pd.Series(metadata, name="Value").to_frame())
"""
            ),
            code(
                """
performance = pd.read_csv(TABLES / "contamination_model_performance.csv")
degradation = pd.read_csv(TABLES / "contamination_prediction_degradation.csv")
display(performance.style.format(precision=3))
display(degradation.style.format(precision=3))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for axis, metric in zip(axes, ["mae", "rmse"], strict=True):
    pivot = performance.pivot(index="estimator", columns="training_scenario", values=metric)
    pivot.plot.bar(ax=axis, color=["#315C7C", "#9B2226"])
    axis.set(title=f"Clean-test {metric.upper()}", ylabel="USD", xlabel="")
    axis.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()
"""
            ),
            code(
                """
coefficient_change = pd.read_csv(TABLES / "contamination_coefficient_change.csv")
display(coefficient_change.style.format(precision=4))

slopes = coefficient_change.loc[coefficient_change["term"].ne("const")]
pivot = slopes.pivot(index="term", columns="estimator", values="absolute_change_magnitude")
pivot.plot.bar(figsize=(9, 4.5), color=["#9B2226", "#315C7C"])
plt.ylabel("Absolute coefficient change")
plt.title("Coefficient stability after +$100 contamination")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()
"""
            ),
            code(
                """
detection = pd.read_csv(TABLES / "contamination_detection_summary.csv")
detail = pd.read_csv(TABLES / "contamination_detection_detail.csv")
display(detection.style.format(precision=4))
display(detail.loc[detail["known_artificial_contamination"]].head(15).style.format(precision=3))
"""
            ),
            code(
                """
from IPython.display import Image
from src.taxi_lts.plot_actual_points import save_comparable_trip_plots
from src.taxi_lts.stress_test import contaminate_training_response

modeling = pd.read_csv(PROJECT_ROOT / "data/processed/taxi_modeling_50000.csv")
training = modeling.loc[modeling["split"].eq("train")].copy()
_, known_mask = contaminate_training_response(training)
coefficients = pd.read_csv(TABLES / "contamination_coefficient_change.csv")
plot_details = save_comparable_trip_plots(
    training, coefficients, known_mask, PROJECT_ROOT / "outputs/figures"
)
display(pd.Series(plot_details, name="Display design").to_frame())
display(Image(filename=PROJECT_ROOT / "outputs/figures/actual_contamination_overview.png"))
display(Image(filename=PROJECT_ROOT / "outputs/figures/actual_points_clean_stress_fitted_lines.png"))
"""
            ),
        ]
    )


def execute_and_write(document: nbformat.NotebookNode, filename: str) -> None:
    """Execute a notebook from the project root and write the executed file."""
    client = NotebookClient(
        document,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    client.execute()
    output_path = NOTEBOOK_DIR / filename
    nbformat.write(document, output_path)
    print(output_path)


def main() -> None:
    """Build and execute all completed-stage analysis notebooks."""
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    execute_and_write(
        build_data_notebook(), "01_data_preparation_sampling_and_split.ipynb"
    )
    execute_and_write(
        build_lts_notebook(), "02_fast_lts_validation_and_scalability.ipynb"
    )
    execute_and_write(
        build_eda_notebook(), "03_training_eda_and_model_specification.ipynb"
    )
    execute_and_write(
        build_clean_model_notebook(), "04_clean_ols_and_lts_comparison.ipynb"
    )
    execute_and_write(
        build_contamination_notebook(), "05_contamination_stress_test.ipynb"
    )


if __name__ == "__main__":
    main()
