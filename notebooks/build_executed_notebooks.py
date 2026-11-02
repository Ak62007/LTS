"""Build and execute the teaching notebooks for all completed project stages."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def markdown(text: str):
    """Return a Markdown notebook cell."""
    # The source contains readable LaTeX in ordinary triple-quoted strings.
    # Restore the few backslash sequences Python interprets as control codes.
    for control, latex in {
        "\a": r"\a",
        "\b": r"\b",
        "\f": r"\f",
        "\r": r"\r",
        "\t": r"\t",
        "\v": r"\v",
    }.items():
        text = text.replace(control, latex)
    return nbformat.v4.new_markdown_cell(text.strip())


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
            markdown(
                """
# 01 — Data provenance, filtering, sampling, and protected splitting

**Project:** Robust Detection of Anomalous Taxi Fares Using Least Trimmed Squares Regression  
**Student:** Aditya Kumar — IED/10007/23

This notebook explains how the January 2026 NYC Yellow Taxi data became the fixed modelling dataset. It verifies the saved evidence without repeatedly loading the entire 3.7-million-row Parquet file.

## Learning objectives

1. Establish exactly where the data came from and verify file integrity.
2. Explain every eligibility filter and why no upper limits were imposed.
3. Explain deterministic sampling and why the project moved from 10,000 to 50,000 trips.
4. Verify the 40,000/10,000 train-test split without inspecting test outcomes.
"""
            ),
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
            markdown(
                """
## 1. Official source and integrity

The raw file is the official January 2026 Yellow Taxi Parquet published by the New York City Taxi and Limousine Commission. A SHA-256 checksum acts like a digital fingerprint: if one byte changes, the checksum changes. This lets us prove that later analysis used the preserved raw download.
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
            markdown(
                """
## 2. Eligibility rules

Filters were applied sequentially, so each removal count is conditional on all earlier rules.

| Rule | Statistical/business reason |
|---|---|
| Pickup occurs in January 2026 | Keeps the requested reporting period. |
| `RatecodeID == 1` | Restricts the case to standard metered fares; negotiated, airport, and other tariff regimes follow different rules. |
| Distance > 0 | A distance-based fare model cannot interpret zero or negative recorded distance normally. |
| Fare > 0 | The response must be a positive passenger fare; refunds and corrections represent a different process. |
| Duration ≥ 1 minute | Removes zero/negative timestamp errors while retaining long trips. |
| Passenger count ≥ 1 | Removes missing/zero occupancy records from the passenger-fare case. |

**Important modification:** there is no maximum duration and no maximum passenger-count filter. There is also no upper fare or distance filter. Extreme observations remain available for LTS to assess rather than being deleted in advance.
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
            markdown(
                """
## 3. Feature engineering

Duration is computed from timestamps:

\[
\text{duration minutes}=\frac{\text{drop-off time}-\text{pickup time}}{60\text{ seconds}}.
\]

Clock hour is circular: hour 23 is close to hour 0, although ordinary numeric coding makes them look far apart. We therefore encode hour using

\[
\sin(2\pi h/24),\qquad \cos(2\pi h/24).
\]

Together, these two terms represent a smooth 24-hour cycle. `is_weekend` equals 1 for Saturday or Sunday and 0 otherwise.
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
            markdown(
                """
## 4. Why 50,000 observations rather than only 10,000

The first 10,000-row sample was a conservative planning sample. A guarded FAST-LTS benchmark subsequently showed that 50,000 rows were computationally safe on this device. The modelling sample was therefore frozen at 50,000 to represent rare tail patterns better while remaining reproducible and manageable.

Selection uses the first 50,000 positions of one NumPy seed-42 permutation of all 2,278,082 eligible trips. This is deterministic: rerunning the same code and data produces the same source-row checksum.
"""
            ),
            code(
                """
metadata = json.loads((PROJECT_ROOT / "data/modeling_metadata.json").read_text())
display(pd.Series(metadata, name="Value").to_frame())
"""
            ),
            markdown(
                """
## 5. Protected 80/20 split

The 50,000 selected rows were randomly assigned using seed 42:

- 40,000 training rows for exploration, preprocessing, specification, and fitting.
- 10,000 test rows reserved for one final out-of-sample evaluation.

The split happens before model exploration. Test outcomes must not influence transformation choices, predictor selection, or anomaly thresholds.
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
            markdown(
                """
## What to explain to the teacher

1. The checksum proves which raw file was analysed.
2. Filters define a coherent standard-rate business process; they do not manufacture a clean linear relationship.
3. No upper-bound filtering removes the anomalies that robust regression is supposed to study.
4. Seed 42 and source-row checksums make selection and splitting reproducible.
5. The test set is a locked examination set, not a source of modelling decisions.
"""
            ),
        ]
    )


def build_lts_notebook() -> nbformat.NotebookNode:
    """Create the LTS validation and scalability notebook."""
    return notebook(
        [
            markdown(
                """
# 02 — FAST-LTS validation and device-safe scalability

This notebook demonstrates why the local FAST-LTS implementation is statistically credible and why 50,000 taxi observations are safe to use.

## LTS objective

For residuals \(r_i(\beta)=y_i-x_i^T\beta\), Least Trimmed Squares sorts the squared residuals and minimizes only the smallest \(h\):

\[
\hat\beta_{LTS}=\arg\min_\beta\sum_{i=1}^{h}r_{(i)}^2(\beta).
\]

With \(\alpha=0.75\), \(h=\lfloor0.75n\rfloor\). The largest 25% of squared residuals cannot directly determine the raw fit. FAST-LTS searches efficiently using random elemental subsets and C-steps.
"""
            ),
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
            markdown(
                """
## 1. Controlled synthetic validation

We generate data from a known linear model with coefficients \((2,-1.5,0.75)\), Gaussian noise with standard deviation 0.2, and then add exactly 50 to 100 of 1,000 responses. Because the truth and contaminated rows are known, we can measure coefficient recovery and anomaly detection directly.
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
            markdown(
                """
### Interpretation

OLS moves toward the contaminated responses: its intercept is about 9.98 instead of 5.00 and its slope-vector error is about 0.77. Raw and reweighted LTS recover the known relationship with errors near 0.02. Reweighting flags all 100 contaminated responses, plus 16 clean observations, giving 100% recall and about 86.2% precision. These flags are screening decisions, not proof of wrongdoing.

## 2. Transparent reweighting

After raw FAST-LTS, residuals are standardized using the robust median absolute deviation (MAD). A row receives weight 1 when

\[
|r_i / \widehat\sigma_{MAD}| \leq \sqrt{\chi^2_{1,0.975}}\approx2.241,
\]

and weight 0 otherwise. Ordinary least squares is then refitted to weight-1 rows. This improves efficiency while keeping the resistant raw fit as the reference.
"""
            ),
            markdown(
                """
## 3. Guarded scalability benchmark

The benchmark used nested seed-42 samples of 10,000, 25,000, and 50,000 eligible taxi trips, 500 initial subsets, and a two-thread limit. Each size was run separately while monitoring elapsed time and process memory. We display the preserved benchmark rather than repeatedly spending resources rerunning all three fits.
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
            markdown(
                """
## Decision

The 50,000-row fit completed in about 7.18 seconds with approximately 213 MiB additional peak RSS under the guarded benchmark. Therefore 50,000—not only 10,000—was selected as the final modelling sample. It better represents rare patterns while remaining well inside the device's demonstrated capability.

The later model comparison will fit raw and reweighted FAST-LTS at primary \(\alpha=0.75\), then check \(0.50\) and \(0.90\) for sensitivity.
"""
            ),
        ]
    )


def build_eda_notebook() -> nbformat.NotebookNode:
    """Create the complete training-only exploration notebook."""
    return notebook(
        [
            markdown(
                """
# 03 — Training-only exploration and model-specification assessment

This notebook reproduces every statistical calculation used in the stage report. It immediately isolates the 40,000 training records and never summarizes, plots, or models the 10,000 test outcomes.

## Questions answered

1. What do the training distributions and tails look like?
2. How are fare and the proposed predictors associated?
3. Is multicollinearity severe?
4. What does a diagnostic OLS fit reveal?
5. Do linearity, constant variance, normality, and influence assumptions look credible?
6. What specification should be frozen before test evaluation?
"""
            ),
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
            markdown(
                """
## 1. Lock the analytical scope to training data

`load_training_data` filters to `split == "train"` immediately and checks that exactly 40,000 unique source IDs remain. We retain raw units for interpretation; scaling will be learned from training data only during model fitting.
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
            markdown(
                """
## 2. Descriptive statistics and tail behaviour

Means alone are insufficient for skewed trip data, so we report medians, percentiles, maxima, skewness, and excess kurtosis. Plots stop at the 99.5th percentile only for readability; maxima are printed and every observation remains in every calculation.
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
            markdown(
                """
**Interpretation.** Fare, distance, and duration are right-skewed. Duration is exceptionally heavy-tailed: its median is about 11.58 minutes but its maximum is 1,437.05 minutes. Such observations may be valid, interrupted, or erroneous; they are retained so robust regression can assess compatibility with the dominant pattern.
"""
            ),
            markdown(
                """
## 3. Fare relationships

Hexagonal density plots are preferable to ordinary scatterplots because 40,000 points overlap heavily. Darker cells contain more trips.
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
            markdown(
                """
Distance has a narrow positive relationship with fare. The fare-duration cloud spreads outward for longer trips, suggesting nonconstant error variance. Parallel fare bands probably reflect tariff and operational details not represented by the current predictors.

## 4. Pearson versus Spearman association

- **Pearson** measures linear association and is sensitive to extreme magnitudes.
- **Spearman** computes Pearson correlation on ranks, measuring monotonic association with less sensitivity to magnitude.

A large difference between them is evidence that the tail geometry matters.
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
            markdown(
                """
Fare-distance correlation is high under both measures (Pearson 0.953, Spearman 0.909). Fare-duration correlation changes from Pearson 0.494 to Spearman 0.967. Ordinary trips therefore follow a very strong monotonic duration-fare pattern, while a few extreme durations do not carry proportionally extreme fares.

## 5. Variance inflation factors

For predictor \(j\),

\[
VIF_j=\frac{1}{1-R_j^2},
\]

where \(R_j^2\) comes from regressing predictor \(j\) on all other predictors. Large VIF means the slope is difficult to distinguish from linear combinations of the remaining predictors.
"""
            ),
            code(
                """
vif = vif_table(training)
display(vif.style.format({"vif": "{:.3f}"}))
"""
            ),
            markdown(
                """
All predictor VIFs are between about 1.01 and 1.21. Conventional linear multicollinearity is therefore low. This does not contradict the large Spearman distance-duration correlation because VIF concerns linear predictability and is affected by the extreme durations.

## 6. Diagnostic OLS fit and HC3 uncertainty

The provisional additive conditional-mean model is

\[
fare_i=\beta_0+\beta_1 distance_i+\beta_2 duration_i+\beta_3 passengers_i
+\beta_4\sin(hour_i)+\beta_5\cos(hour_i)+\beta_6 weekend_i+\varepsilon_i.
\]

OLS minimizes all squared residuals, so large residuals and high-leverage predictor values can dominate it. HC3 standard errors inflate each squared residual according to leverage and are more credible under heteroskedasticity than conventional OLS standard errors.
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
            markdown(
                """
Holding other included predictors fixed, the provisional slopes are about $3.62 per mile and $0.08 per minute. They are conditional associations, not causal effects. HC3 errors are roughly 16 times the conventional distance error and 39 times the conventional duration error, demonstrating how misleading homoskedastic uncertainty would be.

## 7. Fit and assumption diagnostics
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
            markdown(
                """
### How to read the tests

- **Breusch–Pagan:** null hypothesis is constant residual variance. The enormous statistic rejects it.
- **Ramsey RESET:** tests whether powers of fitted values add explanatory information. Rejection indicates omitted nonlinearity or interactions.
- **Jarque–Bera and Shapiro–Wilk:** test residual normality. Both reject strongly; graphical magnitude matters because \(n=40{,}000\) makes tiny deviations significant.
- **Durbin–Watson:** approximately 2 in the stored random order. This does not prove temporal independence because rows are not one continuous time series.

The in-sample \(R^2=0.9269\), MAE=$2.07, and RMSE=$3.28 describe training fit only—not test performance.
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
            markdown(
                """
The funnel-shaped residual plot confirms increasing variance. The Q-Q plot has heavy asymmetric tails, consistent with residual skewness −3.21 and kurtosis 187.4. Normal-theory OLS inference is therefore unsuitable without qualification.

## 8. Functional form using binned residuals

If the conditional mean is well specified, mean residuals should remain near zero across predictor values. Quantile bins provide a readable nonparametric check without fitting a new model.
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
            markdown(
                """
Distance residual means curve from negative to positive and back to negative. Duration residual means increase from approximately −$2.48 to +$4.64. Residual RMSE rises from $1.91 in the lowest fitted-fare decile to $6.22 in the highest. The additive linear mean is useful but incomplete, and the variance is clearly nonconstant.

## 9. Leverage, studentized residuals, and Cook's distance

- **Leverage** measures how unusual a predictor vector is. Screening threshold: \(2p/n\).
- **Studentized residual** measures response disagreement relative to an estimated residual scale. Screening rule: absolute value above 3.
- **Cook's distance** estimates how much the fitted coefficient vector changes when one row is removed. Screening threshold: \(4/n\).

These are diagnostic screens, never automatic deletion rules.
"""
            ),
            code(
                """
display(influence_summary.T.rename(columns={0: "Value"}).style.format(precision=6))
display(influential.head(10).style.format(precision=3))
"""
            ),
            markdown(
                """
There are 2,038 high-leverage rows, 1,663 rows above the Cook's-distance screen, and 575 with absolute internally studentized residual above 3. The four largest Cook's distances belong to reported durations around 1,412–1,437 minutes with relatively small fares. They remain in the data for robust modelling and possible audit review.

## 10. Frozen model-specification decision

The primary comparison will keep the same predictors for OLS and LTS so that the estimator—not a different feature set—is being compared:

1. Distance, duration, passenger count, pickup-hour sine/cosine, and weekend status.
2. Fit all scaling parameters on the training records only and apply them unchanged to the test set.
3. Report OLS with HC3 uncertainty.
4. Fit raw and explicitly reweighted FAST-LTS at \(\alpha=0.75\); check \(0.50\) and \(0.90\).
5. Keep extreme observations. Call robustly flagged rows **audit candidates**, not fraud.
6. After robust fitting, repeat binned-residual diagnostics. If curvature remains, add a training-prespecified nonlinear sensitivity model without consulting test outcomes.

The next phase is clean OLS and FAST-LTS fitting, followed by one evaluation on the untouched test set. The controlled +$100 contamination experiment comes only after clean results are frozen.
"""
            ),
            markdown(
                """
## Short oral explanation

> The training data have a strong fare-distance relationship, but ordinary least squares assumptions are not credible. Residual variance grows with predicted fare, residual means show curvature, residual tails are highly non-normal, and a few extreme durations have enormous influence. Low VIF values show this is not mainly a multicollinearity problem. We therefore retain an interpretable common linear specification, use HC3 uncertainty for OLS, and compare it with resistant FAST-LTS while preserving all extreme records for audit assessment.
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
    """Build and execute all completed-stage teaching notebooks."""
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


if __name__ == "__main__":
    main()
