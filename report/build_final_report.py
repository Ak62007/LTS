"""Build the final ED409 report from executed evidence and verified tables."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from report.create_training_eda_report import (  # noqa: E402
    add_body,
    add_figure,
    add_heading,
    add_page_number,
    add_table,
)

ASSETS = ROOT / "report/assets/final"
TABLES = ROOT / "outputs/tables"
FIGURES = ROOT / "outputs/figures"


def evidence() -> dict[str, str]:
    """Execute the code shown in the report and save an evidence notebook."""
    sources = {
        "provenance": """import json
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from src.taxi_lts.run_analysis import load_protected_split
from src.taxi_lts.explore_training import PREDICTORS, TARGET
from src.taxi_lts.stress_test import fit_pair, contaminate_training_response
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
root = Path.cwd()
train, test = load_protected_split(root / "data/processed/taxi_modeling_50000.csv")
flow = pd.read_csv(root / "outputs/tables/data_quality_flow.csv")
display(flow.iloc[:-1])
print("Training", len(train), "Test", len(test))""",
        "fit": """scaler = StandardScaler().fit(train[PREDICTORS])
X = scaler.transform(train[PREDICTORS])
y = train[TARGET].to_numpy(dtype=float)
with threadpool_limits(limits=2):
    clean_ols, clean_lts = fit_pair(X, y)
print("Raw LTS h:", len(clean_lts.raw_estimator.best_h_subset))
print("MAD scale:", round(clean_lts.raw_robust_scale, 6))
print("Retained for reweighting:", clean_lts.reweighting_weights.sum())""",
        "lts_algorithm": """from robpy.regression import FastLTSRegression
from scipy.stats import median_abs_deviation, chi2
from sklearn.linear_model import LinearRegression
import logging
with threadpool_limits(limits=2):
    raw = FastLTSRegression(alpha=0.75, n_initial_subsets=500,
                            reweighting=False, random_state=42)
    raw.fit(X, y, verbosity=logging.WARNING)
r = y - raw.predict(X).reshape(-1)
scale = median_abs_deviation(r, scale="normal")
cutoff = np.sqrt(chi2.ppf(0.975, df=1))
weights = np.abs(r / scale) <= cutoff
reweighted = LinearRegression().fit(X[weights], y[weights])
assert np.allclose(reweighted.coef_, clean_lts.reweighted_model.coef_)
print("h", len(raw.best_h_subset))
print("Scale", round(scale, 6), "Weight 1", int(weights.sum()))""",
        "diagnostics": """from src.taxi_lts.explore_training import fit_diagnostic_ols, coefficient_table
diagnostic_ols, hc3 = fit_diagnostic_ols(train)
display(coefficient_table(diagnostic_ols, hc3).round(4))""",
        "clean": """from src.taxi_lts.run_analysis import prediction_metrics
Xt = scaler.transform(test[PREDICTORS])
yt = test[TARGET].to_numpy(dtype=float)
rows = []
for name, model in [("OLS", clean_ols),
                    ("Reweighted LTS", clean_lts.reweighted_model)]:
    rows.append({"Estimator": name, **prediction_metrics(yt, model.predict(Xt))})
display(pd.DataFrame(rows).round(6))""",
        "stress": """ys, known = contaminate_training_response(train)
assert known.sum() == 2000
assert np.allclose(ys[known] - y[known], 100)
assert np.array_equal(ys[~known], y[~known])
with threadpool_limits(limits=2):
    stressed_ols, stressed_lts = fit_pair(X, ys)
rows = []
for name, model in [("OLS", stressed_ols),
                    ("Reweighted LTS", stressed_lts.reweighted_model)]:
    rows.append({"Estimator": name, **prediction_metrics(yt, model.predict(Xt))})
display(pd.DataFrame(rows).round(6))""",
        "detection": """flag = (~stressed_lts.reweighting_weights) | (
    np.abs(stressed_lts.raw_standardized_residuals) > 2.5)
tp = int(np.sum(flag & known))
fp = int(np.sum(flag & ~known))
fn = int(np.sum(~flag & known))
print("TP", tp, "FP", fp, "FN", fn)
print("Recall", tp / known.sum())
print("Precision", round(tp / (tp + fp), 6))""",
    }
    for name, source in sources.items():
        sources[name] = subprocess.run(
            [
                str(ROOT / ".venv/bin/ruff"),
                "format",
                "--stdin-filename",
                "excerpt.py",
                "--line-length",
                "85",
                "-",
            ],
            input=source,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# Final report execution evidence\nThese are the exact code excerpts imaged in the report. "
            "The calls execute project functions; their definitions are in src/taxi_lts."
        )
    ]
    for label, source in sources.items():
        nb.cells.extend(
            [
                nbformat.v4.new_markdown_cell(f"## {label.title()}"),
                nbformat.v4.new_code_cell(source),
            ]
        )
    NotebookClient(
        nb, timeout=300, resources={"metadata": {"path": str(ROOT)}}, allow_errors=False
    ).execute()
    nbformat.write(nb, ROOT / "notebooks/06_final_report_execution_evidence.ipynb")
    for name, source in sources.items():
        lines = source.splitlines()
        fig = plt.figure(figsize=(10.6, 0.23 * len(lines) + 0.35))
        fig.text(
            0.025,
            0.96,
            source,
            va="top",
            family="DejaVu Sans Mono",
            fontsize=12.5,
            linespacing=1.15,
        )
        fig.savefig(ASSETS / f"code_{name}.png", dpi=210, facecolor="#F4F6F8")
        plt.close(fig)
    return sources


def table(
    d: Document, filename: str, columns: list[str], headers: list[str], select=None
) -> None:
    """Insert selected source values with restrained precision."""
    frame = pd.read_csv(TABLES / filename)
    if select is not None:
        frame = select(frame)
    rows = []
    for row in frame[columns].itertuples(index=False, name=None):
        rows.append(
            [
                "" if pd.isna(v) else f"{v:,.3f}" if isinstance(v, float) else str(v)
                for v in row
            ]
        )
    add_table(d, headers, rows, font_size=9)


def p(d: Document, text: str) -> None:
    add_body(d, text)


def code(d: Document, name: str) -> None:
    add_figure(
        d,
        ASSETS / f"code_{name}.png",
        f"Executed code excerpt from notebook 06   {name}",
        width=6.85,
    )


def page(d: Document, title: str) -> None:
    d.add_page_break()
    add_heading(d, title)


def equation(d: Document, name: str, expression: str) -> None:
    """Insert a sharp mathematical expression at a readable size."""
    fig = plt.figure(figsize=(9.5, 0.6))
    fig.text(0.03, 0.25, expression, fontsize=19)
    path = ASSETS / f"equation_{name}.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    d.add_picture(str(path), width=Inches(3.3))


def build(execute: bool = True) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    if execute:
        evidence()
    d = Document()
    sec = d.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin = sec.bottom_margin = Inches(0.65)
    sec.left_margin = sec.right_margin = Inches(0.72)
    normal = d.styles["Normal"]
    normal.font.name, normal.font.size = "Arial", Pt(10.5)
    normal.paragraph_format.line_spacing = 1.08
    normal.paragraph_format.space_after = Pt(7)
    for name, size in [("Title", 19), ("Heading 1", 15), ("Heading 2", 11.5)]:
        style = d.styles[name]
        style.font.name, style.font.size = "Arial", Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(7)
        style.paragraph_format.space_after = Pt(7)
        style.paragraph_format.keep_with_next = True
        borders = style.element.xpath(".//w:pBdr")
        for border in borders:
            border.getparent().remove(border)
    add_page_number(sec.footer.paragraphs[0])
    d.core_properties.author = "Aditya Kumar"
    d.core_properties.title = "Robust Detection of Anomalous Taxi Fares Using Least Trimmed Squares Regression"

    d.add_paragraph(d.core_properties.title, "Title")
    p(
        d,
        "Aditya Kumar | IED/10007/23 | ED409 Regression Techniques\nInstructor Dr. Tina Dutta | Assigned topic Least Trimmed Squares regression",
    )
    add_heading(d, "Business case and dataset", 1)
    p(
        d,
        "A taxi operator or transport audit team needs a reliable account of how recorded metered fares vary with trip characteristics. The team must distinguish the fare relationship followed by ordinary trips from transactions that deserve a closer look. A very large fare can be reasonable for a long journey, while a small fare paired with a nearly day-long recorded duration may indicate an interrupted meter, a timestamp problem, or an unusual operation. A rule based only on the fare amount would miss this distinction.",
    )
    p(
        d,
        "The business decision is to establish a stable descriptive fare benchmark and use departures from it to prioritize manual review. Ordinary least squares (OLS) is a useful baseline, but squaring every residual allows a few unusual records to affect its coefficients. Least Trimmed Squares (LTS) offers a resistant benchmark based on the observations most compatible with a common regression relationship. Comparing the two exposes how the apparent fare drivers depend on extreme observations.",
    )
    add_heading(d, "Official data source and observation unit", 2)
    p(
        d,
        "The source is the New York City Taxi and Limousine Commission (TLC), January 2026 Yellow Taxi Trip Records, downloaded on 1 November 2026 from the official TLC Trip Record Data portal. The original Parquet file contains 3,724,889 rows. Each row represents a submitted taxi trip record, not a passenger, driver, or taxi observed repeatedly with an identifiable panel key. TLC publishes records supplied by authorized technology providers and does not guarantee their accuracy (NYC TLC, n.d.-a).",
    )
    p(
        d,
        "Official portal: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page\nFile: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet",
    )
    p(
        d,
        "The outcome is fare_amount, measured in US dollars: the time-and-distance amount recorded by the meter. It differs from the total passenger bill; tips, tolls, and separately itemized extras are separate fields. Predictors are distance in miles, duration calculated from pickup and drop-off timestamps, passenger count, a daily pickup-hour cycle, and weekend status. RatecodeID = 1 identifies the standard-rate population (NYC TLC, 2025).",
    )
    p(
        d,
        "After basic eligibility checks, 2,278,082 trips remain. A deterministic random sample of 50,000 is used: 40,000 for analysis and estimation and 10,000 for evaluation. No upper limit is imposed on duration, passenger count, distance, or fare. Thus the case preserves extreme records whose effect is central to LTS. These data describe January standard-rate trips; they do not represent every taxi tariff or every month.",
    )

    page(d, "Business problems and questions")
    p(
        d,
        "The proposed user is an audit analyst deciding which fare patterns to trust and which records to examine. The available records contain neither adjudicated fraud labels nor independently verified correct fares. The exercise therefore evaluates a regression-based review process and a controlled experiment rather than estimating the prevalence of wrongdoing.",
    )
    add_heading(d, "Question 1   What describes the dominant fare relationship", 2)
    p(
        d,
        "How does recorded metered fare vary with distance, duration, passenger count, pickup time, and weekend status when these characteristics are considered together? The analyst needs coefficients in recognizable units, such as dollars per mile and dollars per minute, to understand the benchmark. OLS and LTS will be fitted to the same predictor set so differences can be attributed to estimation rather than different information. The coefficients describe associations, not causal tariff effects or legally correct prices.",
    )
    add_heading(d, "Question 2   How sensitive is OLS to unusual genuine trips", 2)
    p(
        d,
        "Do a small number of unusual duration-fare combinations pull the OLS benchmark away from the pattern followed by most trips? Training diagnostics will examine unusual predictor values, large residuals, and influence on the fit. Clean OLS and LTS will then be compared using mean absolute error, median absolute error, root mean squared error, and R-squared on the same evaluation observations. Reporting all four measures prevents a misleading declaration of a universal winner when typical errors and extreme errors favour different estimators.",
    )
    add_heading(d, "Question 3   Which records warrant review", 2)
    p(
        d,
        "Which training records disagree strongly with the resistant fare relationship, and how many would enter an audit queue? Raw LTS residuals and explicit reweighting decisions will provide reproducible screening rules. The analyst should examine timestamps, trip attributes, and recording context before taking action. The deliverable is a ranked set of audit candidates, including both unusually high and unusually low fares relative to the model. A flag alone cannot establish overcharging, a recording error, or fraud.",
    )
    add_heading(d, "Question 4   Does the benchmark survive controlled corruption", 2)
    p(
        d,
        "If 5% of training fares are artificially increased by exactly $100, how much do OLS and LTS coefficients and predictions change? Exactly 2,000 training rows are selected with seed 42. Their predictor values remain fixed, and both refitted models are evaluated against the original 10,000 test fares. Because the corrupted rows are known, the experiment can also measure screening recall and precision. This separates resistance to a known artificial shift from the uncertainty surrounding unusual genuine records.",
    )
    add_heading(d, "Business value and boundaries", 2)
    p(
        d,
        "A useful outcome is a benchmark that is interpretable, resistant to recording anomalies, and explicit about review workload. The proposed application supports audit prioritization and descriptive fare monitoring. It does not automate fare enforcement. The experiment is limited to upward response corruption in one fixed sample; it cannot establish robustness to every possible anomaly, seasonal change, or coordinated manipulation of predictors. The solution begins on the following page.",
    )

    page(d, "Solution and data preparation")
    p(
        d,
        "The preparation process attaches a stable source_row_id to each original row. Filters are applied sequentially, so each removal count is conditional on the earlier rules. The code below reads the protected split and displays the saved preparation audit; the full filtering implementation is in src/taxi_lts/prepare_data.py.",
    )
    code(d, "provenance")
    table(
        d,
        "data_quality_flow.csv",
        ["stage", "records_removed", "records_after"],
        ["Stage", "Removed", "Remaining"],
        lambda x: x.iloc[:-1],
    )
    p(
        d,
        "The January pickup rule fixes the study period; rate code 1 restricts the tariff regime. Positive distance and fare exclude records incompatible with the target positive-fare process. Duration of at least one minute removes nonpositive and extremely short timestamp combinations; passenger count of at least one defines occupied trips. Missing values in required filter fields fail the relevant rule. None of these rules removes a trip merely for being large.",
    )

    page(d, "Variables and modelling design")
    add_table(
        d,
        ["Field", "Definition and role"],
        [
            ["fare_amount", "Response in USD; meter fare, distinct from total_amount"],
            ["trip_distance", "Predictor; taximeter distance in miles"],
            ["duration_min", "Predictor; (drop-off minus pickup) in minutes"],
            ["passenger_count", "Predictor; reported passengers"],
            [
                "pickup_hour_sin and cos",
                "Predictors; sin(2 pi hour / 24) and cos(2 pi hour / 24)",
            ],
            ["is_weekend", "Predictor; Saturday or Sunday = 1, otherwise 0"],
            [
                "source_row_id",
                "Original record position for traceability; excluded from model",
            ],
            ["split", "Fixed train or test label; excluded from model"],
        ],
        font_size=9.5,
    )
    p(
        d,
        "Sine and cosine preserve the circular nature of time: 23:00 is close to 00:00. They enter together and represent one smooth daily cycle, rather than separate hourly effects. Duration and distance are observable at trip completion, so this is a retrospective fare analysis rather than a pickup-time quote model.",
    )
    table(
        d,
        "training_descriptive_statistics.csv",
        ["label", "median", "p99", "maximum"],
        ["Training variable", "Median", "99th percentile", "Maximum"],
        lambda x: x.iloc[:4],
    )
    p(
        d,
        "All 40,000 training records have complete exported analytical fields. The duration maximum is 1,437.05 minutes although the median is only 11.58. Retaining these records permits direct examination of resistance. Zero upper filtering is a modelling choice, not a guarantee that the extreme values are correct.",
    )
    p(
        d,
        "A seed-42 permutation selects 50,000 eligible trips; a seed-42 random 80/20 split fixes their training and test labels. Standardization uses training means and standard deviations only. The response stays in dollars. The earlier 10,000-row preparation sample is retained as an exploratory artifact; the results here use the later fixed 50,000-row sample.",
    )
    p(
        d,
        "Validation boundary: an earlier feasibility benchmark fitted nested samples up to all 50,000 rows before the final split. It assessed computing time and memory, and no test-based tuning was performed in the final comparison. Nevertheless, the evaluation is an internal random holdout, not an independently collected external validation set. No new predictors or thresholds were selected after the reported test comparison.",
    )

    page(d, "OLS and Least Trimmed Squares")
    p(
        d,
        "Both estimators use the same additive mean specification with six predictors and an intercept: fare = beta0 + beta1 distance + beta2 duration + beta3 passengers + beta4 sin(2 pi hour / 24) + beta5 cos(2 pi hour / 24) + beta6 weekend + error. Predictors are standardized for fitting and coefficients are returned to original units for interpretation.",
    )
    p(
        d,
        "For residual r_i(beta) = y_i - x_i' beta, OLS minimizes the sum of r_i(beta)^2 over all n observations. Large residuals receive a quadratic penalty, and unusual predictor values can have high influence. LTS instead orders squared residuals as r_(1)^2 <= ... <= r_(n)^2 and minimizes the sum from i = 1 to h of r_(i)^2. The selected subset depends on the candidate fit; it is not the h smallest fares (Rousseeuw & Van Driessen, 2006).",
    )
    equation(
        d,
        "lts",
        r"$\hat{\beta}_{\mathrm{LTS}}=\arg\min_{\beta}\sum_{i=1}^{h}r_{(i)}^{2}(\beta)$",
    )
    p(
        d,
        "The primary retained fraction is alpha = 0.75, giving h = 30,000 of 40,000 training rows. Smaller h generally increases resistance at a cost in efficiency under an ideal model. The choice does not assert that 25% of trips are bad. Sensitivity settings 0.50 and 0.90 were fixed beforehand. At 0.50, RobPy enforces its parameter-dependent minimum h = 20,003; at 0.90, h = 36,000.",
    )
    add_heading(d, "FAST LTS computation", 2)
    p(
        d,
        "RobPy 0.0.6 starts from 500 random subsets with seed 42. For a trial fit, a concentration step selects the h observations with smallest squared residuals and refits least squares to that subset. Repeating such steps decreases the trimmed objective until convergence. Promising starts receive further iterations; the best trimmed objective is retained. The search is approximate on this large sample and does not prove a global optimum (RobPy, n.d.).",
    )
    add_heading(d, "Explicit reweighting used in this project", 2)
    p(
        d,
        "RobPy is called with built-in reweighting disabled. After raw LTS, the project computes s = median(|r_i - median(r)|) / 0.67449 across all training residuals. It uses z_i = r_i / s and weight w_i = 1 when |z_i| <= 2.241403, otherwise zero. Least squares is then refitted to weight-1 observations. The cutoff is sqrt(chi-square quantile 0.975 with 1 degree of freedom). This normal-consistent MAD rule is an explicit implementation choice, not a claim of exact standard LTS finite-sample calibration.",
    )
    p(
        d,
        "For the clean primary fit, raw MAD scale is $0.510606 and 33,548 rows receive weight 1. Raw and reweighted results are reported separately. The reweighted model is the main LTS comparison. No conventional OLS significance tests are applied to its selected-subset coefficients.",
    )

    page(d, "Executed LTS algorithm and robust reweighting")
    p(
        d,
        "This excerpt exposes the raw estimator and the explicit robust reweighting used inside the project helper. It is executed on the same training design and checked against the helper result. The response stays in dollars; X contains the six standardized training predictors.",
    )
    code(d, "lts_algorithm")
    add_table(
        d,
        ["Executed output", "Value"],
        [
            ["Raw subset h", "30,000"],
            ["MAD scale", "0.510606"],
            ["Weight 1 observations", "33,548"],
        ],
    )
    p(
        d,
        "The fit first searches for a resistant subset. Only after that search does the robust residual rule decide which observations enter the final refit. A raw subset member and a positive reweighting observation are different concepts: h = 30,000 governs the trimmed objective, whereas 33,548 observations enter the final least-squares refit. Records outside that refit remain in the dataset and in full-sample error calculations.",
    )
    p(
        d,
        "For a standardized predictor x* = (x - training mean) / training scale, the original-unit slope is beta* / training scale. The original intercept subtracts the sum of beta* times training mean divided by training scale. This conversion makes the coefficient comparison interpretable without using test-set preprocessing information.",
    )
    p(
        d,
        "A separate synthetic check in notebook 02 uses 1,000 observations with a known linear relationship and 100 responses shifted by +$50. The slope-vector error is about 0.773 for OLS, 0.020 for raw LTS, and 0.022 for reweighted LTS. That software-validation exercise has different parameters from the taxi stress test and is not evidence about genuine taxi fraud.",
    )

    page(d, "Executed fitting and training diagnostics")
    code(d, "fit")
    add_table(
        d,
        ["Executed output", "Value"],
        [
            ["Raw LTS h", "30,000"],
            ["MAD residual scale USD", "0.510606"],
            ["Reweighting observations", "33,548"],
        ],
    )
    code(d, "diagnostics")
    table(
        d,
        "training_ols_coefficients.csv",
        ["term", "coefficient", "standard_error_conventional", "standard_error_hc3"],
        ["Term", "OLS coefficient", "Usual SE", "HC3 SE"],
    )
    p(
        d,
        "HC3 is a member of the White heteroskedasticity-consistent covariance family taught in Module I. It adjusts residual contributions for leverage; the OLS coefficients themselves are unchanged. The distance SE rises from about 0.006 to 0.096 and duration SE from 0.001 to 0.039. Constant-variance standard errors would substantially overstate precision. HC3 still relies on suitable independence and mean-model conditions; it does not repair omitted variables or dependence.",
    )

    page(d, "Evidence on assumptions and influential records")
    add_figure(
        d,
        FIGURES / "training_residual_diagnostics.png",
        "Figure 1   Training OLS residual spread and normal quantiles",
        width=6.8,
    )
    p(
        d,
        "Residual spread rises with fitted fare and the normal quantile plot has pronounced tail departures. Residual skewness is -3.21 and ordinary kurtosis is 187.40 (normal kurtosis is 3). Residual RMSE increases from $1.91 in the lowest fitted-fare decile to $6.22 in the highest. These magnitudes support heteroskedasticity and heavy tails beyond merely obtaining small p-values in a large sample.",
    )
    add_table(
        d,
        ["Supporting diagnostic", "Training result", "Interpretation"],
        [
            ["Breusch-Pagan LM", "33,831.46; p < 10^-16", "Reject constant variance"],
            ["Ramsey RESET F", "3,692.56; p < 10^-16", "Mean form is incomplete"],
            [
                "VIF for predictors",
                "1.01 to 1.21",
                "Low conventional linear collinearity",
            ],
            [
                "Cook distance above 4/n",
                "1,663 observations",
                "Influence screening, not deletion",
            ],
        ],
        font_size=9,
    )
    p(
        d,
        "Breusch-Pagan and RESET are supporting implementations of assumption checking in Module II; their null hypotheses concern constant variance and omitted nonlinear mean structure respectively. Under departures from other assumptions, neither test identifies a unique cause. Binned residual means also show curvature, so the common linear specification should be interpreted as an approximation. VIF can look small despite strong rank association when extreme durations distort linear covariances.",
    )
    p(
        d,
        "The four highest Cook distances occur at durations about 1,412-1,437 minutes with fares $7.20-$24.00. Cook distances range about 47.77-56.62. These rows can heavily affect the duration slope. Their truth cannot be established from the table, so they are retained. Durbin-Watson near 2 in randomly ordered data is not evidence of temporal independence. Additional normality tests remain in notebook 03 as supplementary checks.",
    )

    page(d, "Clean performance and answers to Questions 1 and 2")
    code(d, "clean")
    table(
        d,
        "clean_model_performance.csv",
        [
            "evaluation_split",
            "estimator",
            "mae",
            "median_absolute_error",
            "rmse",
            "r_squared",
        ],
        ["Split", "Estimator", "MAE USD", "Median AE", "RMSE USD", "R squared"],
        lambda x: x[
            x.estimator.eq("OLS")
            | (x.estimator.eq("Reweighted LTS") & x.alpha.eq(0.75))
        ],
    )
    p(
        d,
        "For evaluation residual e = observed fare minus predicted fare, MAE is mean(|e|), median AE is median(|e|), and RMSE is sqrt(mean(e squared)). R-squared is 1 minus the sum of squared prediction errors divided by the sum of squared deviations from the evaluation response mean. All measures here use the complete indicated split.",
    )
    p(
        d,
        "Question 1: holding the other predictors fixed, OLS estimates $3.622 per mile and $0.082 per minute. Reweighted LTS estimates $1.813 per mile and $0.605 per minute. These differences show that the estimated allocation between distance and duration depends strongly on which observations influence the fit. They cannot be read as official tariff rates: the meter uses operational rules, and the additive model omits important trip detail.",
    )
    p(
        d,
        "Question 2: LTS reduces holdout MAE from $2.086 to $1.015 and median absolute error from $1.594 to $0.347. It describes many ordinary trips more closely. OLS has much lower RMSE ($3.253 versus $8.756) and higher R-squared (0.923 versus 0.445). Squaring errors gives disproportionate importance to large misses; LTS resists unusual training configurations but can make very large prediction errors on similar extreme test trips. OLS balances those errors at the expense of typical-trip fit. Neither estimator dominates every loss function.",
    )
    p(
        d,
        "Training fit is reported for transparency and is optimistic relative to evaluation of a newly observed sample. Here train and holdout absolute-error measures tell a similar story. No statistical significance claim is made about the differences; these are descriptive results for one fixed split.",
    )

    page(d, "Raw fits sensitivity and audit candidates")
    table(
        d,
        "clean_model_performance.csv",
        ["estimator", "alpha", "mae", "median_absolute_error", "rmse", "r_squared"],
        ["Test estimator", "Alpha", "MAE", "Median AE", "RMSE", "R squared"],
        lambda x: x[x.evaluation_split.eq("test") & x.estimator.ne("OLS")],
    )
    p(
        d,
        "At alpha 0.90, the reweighted test MAE is $0.848 and RMSE $8.280; at 0.50 they are $1.043 and $8.825. Greater retained coverage changes the fitted compromise. The primary alpha remains 0.75 because it was specified before test comparison. Choosing the best reported alpha on the test set would turn that set into tuning data. Raw and reweighted primary test errors are close, so reweighting is a refinement rather than the source of the main OLS-LTS difference.",
    )
    add_heading(d, "Question 3   Scope of the audit queue", 2)
    p(
        d,
        "The operational screen flags a training row when its primary raw-LTS weight is zero OR its absolute standardized robust residual exceeds 2.5. Since the reweighting cutoff is about 2.241, every >2.5 flag already has zero weight. The union therefore equals the zero-weight set; these are nested thresholds, not independent pieces of evidence. This detail matters when interpreting the queue size.",
    )
    add_table(
        d,
        ["Clean primary screen", "Training rows"],
        [
            ["Zero reweighting weight", "6,452"],
            ["Absolute robust residual above 2.5", "5,929"],
            ["Union called audit candidates", "6,452 of 40,000 or 16.13%"],
        ],
    )
    p(
        d,
        "The ranked CSV retains source IDs, timestamps, original fare and trip attributes, raw residuals, and screen flags. Records with extreme duration and modest fare merit timestamp and meter review; unusually high positive residuals merit examination of fare entry and operational context. With 16.13% flagged, the screen creates a substantial workload. The team should use a ranked review sample and verified case outcomes before selecting a deployment threshold. The present study cannot estimate fraud precision for genuine trips.",
    )

    page(d, "Controlled contamination and prediction stability")
    code(d, "stress")
    table(
        d,
        "contamination_model_performance.csv",
        [
            "training_scenario",
            "estimator",
            "mae",
            "median_absolute_error",
            "rmse",
            "r_squared",
        ],
        [
            "Training fares",
            "Estimator",
            "Test MAE",
            "Median AE",
            "Test RMSE",
            "R squared",
        ],
    )
    p(
        d,
        "Exactly 2,000 training fares receive +$100 after selection without replacement with seed 42. The selection changes no predictors and the same training-fitted scaler is used for both response vectors. Test labels and fares are unchanged. This is vertical response contamination; it does not simulate corruption of trip distances or timestamps.",
    )
    p(
        d,
        "Question 4: OLS MAE increases by $3.297 and RMSE by $2.686; R-squared falls by 0.179. Reweighted LTS MAE changes by -$0.015 and RMSE by -$0.019. LTS is stable under this experiment. Its small improvement is a consequence of the particular selected sample and changed fit, not evidence that corruption generally improves prediction. The stressed OLS RMSE still remains below LTS RMSE because LTS's pre-existing errors on genuine extremes persist.",
    )

    page(d, "Coefficient stability and detection results")
    table(
        d,
        "contamination_coefficient_change.csv",
        ["estimator", "term", "clean", "contaminated", "absolute_change"],
        ["Estimator", "Term", "Clean", "Stressed", "Signed change"],
        lambda x: x[x.term.isin(["const", "trip_distance", "duration_min"])],
    )
    p(
        d,
        "The OLS intercept rises by $4.827, close to the mean response shift 0.05 x $100 = $5. The reweighted LTS intercept changes by only -$0.025, the distance slope by +$0.020 per mile, and duration slope by -$0.000542 per minute. Percent changes for near-zero coefficients are unstable summaries; original-unit changes are more informative. Signed change is the stressed coefficient minus the clean coefficient.",
    )
    code(d, "detection")
    add_table(
        d,
        ["Known experiment classification", "Count or rate"],
        [
            ["True positives and false negatives", "2,000 and 0"],
            ["False positives and true negatives", "5,684 and 32,316"],
            ["Recall   TP / (TP + FN)", "100%"],
            ["Precision   TP / (TP + FP)", "26.03%"],
            ["False positive rate   FP / (FP + TN)", "14.96%"],
        ],
    )
    p(
        d,
        "Here a false positive means an unaltered row flagged relative to the artificial-corruption label. It may still be an unusual genuine trip deserving review. All artificial corruptions are detected, but 5,684 unaltered rows are also flagged. The result demonstrates excellent recall for a large upward shift and limited specificity. It does not validate a fraud classifier.",
    )

    page(d, "Actual observations and fitted relationships")
    add_figure(
        d,
        FIGURES / "actual_points_clean_stress_fitted_lines.png",
        "Figure 2   Comparable genuine trips and artificial response shifts",
        width=7.0,
    )
    p(
        d,
        "The points are actual training records selected for display: one passenger, weekdays, pickup hours 11-13, and distance 1.18-1.88 miles. To reduce overlap, up to 250 unaltered records are sampled with seed 42, together with all selected corruptions in this comparable group. The same recorded fares before corruption appear in panel A. Red points in panel B receive exactly +$100. The horizontal range is limited near the 99.5th percentile of duration in the comparable group; these display restrictions do not alter any fit or performance metric.",
    )
    p(
        d,
        "Lines are predictions from the original six-predictor models evaluated at distance 1.53 miles, one passenger, weekday, noon, while duration varies. The points occupy a narrow distance and time window; they are close to, but not identical to, the exact line reference. Thus this is a conditional illustration rather than a new regression fitted only to the displayed points. It cannot depict the complete multivariable data distribution.",
    )
    p(
        d,
        "Panel A shows that the LTS duration profile follows the dense ordinary-trip pattern, while clean OLS has a much smaller duration slope. Panel B reveals the visibly separated artificial fare band. Panel C enlarges the ordinary fare range: stressed OLS shifts upward; clean and stressed LTS nearly coincide. This illustrates the fitted response to corruption without relying on PCA, which would hide the original variable units. Panel B uses a taller dollar range than A and C, as its axis labels indicate.",
    )
    p(
        d,
        "The picture supports the numerical stability finding for this specified subgroup. It should not be treated as proof that every flagged genuine trip is wrong or that LTS outperforms OLS everywhere. The complete holdout RMSE comparison shows why that stronger conclusion would be incorrect.",
    )

    page(d, "Business conclusions and study limitations")
    p(
        d,
        "The first business question is answered by the original-unit coefficients: the resistant fit assigns materially more weight to duration and less to distance than OLS. The second is answered by the different error rankings and extreme-duration influence: OLS is sensitive to the tails, but it better accommodates them under squared loss. LTS gives substantially better typical absolute errors. A fare-monitoring team should state which loss and which population it values rather than selecting an estimator using one attractive number.",
    )
    p(
        d,
        "For the third question, the 6,452 clean LTS flags form a reproducible review queue. A practical investigation would verify timestamps, meter data, route context, and fare regime, starting with a small ranked sample. A discrepancy is a lead for human assessment. The data provide no ground truth for genuine fare irregularity, so no misconduct rate or monetary fraud loss can be inferred.",
    )
    p(
        d,
        "For the fourth question, the controlled upward shift gives clear evidence of coefficient and prediction stability under 5% response contamination. It is one seeded experiment, with no repeated-run uncertainty interval. The +$100 magnitude is deliberately conspicuous. Performance against subtle shifts, downward changes, predictor contamination, or an adversarial pattern could differ.",
    )
    add_heading(d, "Limits of statistical interpretation", 2)
    p(
        d,
        "One month and one tariff regime constrain generalization. Filters exclude zero or negative values and missing required fields, so the conclusions apply to eligible positive-fare occupied trips. Trips may share drivers, traffic, routes, or other conditions; random row splitting does not establish independent errors or future-month accuracy. There is no cluster-robust or temporal validation. The earlier computing benchmark also limits claims of completely untouched external validation.",
    )
    p(
        d,
        "The additive model simplifies the meter's dependence on travel and waiting conditions. Remaining curvature, changing residual variance, and extreme predictor records explain why robust fitting alone is not a complete prediction solution. The MAD scale and fixed cutoff create a wide review queue in heterogeneous data. Small LTS coefficients are not supported by conventional significance claims. More flexible models or calibrated thresholds would require a new training/validation design and fresh evaluation data.",
    )
    p(
        d,
        "The recommended role of LTS is a stable descriptive benchmark and audit prioritization tool. OLS remains a useful reference and achieves lower overall squared prediction error in this dataset. Reporting the trade-off, the broad review workload, and the experimental boundaries is necessary for an evidence-based business recommendation.",
    )

    page(d, "References and reproducibility")
    refs = [
        "New York City Taxi and Limousine Commission. (n.d.-a). TLC trip record data [Data set]. Retrieved November 1, 2026, from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
        "New York City Taxi and Limousine Commission. (2025, March 18). Data dictionary - Yellow taxi trip records. https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf",
        "Rousseeuw, P. J., & Van Driessen, K. (2006). Computing LTS regression for large data sets. Data Mining and Knowledge Discovery, 12, 29-45. https://doi.org/10.1007/s10618-005-0024-4",
        "RobPy. (n.d.). robpy.regression [Software documentation]. https://robpy.readthedocs.io/en/stable/api/robpy.regression.html. Analysis uses installed release 0.0.6; local wrapper defines the explicit MAD reweighting.",
    ]
    for ref in refs:
        p(d, ref)
    add_heading(d, "Execution evidence and artifact traceability", 2)
    p(
        d,
        "All code images in this report are generated from exact cells executed in notebooks/06_final_report_execution_evidence.ipynb. Cells call the shared project functions; they are excerpts, not the complete implementation. Their numerical outputs follow the relevant images as tables. Notebooks 01-05 give the fuller teaching narrative, and src/taxi_lts contains the reusable preparation, fitting, and stress-test definitions. No manually invented output is used.",
    )
    p(
        d,
        "Main commands: uv sync; uv run python -m src.taxi_lts.prepare_data; uv run python -m src.taxi_lts.split_data; uv run python -m src.taxi_lts.explore_training; uv run python -m src.taxi_lts.run_analysis; uv run python -m src.taxi_lts.stress_test. Rebuild this report with uv run python report/build_final_report.py. Execute tests with uv run pytest. FAST-LTS fitting limits numerical threads to two.",
    )
    p(
        d,
        "Raw file SHA-256: 8b3933fe6f0d7b6d8826613c0dd724edc680ff7c49e2bd4c7635c05102728637. Final modelling CSV SHA-256: 9325f572dc4d2036b2df67f2f2e63350a3b88bc52b8b7cde84b7d1f727a7fe39. Metadata in data/source_metadata.json describes the initial 10,000-row artifact; data/modeling_metadata.json describes the later 50,000-row modelling sample.",
    )
    p(
        d,
        "Syllabus alignment: the substantive analysis uses Modules I, II, and V: multiple regression, heteroskedasticity-consistent uncertainty, influence diagnostics, LTS, and applied regression. FAST-LTS, MAD, train/test evaluation, and named diagnostic tests are explained supporting concepts. report/SYLLABUS_ALIGNMENT.md provides preparation questions and definitions for the presentation.",
    )

    output = ROOT / "report/ED409_LTS_Final_Report.docx"
    d.save(output)
    return output


if __name__ == "__main__":
    print(build(execute="--reuse-evidence" not in sys.argv))
