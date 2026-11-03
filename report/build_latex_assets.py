"""Prepare figures and LaTeX tables from existing verified analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.taxi_lts.explore_training import PREDICTORS
from src.taxi_lts.plot_actual_points import save_comparable_trip_plots
from src.taxi_lts.stress_test import contaminate_training_response

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs/figures"
TABLES = ROOT / "outputs/tables"
GENERATED = ROOT / "report/latex/generated"


def draw_actual_points() -> None:
    """Regenerate the shared plots from the fixed training split and CSV fits."""
    data = pd.read_csv(ROOT / "data/processed/taxi_modeling_50000.csv")
    train = data.loc[data.split.eq("train")].copy()
    _, known = contaminate_training_response(train)
    coefficients = pd.read_csv(TABLES / "contamination_coefficient_change.csv")
    metadata = save_comparable_trip_plots(train, coefficients, known, FIGURES)
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def escape(value: object) -> str:
    """Escape text inserted in generated LaTeX tables."""
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_table(
    name: str, headers: list[str], rows: list[list[object]], spec: str
) -> None:
    """Write a compact booktabs table fragment."""
    lines = [
        r"\begin{tabular}{" + spec + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(escape(v) for v in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED / f"{name}.tex").write_text("\n".join(lines) + "\n")


def create_tables() -> None:
    """Generate numeric tables directly from the source CSVs."""
    GENERATED.mkdir(parents=True, exist_ok=True)
    flow = pd.read_csv(TABLES / "data_quality_flow.csv").iloc[:-1]
    labels = [
        "Original records",
        "January pickup",
        "Standard rate",
        "Positive distance",
        "Positive fare",
        "Duration at least one minute",
        "At least one passenger",
        "Eligible records",
    ]
    write_table(
        "filter_audit",
        ["Step", "Removed", "Remaining"],
        [
            [label, f"{r.records_removed:,}", f"{r.records_after:,}"]
            for label, r in zip(labels, flow.itertuples(), strict=True)
        ],
        "lrr",
    )
    stats = pd.read_csv(TABLES / "training_descriptive_statistics.csv").iloc[:4]
    write_table(
        "descriptive",
        ["Variable", "Median", "99th percentile", "Maximum"],
        [
            [r.label, f"{r.median:.2f}", f"{r.p99:.2f}", f"{r.maximum:,.2f}"]
            for r in stats.itertuples()
        ],
        "lrrr",
    )
    perf = pd.read_csv(TABLES / "clean_model_performance.csv")
    primary = perf[
        perf.estimator.eq("OLS")
        | (perf.estimator.eq("Reweighted LTS") & perf.alpha.eq(0.75))
    ]
    headers = ["Sample", "Estimator", "MAE", "Median AE", "RMSE", "$R^2$"]
    rows = [
        [
            r.evaluation_split.title(),
            r.estimator,
            f"{r.mae:.3f}",
            f"{r.median_absolute_error:.3f}",
            f"{r.rmse:.3f}",
            f"{r.r_squared:.3f}",
        ]
        for r in primary.itertuples()
    ]
    write_table("clean_performance", headers, rows, "llrrrr")
    sensitivity = perf[perf.evaluation_split.eq("test") & perf.estimator.ne("OLS")]
    write_table(
        "sensitivity",
        ["Estimator", "$\\alpha$", "MAE", "Median AE", "RMSE", "$R^2$"],
        [
            [
                r.estimator,
                f"{r.alpha:.2f}",
                f"{r.mae:.3f}",
                f"{r.median_absolute_error:.3f}",
                f"{r.rmse:.3f}",
                f"{r.r_squared:.3f}",
            ]
            for r in sensitivity.itertuples()
        ],
        "lrrrrr",
    )
    stress = pd.read_csv(TABLES / "contamination_model_performance.csv")
    write_table(
        "stress_performance",
        headers,
        [
            [
                r.training_scenario.title(),
                r.estimator,
                f"{r.mae:.3f}",
                f"{r.median_absolute_error:.3f}",
                f"{r.rmse:.3f}",
                f"{r.r_squared:.3f}",
            ]
            for r in stress.itertuples()
        ],
        "llrrrr",
    )
    coeff = pd.read_csv(TABLES / "contamination_coefficient_change.csv")
    pretty = {
        "const": "Intercept",
        "trip_distance": "Distance",
        "duration_min": "Duration",
        "passenger_count": "Passengers",
        "pickup_hour_sin": "Hour sine",
        "pickup_hour_cos": "Hour cosine",
        "is_weekend": "Weekend",
    }
    clean = coeff.pivot(index="term", columns="estimator", values="clean")
    hc3 = pd.read_csv(TABLES / "training_ols_coefficients.csv")
    write_table(
        "hc3",
        ["Term", "OLS coefficient", "Usual SE", "HC3 SE"],
        [
            [
                pretty[r.term],
                f"{r.coefficient:.4f}",
                f"{r.standard_error_conventional:.4f}",
                f"{r.standard_error_hc3:.4f}",
            ]
            for r in hc3.itertuples()
        ],
        "lrrr",
    )
    write_table(
        "coefficients",
        ["Term", "OLS", "Reweighted LTS"],
        [
            [
                pretty[k],
                f"{clean.loc[k, 'OLS']:.4f}",
                f"{clean.loc[k, 'Reweighted LTS']:.4f}",
            ]
            for k in ["const", *PREDICTORS]
        ],
        "lrr",
    )
    write_table(
        "coefficient_change",
        ["Estimator", "Term", "Before", "After", "Change"],
        [
            [
                r.estimator,
                pretty[r.term],
                f"{r.clean:.4f}",
                f"{r.contaminated:.4f}",
                f"{r.absolute_change:+.4f}",
            ]
            for r in coeff.itertuples()
            if r.term in ["const", "trip_distance", "duration_min"]
        ],
        "llrrr",
    )


if __name__ == "__main__":
    draw_actual_points()
    create_tables()
    print("Prepared LaTeX tables and revised actual-point figures.")
