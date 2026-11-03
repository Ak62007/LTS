"""Execute the report excerpts and regenerate their notebook and code images."""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "report/assets/final"


def evidence() -> dict[str, str]:
    """Execute the code shown in the report and save an evidence notebook."""
    ASSETS.mkdir(parents=True, exist_ok=True)
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
        "diagnostics": (
            "from src.taxi_lts.explore_training import "
            "fit_diagnostic_ols, coefficient_table\n"
            "diagnostic_ols, hc3 = fit_diagnostic_ols(train)\n"
            "display(coefficient_table(diagnostic_ols, hc3).round(4))"
        ),
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
    nb.cells = [nbformat.v4.new_code_cell(source) for source in sources.values()]
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


if __name__ == "__main__":
    evidence()
