# Robust Detection of Anomalous Taxi Fares Using Least Trimmed Squares Regression

**Aditya Kumar — IED/10007/23 · ED409 Regression Techniques**

OLS and reweighted FAST-LTS analysis of January 2026 NYC Yellow Taxi records.
The fixed modelling sample contains 50,000 trips: 40,000 training observations
and 10,000 evaluation observations. Sampling and model initialization use seed 42.
The stress test adds $100 to 5% of training fares and leaves evaluation data unchanged.

## Files

- `data/`: source metadata, ignored raw Parquet, and processed samples.
- `src/taxi_lts/`: preparation, fitting, diagnostics, and stress-test modules.
- `notebooks/`: six executed notebooks containing code and saved outputs.
- `outputs/`: verified figures, CSV tables, and analysis metadata.
- `report/ED409_LTS_Report.pdf`: final report.
- `report/latex/ED409_LTS_Report.tex`: editable report source.
- `report/ED409_LTS_LaTeX_Source.zip`: standalone LaTeX source package.
- `tests/`: data, split, estimator, contamination, and notebook checks.

## Environment and analysis

Install [uv](https://docs.astral.sh/uv/) and run from the repository root:

```bash
uv sync
uv run python -m src.taxi_lts.prepare_data
uv run python -m src.taxi_lts.split_data
uv run python -m src.taxi_lts.explore_training
uv run python -m src.taxi_lts.run_analysis
uv run python -m src.taxi_lts.stress_test
uv run pytest
```

Data preparation also reproduces the initial 10,000-row exploratory sample.
The final model comparison uses `data/processed/taxi_modeling_50000.csv`.
See [data/README.md](data/README.md) for provenance and eligibility rules.

Synthetic estimator validation and the optional scalability benchmark:

```bash
uv run python -m src.taxi_lts.validate_lts
uv run python -m src.taxi_lts.benchmark_lts --sample-size 50000
```

Model-fitting workflows limit numerical libraries to two threads.

## Notebooks and report

The notebooks are numbered in analysis order: data preparation, estimator
validation, training diagnostics, clean-model comparison, contamination, and
report execution evidence. Open them from the repository root with:

```bash
uv run jupyter lab
```

Regenerate notebooks 01–05 and notebook 06 with:

```bash
uv run python -m notebooks.build_executed_notebooks
uv run python -m report.build_execution_evidence
```

These commands execute code; notebook 06 also regenerates the report's code
images. All six notebooks contain code cells only, with execution outputs retained.

With Tectonic installed, rebuild the PDF and standalone source ZIP:

```bash
uv run python -m report.build_latex_report
```

See [report/latex/README.md](report/latex/README.md) for editing and standalone
compilation. [report/SYLLABUS_ALIGNMENT.md](report/SYLLABUS_ALIGNMENT.md) records
the course concepts and presentation preparation notes.

Raw source data remain unchanged. Fit preprocessing on training rows only,
use the CSV results as the source of reported values, and interpret flags as
audit candidates rather than proven fraud.
