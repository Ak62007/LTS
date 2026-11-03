# Robust Detection of Anomalous Taxi Fares Using Least Trimmed Squares Regression

Course assignment repository for comparing ordinary least squares (OLS) with least trimmed squares (LTS) on January 2026 NYC Yellow Taxi trips.

**Student:** Aditya Kumar  
**Enrollment:** IED/10007/23

## Planned workflow

1. Acquire and verify the official NYC TLC Yellow Taxi Parquet file.
2. Apply documented quality filters and create a deterministic 10,000-row sample.
3. Compare OLS and reweighted FAST-LTS on clean and deliberately contaminated training data.
4. Export auditable tables, figures, an executed notebook, the written assignment, and a nine-slide presentation.

## Repository layout

- `data/raw/` — untouched source data (ignored by Git)
- `data/processed/` — reproducible analysis-ready data
- `notebooks/` — executed analysis notebook
- `src/taxi_lts/` — reusable preparation and modelling code
- `tests/` — deterministic and synthetic validation tests
- `outputs/` — generated figures and machine-readable tables
- `report/` — editable report and final PDF
- `presentation/` — slide deck and render checks

## Setup

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run pytest
```

Prepare the verified January 2026 data sample with:

```bash
uv run python -m src.taxi_lts.prepare_data
```

The command verifies the raw-file checksum, applies the documented eligibility
rules, produces a deterministic 10,000-record sample, and exports data-quality
tables. See `data/README.md` for the detailed data policy.

Validate FAST-LTS on a known contaminated linear relationship with:

```bash
uv run python -m src.taxi_lts.validate_lts
```

Run the guarded scalability benchmark one sample size at a time with:

```bash
uv run python -m src.taxi_lts.benchmark_lts --sample-size 10000
uv run python -m src.taxi_lts.benchmark_lts --sample-size 25000
uv run python -m src.taxi_lts.benchmark_lts --sample-size 50000
```

The benchmark uses deterministic nested samples, 500 FAST-LTS starting subsets,
and a two-thread numerical-library limit. Results are written to
`outputs/tables/lts_scalability_benchmark.csv`.

Create the fixed 50,000-record modelling sample and its 80/20 split with:

```bash
uv run python -m src.taxi_lts.split_data
```

Run training-only exploratory and model-specification diagnostics with:

```bash
uv run python -m src.taxi_lts.explore_training
```

## Executed teaching notebooks

Open the notebooks in this order to follow the completed work from source data
to the frozen model specification:

1. `notebooks/01_data_preparation_sampling_and_split.ipynb`
2. `notebooks/02_fast_lts_validation_and_scalability.ipynb`
3. `notebooks/03_training_eda_and_model_specification.ipynb`
4. `notebooks/04_clean_ols_and_lts_comparison.ipynb`
5. `notebooks/05_contamination_stress_test.ipynb`
6. `notebooks/06_final_report_execution_evidence.ipynb` (exact report excerpts)

The final report is now written in **LaTeX**. The submission PDF is
`report/ED409_LTS_Final_Report.pdf`, with editable source in
`report/latex/ED409_LTS_Final_Report.tex` and a complete portable package at
`report/ED409_LTS_LaTeX_Source.zip`. Its first two pages contain the case,
dataset introduction, and business questions; the solution starts on page 3.
Rebuild the final PDF and source ZIP with Tectonic installed:

```bash
uv run python -m report.build_latex_report
```

See `report/latex/README.md` for standalone compilation and editing guidance,
and `report/latex/REVIEW_LOG.md` for the iterative review. The earlier DOCX and
its builder are retained as a previous edition. They do not update the LaTeX
report. `report/build_final_report.py` also regenerates notebook 06 and the
executed-code images; use it only when that evidence needs refreshing.

Every code cell is executed and its output is stored in the notebook. Rebuild
notebooks 01-05 from the repository root with:

```bash
uv run python notebooks/build_executed_notebooks.py
```

Run the controlled 5% training-response contamination experiment with:

```bash
uv run python -m src.taxi_lts.stress_test
```

Before preparing the report or presentation, review
`report/SYLLABUS_ALIGNMENT.md`. It maps every current method to ED409 and
identifies supporting concepts that require a short explanation.

## Reproducibility policy

All random sampling and splitting uses seed `42`. Raw data provenance and checksums will be recorded before analysis. Reported claims must be generated from the checked-in workflow; flagged records are audit candidates, not allegations of fraud.
