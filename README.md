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

## Reproducibility policy

All random sampling and splitting uses seed `42`. Raw data provenance and checksums will be recorded before analysis. Reported claims must be generated from the checked-in workflow; flagged records are audit candidates, not allegations of fraud.
