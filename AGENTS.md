# Repository Guidelines

## Project Structure

Keep source data in `data/raw/` and reproducible derived data in `data/processed/`. Put exploratory and executed notebooks in `notebooks/`; reusable Python modules belong in `src/taxi_lts/`, with matching tests in `tests/`. Generated charts and tables go in `outputs/`, the written submission in `report/`, and slides in `presentation/`. Do not commit the raw TLC Parquet file, generated environments, notebook checkpoints, or temporary render files.

## Environment and Commands

Use `uv` to manage the environment and lock dependencies. Run `uv sync` after dependency changes. The principal workflows must remain callable as:

```bash
uv run python -m src.taxi_lts.prepare_data
uv run python -m src.taxi_lts.run_analysis
uv run pytest
```

Prefer small, typed functions and PEP 8 naming. Format with `uv run ruff format .` and lint with `uv run ruff check .`. Use `pathlib` for paths and store shared constants centrally. Every stochastic operation must use the documented seed `42` unless a test explicitly checks another seed.

## Data, Tests, and Evidence

Preserve raw downloads unchanged and record the official source URL, retrieval date, and SHA-256 checksum. Never fabricate data, metrics, screenshots, or conclusions. Tests must cover filtering, deterministic sampling, train/test separation, contamination scope, and recovery on synthetic linear data. Fit preprocessing only on training observations. Machine-readable result tables are the source of truth for the report and presentation.

Treat LTS-flagged trips as **audit candidates**, never as proven fraud. Do not discard legitimate-looking extremes merely because they are large. Clearly distinguish genuine records from the controlled `$100` response-contamination experiment.

The business case and its questions must occupy no more than two pages. In the report, place images of genuinely executed code before the corresponding output. Before delivery, render and inspect every PDF page and presentation slide for clipping, pagination, legibility, and consistency with computed results.
