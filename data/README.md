# Data provenance and preparation

The source is the official NYC Taxi and Limousine Commission January 2026
Yellow Taxi Trip Records file. The raw Parquet file is intentionally excluded
from version control; its URL, size, retrieval date, record count, and SHA-256
checksum are recorded in `source_metadata.json` after preparation.

Recreate the analysis-ready sample from the repository root with:

```bash
uv run python -m src.taxi_lts.prepare_data
```

The workflow keeps January 2026 pickups with standard rate code `1`, positive
distance and fare, duration of at least 1 minute, and passenger count of at
least 1. It does not impose an upper cutoff on fare, distance, duration, or
passenger count. From the eligible population it samples 10,000 records without
replacement using seed `42`, then sorts selected records by their original
source-row identifier.

Generated audit tables are stored in `outputs/tables/`. The processed CSV is a
reproducible modelling input, but the TLC cautions that provider-submitted trip
records are not guaranteed to be fully accurate.

After FAST-LTS scalability validation, create the fixed 50,000-trip modelling
sample and its 40,000/10,000 train-test assignment with:

```bash
uv run python -m src.taxi_lts.split_data
```

The selection is the exact seed-42 nested sample used by the 50,000-row
benchmark. Split membership is assigned once and stored before any scaling,
exploration, contamination, or model evaluation.
