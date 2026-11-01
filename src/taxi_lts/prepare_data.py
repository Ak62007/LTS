"""Prepare a deterministic January 2026 Yellow Taxi analysis sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.taxi_lts import RANDOM_SEED

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILENAME = "yellow_tripdata_2026-01.parquet"
SOURCE_PAGE = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
SOURCE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet"
)
EXPECTED_RAW_SHA256 = "8b3933fe6f0d7b6d8826613c0dd724edc680ff7c49e2bd4c7635c05102728637"
REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "fare_amount",
]
OUTPUT_COLUMNS = [
    "source_row_id",
    "pickup_datetime",
    "dropoff_datetime",
    "fare_amount",
    "trip_distance",
    "duration_min",
    "passenger_count",
    "pickup_hour",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "is_weekend",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load only columns required for preparation and attach stable row IDs."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found at {path}. Download it from {SOURCE_URL}."
        )
    data = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    data.insert(0, "source_row_id", np.arange(len(data), dtype=np.int64))
    return data


def add_duration(data: pd.DataFrame) -> pd.DataFrame:
    """Create trip duration in minutes from TLC timestamps."""
    result = data.copy()
    result["duration_min"] = (
        result["tpep_dropoff_datetime"] - result["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    return result


def filter_trip_records(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply eligibility rules sequentially and return an exclusion audit."""
    working = add_duration(data)
    audit_rows: list[dict[str, int | str]] = []

    rules: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        (
            "pickup_in_january_2026",
            lambda frame: (
                frame["tpep_pickup_datetime"].ge("2026-01-01")
                & frame["tpep_pickup_datetime"].lt("2026-02-01")
            ),
        ),
        ("standard_rate_code_1", lambda frame: frame["RatecodeID"].eq(1)),
        ("positive_trip_distance", lambda frame: frame["trip_distance"].gt(0)),
        ("positive_fare_amount", lambda frame: frame["fare_amount"].gt(0)),
        (
            "duration_1_to_120_minutes",
            lambda frame: frame["duration_min"].between(1, 120, inclusive="both"),
        ),
        (
            "passenger_count_1_to_6",
            lambda frame: frame["passenger_count"].between(1, 6, inclusive="both"),
        ),
    ]

    audit_rows.append(
        {
            "stage": "original_records",
            "records_before": len(working),
            "records_removed": 0,
            "records_after": len(working),
        }
    )
    for stage, rule in rules:
        before = len(working)
        mask = rule(working).fillna(False)
        working = working.loc[mask].copy()
        audit_rows.append(
            {
                "stage": stage,
                "records_before": before,
                "records_removed": before - len(working),
                "records_after": len(working),
            }
        )

    audit_rows.append(
        {
            "stage": "eligible_records",
            "records_before": len(working),
            "records_removed": 0,
            "records_after": len(working),
        }
    )
    return working, pd.DataFrame(audit_rows)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable and cyclical time features for modelling."""
    result = data.copy()
    result["pickup_hour"] = result["tpep_pickup_datetime"].dt.hour.astype("int8")
    angle = 2 * np.pi * result["pickup_hour"] / 24
    result["pickup_hour_sin"] = np.sin(angle)
    result["pickup_hour_cos"] = np.cos(angle)
    result["is_weekend"] = (
        result["tpep_pickup_datetime"].dt.day_of_week.ge(5).astype("int8")
    )
    result = result.rename(
        columns={
            "tpep_pickup_datetime": "pickup_datetime",
            "tpep_dropoff_datetime": "dropoff_datetime",
        }
    )
    return result[OUTPUT_COLUMNS]


def deterministic_sample(
    data: pd.DataFrame, sample_size: int = 10_000, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Select a reproducible sample and store it in stable source-row order."""
    if len(data) < sample_size:
        raise ValueError(
            f"Cannot sample {sample_size:,} rows from {len(data):,} eligible rows."
        )
    return (
        data.sample(n=sample_size, replace=False, random_state=seed)
        .sort_values("source_row_id")
        .reset_index(drop=True)
    )


def missingness_table(raw_data: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values before filtering and in the final sample."""
    rows: list[dict[str, int | float | str]] = []
    for dataset_name, frame in [("raw_relevant", raw_data), ("final_sample", sample)]:
        for column in frame.columns:
            missing = int(frame[column].isna().sum())
            rows.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "missing_count": missing,
                    "missing_percent": 100 * missing / len(frame),
                }
            )
    return pd.DataFrame(rows)


def range_table(
    raw_with_duration: pd.DataFrame,
    eligible: pd.DataFrame,
    sample: pd.DataFrame,
) -> pd.DataFrame:
    """Report observed numerical ranges at the principal preparation stages."""
    rows: list[dict[str, float | str]] = []
    datasets = [
        ("raw_relevant", raw_with_duration),
        ("eligible_population", eligible),
        ("final_sample", sample),
    ]
    column_pairs = [
        ("fare_amount", "fare_amount"),
        ("trip_distance", "trip_distance"),
        ("duration_min", "duration_min"),
        ("passenger_count", "passenger_count"),
    ]
    for dataset_name, frame in datasets:
        for label, column in column_pairs:
            values = frame[column].dropna()
            rows.append(
                {
                    "dataset": dataset_name,
                    "variable": label,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def descriptive_statistics(sample: pd.DataFrame) -> pd.DataFrame:
    """Return report-ready distribution summaries for the final sample."""
    numeric_columns = [
        "fare_amount",
        "trip_distance",
        "duration_min",
        "passenger_count",
        "pickup_hour",
        "is_weekend",
    ]
    return (
        sample[numeric_columns]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .T.reset_index(names="variable")
    )


def write_csv(data: pd.DataFrame, path: Path) -> None:
    """Write a stable CSV representation suitable for checksumming."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
        float_format="%.10f",
        lineterminator="\n",
    )


def run_preparation(
    raw_path: Path,
    processed_path: Path,
    tables_dir: Path,
    metadata_path: Path,
    sample_size: int = 10_000,
    seed: int = RANDOM_SEED,
) -> dict[str, int | str]:
    """Execute preparation, export audit evidence, and return metadata."""
    raw_sha256 = sha256_file(raw_path)
    if raw_sha256 != EXPECTED_RAW_SHA256:
        raise ValueError(
            "Raw file checksum differs from the verified January 2026 source: "
            f"{raw_sha256}"
        )

    raw_data = load_raw_data(raw_path)
    eligible, audit = filter_trip_records(raw_data)
    engineered = engineer_features(eligible)
    sample = deterministic_sample(engineered, sample_size=sample_size, seed=seed)

    audit = pd.concat(
        [
            audit,
            pd.DataFrame(
                [
                    {
                        "stage": "deterministic_sample",
                        "records_before": len(engineered),
                        "records_removed": len(engineered) - len(sample),
                        "records_after": len(sample),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    write_csv(sample, processed_path)
    write_csv(audit, tables_dir / "data_quality_flow.csv")
    write_csv(
        missingness_table(raw_data, sample),
        tables_dir / "data_quality_missingness.csv",
    )
    write_csv(
        range_table(add_duration(raw_data), eligible, sample),
        tables_dir / "data_quality_ranges.csv",
    )
    write_csv(
        descriptive_statistics(sample),
        tables_dir / "descriptive_statistics.csv",
    )

    metadata: dict[str, int | str] = {
        "study_title": (
            "Robust Detection of Anomalous Taxi Fares Using Least Trimmed "
            "Squares Regression"
        ),
        "source_page": SOURCE_PAGE,
        "source_url": SOURCE_URL,
        "retrieval_date": date.today().isoformat(),
        "raw_filename": raw_path.name,
        "raw_file_size_bytes": raw_path.stat().st_size,
        "raw_sha256": raw_sha256,
        "raw_record_count": len(raw_data),
        "eligible_record_count": len(eligible),
        "sample_size": len(sample),
        "random_seed": seed,
        "processed_filename": processed_path.name,
        "processed_sha256": sha256_file(processed_path),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line paths while retaining reproducible defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / RAW_FILENAME,
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "taxi_sample.csv",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tables",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "source_metadata.json",
    )
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    """Run the command-line data preparation workflow."""
    args = parse_args()
    metadata = run_preparation(
        raw_path=args.raw,
        processed_path=args.processed,
        tables_dir=args.tables_dir,
        metadata_path=args.metadata,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
