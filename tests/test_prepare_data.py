"""Tests for deterministic taxi-data preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.taxi_lts.prepare_data import (
    OUTPUT_COLUMNS,
    deterministic_sample,
    engineer_features,
    filter_trip_records,
)


def synthetic_trips() -> pd.DataFrame:
    """Build rows where each invalid observation targets one filtering rule."""
    pickup = pd.Timestamp("2026-01-03 23:30:00")
    rows = []
    for source_row_id in range(9):
        rows.append(
            {
                "source_row_id": source_row_id,
                "tpep_pickup_datetime": pickup,
                "tpep_dropoff_datetime": pickup + pd.Timedelta(minutes=30),
                "passenger_count": 2,
                "trip_distance": 4.5,
                "RatecodeID": 1,
                "fare_amount": 22.0,
            }
        )
    rows[1]["tpep_pickup_datetime"] = pd.Timestamp("2025-12-31 23:30:00")
    rows[1]["tpep_dropoff_datetime"] = pd.Timestamp("2026-01-01 00:00:00")
    rows[2]["RatecodeID"] = 2
    rows[3]["trip_distance"] = 0.0
    rows[4]["fare_amount"] = -5.0
    rows[5]["tpep_dropoff_datetime"] = pickup
    rows[6]["passenger_count"] = 0
    rows[7]["tpep_dropoff_datetime"] = pickup + pd.Timedelta(hours=8)
    rows[8]["passenger_count"] = 12
    return pd.DataFrame(rows)


def test_filters_apply_sequentially_and_keep_all_valid_trips() -> None:
    eligible, audit = filter_trip_records(synthetic_trips())

    assert eligible["source_row_id"].tolist() == [0, 7, 8]
    rule_rows = audit.iloc[1:7]
    assert rule_rows["records_removed"].tolist() == [1, 1, 1, 1, 1, 1]
    assert audit.iloc[-1]["records_after"] == 3


def test_no_upper_duration_or_passenger_limit_is_applied() -> None:
    eligible, _ = filter_trip_records(synthetic_trips())

    assert eligible["duration_min"].max() == 8 * 60
    assert eligible["passenger_count"].max() == 12


def test_engineered_time_features_are_correct() -> None:
    eligible, _ = filter_trip_records(synthetic_trips())
    result = engineer_features(eligible)

    assert result.columns.tolist() == OUTPUT_COLUMNS
    assert result.loc[0, "pickup_hour"] == 23
    assert result.loc[0, "is_weekend"] == 1
    angle = 2 * np.pi * 23 / 24
    assert np.isclose(result.loc[0, "pickup_hour_sin"], np.sin(angle))
    assert np.isclose(result.loc[0, "pickup_hour_cos"], np.cos(angle))


def test_sampling_is_deterministic_and_source_ordered() -> None:
    data = pd.DataFrame({"source_row_id": np.arange(100), "value": np.arange(100)})

    first = deterministic_sample(data, sample_size=20, seed=42)
    second = deterministic_sample(data, sample_size=20, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first["source_row_id"].is_monotonic_increasing


def test_generated_sample_matches_recorded_checksum() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sample_path = project_root / "data" / "processed" / "taxi_sample.csv"
    metadata_path = project_root / "data" / "source_metadata.json"
    assert sample_path.exists()
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    sample = pd.read_csv(
        sample_path, parse_dates=["pickup_datetime", "dropoff_datetime"]
    )

    assert len(sample) == 10_000
    assert sample["source_row_id"].is_unique
    assert sample["source_row_id"].is_monotonic_increasing
    assert sample.isna().sum().sum() == 0
    assert sample["pickup_datetime"].between("2026-01-01", "2026-01-31 23:59:59").all()
    assert sample["fare_amount"].gt(0).all()
    assert sample["trip_distance"].gt(0).all()
    assert sample["duration_min"].ge(1).all()
    assert sample["passenger_count"].ge(1).all()
    assert np.allclose(
        sample["pickup_hour_sin"] ** 2 + sample["pickup_hour_cos"] ** 2,
        1,
    )
    assert digest == metadata["processed_sha256"]
