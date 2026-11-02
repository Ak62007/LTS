"""Tests for the fixed modelling sample and train-test split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.taxi_lts.prepare_data import deterministic_nested_sample, sha256_file
from src.taxi_lts.split_data import (
    EXPECTED_BENCHMARK_SOURCE_ID_SHA256,
    assign_train_test_split,
    source_id_checksum,
)


def test_nested_samples_are_prefixes_of_one_permutation() -> None:
    data = pd.DataFrame({"source_row_id": np.arange(1_000)})

    small = deterministic_nested_sample(data, 100, seed=42)
    large = deterministic_nested_sample(data, 250, seed=42)

    pd.testing.assert_frame_equal(small, large.iloc[:100].reset_index(drop=True))


def test_split_is_deterministic_exclusive_and_complete() -> None:
    sample = pd.DataFrame({"source_row_id": np.arange(500)})

    first = assign_train_test_split(sample, test_size=100, seed=42)
    second = assign_train_test_split(sample, test_size=100, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first["source_row_id"].is_unique
    assert first["split"].value_counts().to_dict() == {"train": 400, "test": 100}
    train_ids = set(first.loc[first["split"].eq("train"), "source_row_id"])
    test_ids = set(first.loc[first["split"].eq("test"), "source_row_id"])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(sample["source_row_id"])


def test_generated_modeling_artifact_matches_metadata() -> None:
    project_root = Path(__file__).resolve().parents[1]
    modeling_path = project_root / "data" / "processed" / "taxi_modeling_50000.csv"
    metadata_path = project_root / "data" / "modeling_metadata.json"
    assert modeling_path.exists()
    assert metadata_path.exists()

    data = pd.read_csv(modeling_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(data) == 50_000
    assert data["modeling_row_id"].tolist() == list(range(50_000))
    assert data["source_row_id"].is_unique
    assert data.isna().sum().sum() == 0
    assert data["split"].value_counts().to_dict() == {"train": 40_000, "test": 10_000}
    train_ids = set(data.loc[data["split"].eq("train"), "source_row_id"])
    test_ids = set(data.loc[data["split"].eq("test"), "source_row_id"])
    assert train_ids.isdisjoint(test_ids)
    assert source_id_checksum(data["source_row_id"]) == (
        EXPECTED_BENCHMARK_SOURCE_ID_SHA256
    )
    assert sha256_file(modeling_path) == metadata["modeling_csv_sha256"]
    assert (
        source_id_checksum(data.loc[data["split"].eq("train"), "source_row_id"])
        == metadata["train_source_id_sha256"]
    )
    assert (
        source_id_checksum(data.loc[data["split"].eq("test"), "source_row_id"])
        == metadata["test_source_id_sha256"]
    )
