"""Tests for training-only exploratory analysis helpers."""

from __future__ import annotations

from pathlib import Path

from src.taxi_lts.explore_training import load_training_data


def test_training_loader_excludes_all_test_rows() -> None:
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "data" / "processed" / "taxi_modeling_50000.csv"
    training = load_training_data(path)

    assert len(training) == 40_000
    assert training["split"].eq("train").all()
    assert "test" not in set(training["split"])
    assert training["source_row_id"].is_unique
