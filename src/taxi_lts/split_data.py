"""Create the fixed 50,000-trip modelling sample and train-test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.taxi_lts import RANDOM_SEED
from src.taxi_lts.prepare_data import (
    PROJECT_ROOT,
    RAW_FILENAME,
    deterministic_nested_sample,
    engineer_features,
    filter_trip_records,
    load_raw_data,
    sha256_file,
    write_csv,
)

MODELING_SAMPLE_SIZE = 50_000
TEST_SIZE = 10_000
EXPECTED_BENCHMARK_SOURCE_ID_SHA256 = (
    "bbb8982488eb73d9bc8637ae27735972c71f35be2e7d6503ca5ea6ea9d0068f8"
)


def source_id_checksum(source_ids: pd.Series | np.ndarray) -> str:
    """Return a stable checksum for an ordered sequence of source row IDs."""
    values = np.asarray(source_ids, dtype="int64")
    return hashlib.sha256(values.tobytes()).hexdigest()


def assign_train_test_split(
    sample: pd.DataFrame,
    *,
    test_size: int = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Assign each modelling row exactly once using a fixed random split."""
    if test_size <= 0 or test_size >= len(sample):
        raise ValueError("test_size must be between zero and the sample size.")
    row_positions = np.arange(len(sample), dtype=np.int64)
    train_positions, test_positions = train_test_split(
        row_positions,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    split = np.full(len(sample), "train", dtype="<U5")
    split[test_positions] = "test"

    result = sample.copy()
    result.insert(0, "modeling_row_id", row_positions)
    result["split"] = split
    if len(train_positions) + len(test_positions) != len(sample):
        raise AssertionError("Train-test assignment did not cover every row.")
    return result


def split_summary(modeling_data: pd.DataFrame) -> pd.DataFrame:
    """Create counts and identity checksums without inspecting outcomes."""
    rows: list[dict[str, int | float | str]] = []
    for label in ["all", "train", "test"]:
        subset = (
            modeling_data
            if label == "all"
            else modeling_data.loc[modeling_data["split"].eq(label)]
        )
        rows.append(
            {
                "split": label,
                "record_count": len(subset),
                "fraction": len(subset) / len(modeling_data),
                "unique_source_ids": int(subset["source_row_id"].nunique()),
                "source_id_sha256": source_id_checksum(subset["source_row_id"]),
            }
        )
    return pd.DataFrame(rows)


def run_split(
    raw_path: Path,
    output_path: Path,
    summary_path: Path,
    metadata_path: Path,
) -> dict[str, int | float | str | bool]:
    """Build, verify, and export the modelling sample and split evidence."""
    raw_data = load_raw_data(raw_path)
    eligible, _ = filter_trip_records(raw_data)
    engineered = engineer_features(eligible)
    sample = deterministic_nested_sample(
        engineered, MODELING_SAMPLE_SIZE, seed=RANDOM_SEED
    )
    sample_checksum = source_id_checksum(sample["source_row_id"])
    if sample_checksum != EXPECTED_BENCHMARK_SOURCE_ID_SHA256:
        raise ValueError(
            "The 50,000-row sample differs from the verified FAST-LTS benchmark."
        )

    modeling_data = assign_train_test_split(sample, seed=RANDOM_SEED)
    summary = split_summary(modeling_data)
    if modeling_data["source_row_id"].nunique() != MODELING_SAMPLE_SIZE:
        raise AssertionError("Modelling source IDs must be unique.")
    if set(modeling_data["split"].unique()) != {"train", "test"}:
        raise AssertionError("Every modelling row must be train or test.")

    write_csv(modeling_data, output_path)
    write_csv(summary, summary_path)
    summary_by_split = summary.set_index("split")
    metadata: dict[str, int | float | str | bool] = {
        "eligible_population_size": len(engineered),
        "modeling_sample_size": len(modeling_data),
        "training_size": int(summary_by_split.loc["train", "record_count"]),
        "test_size": int(summary_by_split.loc["test", "record_count"]),
        "test_fraction": TEST_SIZE / MODELING_SAMPLE_SIZE,
        "selection_seed": RANDOM_SEED,
        "split_seed": RANDOM_SEED,
        "selection_method": "prefix of NumPy Generator permutation",
        "split_method": "sklearn train_test_split with shuffle=True",
        "benchmark_identity_verified": True,
        "sample_source_id_sha256": sample_checksum,
        "train_source_id_sha256": str(
            summary_by_split.loc["train", "source_id_sha256"]
        ),
        "test_source_id_sha256": str(summary_by_split.loc["test", "source_id_sha256"]),
        "modeling_csv_sha256": sha256_file(output_path),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse reproducible path overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / RAW_FILENAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "taxi_modeling_50000.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tables" / "model_split_summary.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "modeling_metadata.json",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line modelling split workflow."""
    args = parse_args()
    metadata = run_split(args.raw, args.output, args.summary, args.metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
