"""Benchmark deterministic nested taxi samples with raw and reweighted FAST-LTS."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import pandas as pd
import psutil
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.taxi_lts import RANDOM_SEED
from src.taxi_lts.prepare_data import (
    PROJECT_ROOT,
    RAW_FILENAME,
    deterministic_nested_sample,
    engineer_features,
    filter_trip_records,
    load_raw_data,
)
from src.taxi_lts.robust_lts import fit_reweighted_fast_lts

PREDICTOR_COLUMNS = [
    "trip_distance",
    "duration_min",
    "passenger_count",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "is_weekend",
]
TARGET_COLUMN = "fare_amount"
DEFAULT_SAMPLE_SIZES = [10_000, 25_000, 50_000]


class PeakMemoryMonitor:
    """Poll process RSS during a fit without external profiling dependencies."""

    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_rss = 0
        self.peak_rss = 0

    def __enter__(self) -> PeakMemoryMonitor:
        process = psutil.Process()
        self.start_rss = process.memory_info().rss
        self.peak_rss = self.start_rss

        def poll() -> None:
            while not self._stop.is_set():
                self.peak_rss = max(self.peak_rss, process.memory_info().rss)
                sleep(self.interval_seconds)

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)


def build_nested_pool(raw_path: Path, maximum_size: int) -> pd.DataFrame:
    """Return the first rows of one seed-42 permutation of eligible trips."""
    raw_data = load_raw_data(raw_path)
    eligible, _ = filter_trip_records(raw_data)
    engineered = engineer_features(eligible)
    return deterministic_nested_sample(engineered, maximum_size, seed=RANDOM_SEED)


def original_unit_coefficients(
    scaler: StandardScaler, standardized_coef: np.ndarray, standardized_intercept: float
) -> tuple[np.ndarray, float]:
    """Back-transform a linear model fitted to standardized predictors."""
    coefficients = standardized_coef / scaler.scale_
    intercept = standardized_intercept - float(
        np.sum(standardized_coef * scaler.mean_ / scaler.scale_)
    )
    return coefficients, intercept


def benchmark_one_size(
    pool: pd.DataFrame,
    sample_size: int,
    *,
    alpha: float,
    n_initial_subsets: int,
) -> dict[str, int | float | str]:
    """Fit one guarded benchmark and return machine-readable measurements."""
    sample = pool.iloc[:sample_size]
    scaler = StandardScaler()
    X = scaler.fit_transform(sample[PREDICTOR_COLUMNS].to_numpy(dtype=float))
    y = sample[TARGET_COLUMN].to_numpy(dtype=float)

    with PeakMemoryMonitor() as memory, threadpool_limits(limits=2):
        started = perf_counter()
        result = fit_reweighted_fast_lts(
            X,
            y,
            alpha=alpha,
            n_initial_subsets=n_initial_subsets,
            random_state=RANDOM_SEED,
        )
        elapsed_seconds = perf_counter() - started

    raw_model = result.raw_estimator.model
    raw_coef = np.asarray(raw_model.coef_).reshape(-1)
    raw_intercept = float(np.asarray(raw_model.intercept_).reshape(-1)[0])
    reweighted_coef = np.asarray(result.reweighted_model.coef_).reshape(-1)
    reweighted_intercept = float(result.reweighted_model.intercept_)
    original_coef, original_intercept = original_unit_coefficients(
        scaler, reweighted_coef, reweighted_intercept
    )
    h = len(result.raw_estimator.best_h_subset)
    raw_subset_residuals = result.raw_residuals[result.raw_estimator.best_h_subset]

    output: dict[str, int | float | str] = {
        "sample_size": sample_size,
        "alpha": alpha,
        "h": h,
        "n_predictors": len(PREDICTOR_COLUMNS),
        "n_initial_subsets": n_initial_subsets,
        "elapsed_seconds": elapsed_seconds,
        "rss_before_fit_mib": memory.start_rss / 1024**2,
        "peak_rss_mib": memory.peak_rss / 1024**2,
        "fit_peak_increase_mib": (memory.peak_rss - memory.start_rss) / 1024**2,
        "raw_trimmed_rmse": float(np.sqrt(np.mean(raw_subset_residuals**2))),
        "robpy_corrected_raw_scale": float(result.raw_estimator.scale),
        "raw_mad_scale": result.raw_robust_scale,
        "reweighting_cutoff": result.reweighting_cutoff,
        "reweighted_observations": int(result.reweighting_weights.sum()),
        "zero_weight_observations": int((~result.reweighting_weights).sum()),
        "reweighted_fraction": float(result.reweighting_weights.mean()),
        "raw_intercept_standardized_x": raw_intercept,
        "reweighted_intercept_standardized_x": reweighted_intercept,
        "reweighted_intercept_original_x": original_intercept,
        "sample_source_id_sha256": hashlib.sha256(
            sample["source_row_id"].to_numpy(dtype="int64").tobytes()
        ).hexdigest(),
    }
    for name, value in zip(PREDICTOR_COLUMNS, raw_coef, strict=True):
        output[f"raw_coef_std_{name}"] = float(value)
    for name, value in zip(PREDICTOR_COLUMNS, reweighted_coef, strict=True):
        output[f"reweighted_coef_std_{name}"] = float(value)
    for name, value in zip(PREDICTOR_COLUMNS, original_coef, strict=True):
        output[f"reweighted_coef_original_{name}"] = float(value)
    return output


def update_results(path: Path, result: dict[str, int | float | str]) -> None:
    """Insert or replace one sample-size result in a stable CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([result])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing.loc[existing["sample_size"] != result["sample_size"]]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined = combined.sort_values("sample_size").reset_index(drop=True)
    combined.to_csv(path, index=False, float_format="%.10f", lineterminator="\n")


def parse_args() -> argparse.Namespace:
    """Parse one guarded benchmark invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--initial-subsets", type=int, default=500)
    parser.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / RAW_FILENAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tables" / "lts_scalability_benchmark.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Run exactly one sample size so larger cases can be guarded externally."""
    args = parse_args()
    pool = build_nested_pool(args.raw, args.sample_size)
    result = benchmark_one_size(
        pool,
        args.sample_size,
        alpha=args.alpha,
        n_initial_subsets=args.initial_subsets,
    )
    update_results(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
