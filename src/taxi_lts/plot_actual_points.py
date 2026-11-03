"""Readable actual-trip plots shared by the analysis and LaTeX report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.taxi_lts import RANDOM_SEED
from src.taxi_lts.explore_training import PREDICTORS


def save_comparable_trip_plots(
    train: pd.DataFrame,
    coefficients: pd.DataFrame,
    known: np.ndarray,
    figures_dir: Path,
) -> dict[str, object]:
    """Show the corruption band separately from conditional fitted lines.

    The models and contamination labels are supplied by the caller. Only the
    displayed ordinary points are sampled here; no model is refitted.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    distance = float(train.trip_distance.median())
    subset = (
        train.passenger_count.eq(1)
        & train.is_weekend.eq(0)
        & train.pickup_hour.between(11, 13)
        & train.trip_distance.between(distance - 0.35, distance + 0.35)
    ).to_numpy()
    ordinary = np.flatnonzero(subset & ~known)
    selected = np.random.default_rng(RANDOM_SEED).choice(
        ordinary, min(250, len(ordinary)), replace=False
    )
    corrupted = np.flatnonzero(subset & known)
    all_selected = np.sort(np.r_[selected, corrupted])
    duration = train.duration_min.to_numpy()
    fare = train.fare_amount.to_numpy()
    experimental = fare + 100 * known
    xmax = float(train.loc[subset, "duration_min"].quantile(0.995))
    x = np.linspace(1, xmax, 250)
    reference = dict(zip(PREDICTORS, [distance, 0, 1, 0, -1, 0], strict=True))

    def line(estimator: str, scenario: str) -> np.ndarray:
        params = coefficients.loc[coefficients.estimator.eq(estimator)].set_index(
            "term"
        )[scenario]
        return (
            params["const"]
            + sum(params[k] * v for k, v in reference.items())
            + params["duration_min"] * x
        )

    plt.rcParams.update(
        {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}
    )
    blue, orange, red = "#2166AC", "#C86A08", "#C62828"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    for axis, scenario, title in zip(
        axes,
        ["clean", "contaminated"],
        ["Before the fare shift", "After the fare shift"],
        strict=True,
    ):
        positions = all_selected if scenario == "clean" else selected
        axis.scatter(
            duration[positions],
            fare[positions],
            s=30,
            alpha=0.65,
            facecolor="#606060",
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )
        axis.plot(
            x, line("OLS", scenario), color=blue, linewidth=2.7, label="OLS", zorder=3
        )
        axis.plot(
            x,
            line("Reweighted LTS", scenario),
            color=orange,
            linewidth=2.7,
            label="Reweighted LTS",
            zorder=3,
        )
        axis.set(
            title=title, xlabel="Trip duration (minutes)", xlim=(1, xmax), ylim=(0, 36)
        )
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Metered fare (USD)")
    axes[0].legend(loc="upper left", frameon=False)
    ix = int(np.argmin(abs(x - 33)))
    low, high = line("OLS", "clean")[ix], line("OLS", "contaminated")[ix]
    axes[1].annotate(
        "",
        xy=(x[ix], high),
        xytext=(x[ix], low),
        arrowprops={"arrowstyle": "<->", "color": blue, "lw": 1.8},
    )
    axes[1].text(
        x[ix] + 1.2, low - 0.3, f"OLS rises\n${high - low:.2f}", color=blue, fontsize=10
    )
    axes[1].text(
        3, 31, "LTS stays close to the grey pattern", color=orange, fontsize=11
    )
    fig.suptitle("How the fitted relationship changes", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figures_dir / "actual_points_clean_stress_fitted_lines.png", dpi=240)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.5, 3.8))
    axis.scatter(
        duration[selected],
        fare[selected],
        s=27,
        color="#606060",
        alpha=0.65,
        edgecolor="white",
        linewidth=0.25,
        label="Unchanged trip",
    )
    axis.scatter(
        duration[corrupted],
        experimental[corrupted],
        s=38,
        color=red,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.3,
        label="Trip with $100 added",
    )
    axis.set(
        xlim=(1, xmax),
        ylim=(0, 135),
        xlabel="Trip duration (minutes)",
        ylabel="Fare used in the stress test (USD)",
        title="Where the artificial outliers appear",
    )
    axis.legend(loc="center right", frameon=False)
    axis.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(figures_dir / "actual_contamination_overview.png", dpi=240)
    plt.close(fig)

    return {
        "comparable_trips": int(subset.sum()),
        "displayed_unchanged": len(selected),
        "displayed_artificial": len(corrupted),
        "x_max": xmax,
        "reference_distance": distance,
        "distance_window": [distance - 0.35, distance + 0.35],
        "training_size": len(train),
        "contaminated_training_size": int(known.sum()),
        "models_refitted_for_plot": False,
    }
