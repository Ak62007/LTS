"""Tests for the controlled response-contamination experiment."""

import numpy as np
import pandas as pd

from src.taxi_lts.explore_training import PREDICTORS, TARGET
from src.taxi_lts.stress_test import (
    CONTAMINATION_ADDITION,
    contaminate_training_response,
    contamination_positions,
)


def test_contamination_positions_are_deterministic_and_exact() -> None:
    """Five percent of 40,000 rows must mean exactly 2,000 fixed positions."""
    first = contamination_positions(40_000)
    second = contamination_positions(40_000)
    assert len(first) == 2_000
    assert len(np.unique(first)) == 2_000
    np.testing.assert_array_equal(first, second)


def test_contamination_changes_only_selected_responses() -> None:
    """Predictors and unselected responses must remain unchanged."""
    n = 100
    training = pd.DataFrame(
        {
            TARGET: np.arange(n, dtype=float),
            **{name: np.arange(n, dtype=float) for name in PREDICTORS},
        }
    )
    predictors_before = training[PREDICTORS].copy(deep=True)
    response_before = training[TARGET].to_numpy(copy=True)
    contaminated, mask = contaminate_training_response(training)

    assert mask.sum() == 5
    np.testing.assert_allclose(
        contaminated[mask] - response_before[mask], CONTAMINATION_ADDITION
    )
    np.testing.assert_array_equal(contaminated[~mask], response_before[~mask])
    pd.testing.assert_frame_equal(training[PREDICTORS], predictors_before)
