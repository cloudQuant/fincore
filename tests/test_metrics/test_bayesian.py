"""Tests for fincore.metrics.bayesian (non-PyMC helpers only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.metrics.bayesian import (
    compute_bayes_cone,
    compute_consistency_score,
    forecast_cone_bootstrap,
    run_model,
    simulate_paths,
    summarize_paths,
)


def test_compute_bayes_cone_shapes_and_keys():
    preds = np.array(
        [
            [0.01, 0.00, -0.01],
            [0.02, -0.01, 0.00],
            [0.00, 0.01, 0.01],
        ],
        dtype=float,
    )
    cone = compute_bayes_cone(preds, starting_value=1.0)
    assert set(cone.keys()) == {1, 5, 25, 50, 75, 95, 99}
    assert all(len(v) == preds.shape[1] for v in cone.values())


def test_compute_consistency_score_length_matches_test_series():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    returns_test = pd.Series([0.01, -0.01, 0.02], index=idx)
    preds = np.array([[0.00, 0.00, 0.00], [0.02, -0.02, 0.02]], dtype=float)
    q = compute_consistency_score(returns_test, preds)
    assert isinstance(q, list)
    assert len(q) == len(returns_test)
    assert all(0.0 <= x <= 1.0 for x in q)


def test_simulate_paths_is_deterministic_with_seed():
    is_returns = pd.Series([0.01, -0.02, 0.03, 0.00], index=pd.date_range("2020-01-01", periods=4))
    s1 = simulate_paths(is_returns, num_days=3, num_samples=5, random_seed=123)
    s2 = simulate_paths(is_returns, num_days=3, num_samples=5, random_seed=123)
    assert s1.shape == (5, 3)
    assert np.array_equal(s1, s2)


def test_summarize_paths_accepts_scalar_and_list_cone_std():
    samples = np.array([[0.01, 0.00, -0.01], [0.02, -0.01, 0.00]], dtype=float)

    b1 = summarize_paths(samples, cone_std=1.0, starting_value=1.0)
    assert list(b1.columns) == [1.0, -1.0]
    assert b1.shape[0] == samples.shape[1]

    b2 = summarize_paths(samples, cone_std=(1.0, 2.0), starting_value=1.0)
    assert set(b2.columns) == {1.0, -1.0, 2.0, -2.0}


def test_forecast_cone_bootstrap_smoke():
    is_returns = pd.Series(np.random.RandomState(0).normal(0, 0.01, 50), index=pd.date_range("2020-01-01", periods=50))
    bounds = forecast_cone_bootstrap(
        is_returns, num_days=5, cone_std=(1.0, 2.0), starting_value=1.0, num_samples=10, random_seed=42
    )
    assert bounds.shape[0] == 5
    assert set(bounds.columns) == {1.0, -1.0, 2.0, -2.0}


def test_run_model_unknown_raises():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    returns_train = pd.Series([0.01, 0.0, -0.01], index=idx)
    with pytest.raises(NotImplementedError, match="not implemented"):
        run_model("nope", returns_train)
