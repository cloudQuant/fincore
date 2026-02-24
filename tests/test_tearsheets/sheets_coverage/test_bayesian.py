"""Tests for bayesian tear sheet.

Tests create_bayesian_tear_sheet coverage scenarios.
Split from test_sheets_more_coverage.py for maintainability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fincore.tearsheets.sheets as sheets

from .test_fixtures import _FakePyfolioBayes


def test_create_bayesian_tear_sheet_requires_live_start_date() -> None:
    """Test that bayesian tear sheet requires live_start_date."""
    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx, name="r")
    with pytest.raises(NotImplementedError):
        sheets.create_bayesian_tear_sheet(_FakePyfolioBayes(), returns)


def test_create_bayesian_tear_sheet_happy_path_with_benchmark_and_stoch_vol(monkeypatch) -> None:
    """Test bayesian tear sheet with benchmark and stoch_vol."""
    # Keep the test fast and deterministic: avoid seaborn overhead.
    monkeypatch.setattr(sheets.sns, "histplot", lambda *args, **kwargs: None)
    monkeypatch.setattr(sheets, "timer", lambda *_args, **_kwargs: _args[1])

    idx = pd.date_range("2022-01-03", periods=650, freq="B", tz="UTC")
    returns = pd.Series(0.0001, index=idx, name="r")
    benchmark = returns * 0.5
    live_start_date = idx[500]

    pyf = _FakePyfolioBayes()
    fig = sheets.create_bayesian_tear_sheet(
        pyf,
        returns,
        benchmark_rets=benchmark,
        live_start_date=live_start_date,
        samples=10,
        run_flask_app=True,
        stoch_vol=True,
        progressbar=False,
    )
    assert fig is not None
    assert "run_model:t" in pyf.calls
    assert "run_model:best" in pyf.calls
    assert "run_model:alpha_beta" in pyf.calls
    assert "model_stoch_vol" in pyf.calls


def test_bayesian_tear_sheet_with_small_dataset_line_889(monkeypatch) -> None:
    """Test bayesian tear sheet when df_train.size <= returns_cutoff (line 889)."""
    # Use a small dataset to hit the else branch at line 888-889
    import warnings

    # Suppress seaborn warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    monkeypatch.setattr(sheets, "timer", lambda *_args, **_kwargs: _args[1])
    monkeypatch.setattr(sheets.sns, "histplot", lambda *args, **kwargs: None)

    # Create small dataset (< 400 returns for stoch_vol)
    idx = pd.date_range("2022-01-03", periods=100, freq="B", tz="UTC")
    returns = pd.Series(0.0001, index=idx, name="r")
    benchmark = returns * 0.5
    live_start_date = idx[50]

    pyf = _FakePyfolioBayes()
    fig = sheets.create_bayesian_tear_sheet(
        pyf,
        returns,
        benchmark_rets=benchmark,
        live_start_date=live_start_date,
        samples=10,
        run_flask_app=True,
        stoch_vol=True,
        progressbar=False,
    )
    assert fig is not None


def test_bayesian_tear_sheet_run_flask_app_false_line_903(monkeypatch) -> None:
    """Test bayesian tear sheet with run_flask_app=False (line 903)."""
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

    idx = pd.date_range("2022-01-03", periods=500, freq="B", tz="UTC")
    returns = pd.Series(0.0001, index=idx, name="r")
    live_start_date = idx[400]

    pyf = _FakePyfolioBayes()
    # run_flask_app=False means function returns None
    result = sheets.create_bayesian_tear_sheet(
        pyf,
        returns,
        live_start_date=live_start_date,
        samples=10,
        run_flask_app=False,
        stoch_vol=False,
        progressbar=False,
    )
    # Should return None when run_flask_app=False
    assert result is None
