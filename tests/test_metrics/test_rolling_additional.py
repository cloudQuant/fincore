from __future__ import annotations

import numpy as np
import pandas as pd

from fincore.metrics import rolling as rm


def test_roll_alpha_returns_empty_for_short_input_series_with_datetime_index():
    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    rets = pd.Series([0.001] * len(idx), index=idx)
    fac = pd.Series([0.0] * len(idx), index=idx)
    out = rm.roll_alpha(rets, fac, window=252)
    assert isinstance(out, pd.Series)
    assert out.empty
    assert out.index.equals(idx[:0])


def test_roll_alpha_returns_empty_for_short_input_ndarray():
    rets = np.array([0.001, 0.002], dtype=float)
    fac = np.array([0.0, 0.0], dtype=float)
    out = rm.roll_alpha(rets, fac, window=10)
    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_roll_beta_returns_empty_for_short_input():
    idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
    rets = pd.Series([0.001] * len(idx), index=idx)
    fac = pd.Series([0.0] * len(idx), index=idx)
    out = rm.roll_beta(rets, fac, window=10)
    assert isinstance(out, pd.Series)
    assert out.empty


def test_roll_alpha_returns_empty_for_short_input_series_with_nondatetime_index():
    rets = pd.Series([0.001, 0.002], index=[0, 1])
    fac = pd.Series([0.0, 0.0], index=[0, 1])
    out = rm.roll_alpha(rets, fac, window=10)
    assert isinstance(out, pd.Series)
    assert out.empty


def test_roll_alpha_ndarray_input_converts_to_series_and_returns_ndarray():
    rets = np.array([0.01, -0.02, 0.005, 0.0, 0.01], dtype=float)
    fac = rets * 0.25
    out = rm.roll_alpha(rets, fac, window=3)
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(rets) - 3 + 1,)


def test_roll_alpha_series_input_returns_series_with_expected_index():
    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    rets = pd.Series([0.01, -0.02, 0.005, 0.0, 0.01, -0.003], index=idx)
    fac = rets * 0.25
    out = rm.roll_alpha(rets, fac, window=3)
    assert isinstance(out, pd.Series)
    assert out.index.equals(idx[2:])


def test_roll_beta_returns_empty_for_short_input_series_with_nondatetime_index():
    rets = pd.Series([0.001, 0.002], index=[0, 1])
    fac = pd.Series([0.0, 0.0], index=[0, 1])
    out = rm.roll_beta(rets, fac, window=10)
    assert isinstance(out, pd.Series)
    assert out.empty


def test_roll_beta_ndarray_input_returns_ndarray_values():
    rets = np.array([0.01, -0.02, 0.005, 0.0, 0.01, -0.003], dtype=float)
    fac = rets * 0.5 + 0.0001
    out = rm.roll_beta(rets, fac, window=3)
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(rets) - 3 + 1,)


def test_roll_beta_returns_empty_for_short_input_ndarray():
    rets = np.array([0.001, 0.002], dtype=float)
    fac = np.array([0.0, 0.0], dtype=float)
    out = rm.roll_beta(rets, fac, window=10)
    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_roll_alpha_beta_returns_empty_dataframe_for_short_input_datetime_index():
    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    rets = pd.Series([0.001] * len(idx), index=idx)
    fac = pd.Series([0.0] * len(idx), index=idx)
    out = rm.roll_alpha_beta(rets, fac, window=252)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["alpha", "beta"]
    assert out.empty
    assert out.index.equals(idx[:0])


def test_roll_alpha_beta_returns_empty_dataframe_for_short_input_nondatetime_index():
    rets = pd.Series([0.001, 0.002], index=[0, 1])
    fac = pd.Series([0.0, 0.0], index=[0, 1])
    out = rm.roll_alpha_beta(rets, fac, window=10)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["alpha", "beta"]
    assert out.empty


def test_roll_alpha_beta_returns_empty_dataframe_for_short_input_ndarray():
    rets = np.array([0.001, 0.002], dtype=float)
    fac = np.array([0.0, 0.0], dtype=float)
    out = rm.roll_alpha_beta(rets, fac, window=10)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["alpha", "beta"]
    assert out.empty


def test_rolling_beta_accepts_factor_dataframe_and_returns_dataframe():
    idx = pd.date_range("2024-01-01", periods=40, freq="B", tz="UTC")
    rets = pd.Series(np.linspace(0.001, -0.001, len(idx)), index=idx)
    factors = pd.DataFrame({"f1": rets * 0.5, "f2": rets * -0.25}, index=idx)
    out = rm.rolling_beta(rets, factors, rolling_window=10)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {"f1", "f2"}


def test_rolling_regression_returns_empty_when_window_too_large():
    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    rets = pd.Series([0.001] * len(idx), index=idx)
    fac = pd.Series([0.0] * len(idx), index=idx)
    out = rm.rolling_regression(rets, fac, rolling_window=252)
    assert list(out.columns) == ["alpha", "beta"]
    assert out.empty


def test_rolling_regression_accepts_ndarray_inputs():
    rets = np.array([0.01, -0.02, 0.01, 0.0, 0.005, -0.003] * 5, dtype=float)
    fac = rets * 0.25
    out = rm.rolling_regression(rets, fac, rolling_window=10)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {"alpha", "beta"}
