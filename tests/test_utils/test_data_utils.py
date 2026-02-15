"""Tests for fincore.utils.data_utils.

Focus on rolling window utilities and branch coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.utils.data_utils import down, roll, rolling_window, up


def test_rolling_window_valid_shape_and_contents():
    arr = np.arange(5, dtype=float)
    out = rolling_window(arr, 3)
    assert out.shape == (3, 3)
    assert np.array_equal(out[0], np.array([0.0, 1.0, 2.0]))
    assert np.array_equal(out[-1], np.array([2.0, 3.0, 4.0]))


def test_rolling_window_rejects_non_1d():
    arr = np.arange(6).reshape(2, 3)
    with pytest.raises(ValueError, match="1D"):
        rolling_window(arr, 2)


def test_rolling_window_rejects_window_larger_than_array():
    with pytest.raises(ValueError, match="greater than array length"):
        rolling_window([1, 2], 3)


def test_roll_pandas_empty_when_window_larger_than_data():
    s = pd.Series([0.1, 0.2, 0.3], index=pd.date_range("2020-01-01", periods=3))
    out = roll(s, function=lambda x: float(np.mean(x)), window=10)
    assert isinstance(out, pd.Series)
    assert len(out) == 0
    assert out.index.equals(s.index[:0])


def test_roll_pandas_one_argument():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.date_range("2020-01-01", periods=5))
    out = roll(s, function=lambda x: float(np.mean(x)), window=3)
    expected = pd.Series([2.0, 3.0, 4.0], index=s.index[2:], dtype=float)
    pd.testing.assert_series_equal(out, expected)


def test_roll_pandas_two_arguments():
    r = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2020-01-01", periods=4))
    f = pd.Series([0.5, 1.0, 1.5, 2.0], index=r.index)

    def fstat(x, y):
        return float(np.mean(x - y))

    out = roll(r, f, function=fstat, window=2)
    expected = pd.Series([0.75, 1.25, 1.75], index=r.index[1:], dtype=float)
    pd.testing.assert_series_equal(out, expected)


def test_roll_ndarray_empty_when_window_larger_than_data():
    arr = np.array([1.0, 2.0, 3.0])
    out = roll(arr, function=lambda x: float(np.mean(x)), window=5)
    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_roll_ndarray_one_argument():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = roll(arr, function=lambda x: float(np.mean(x)), window=2)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, np.array([1.5, 2.5, 3.5, 4.5]))


def test_roll_ndarray_two_arguments():
    r = np.array([1.0, 2.0, 3.0, 4.0])
    f = np.array([0.5, 1.0, 1.5, 2.0])
    out = roll(r, f, function=lambda x, y: float(np.mean(x - y)), window=2)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, np.array([0.75, 1.25, 1.75]))


def test_roll_raises_when_more_than_two_return_sets():
    arr = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="more than 2"):
        roll(arr, arr, arr, function=lambda x: float(np.mean(x)), window=2)


def test_roll_raises_when_two_args_types_differ():
    s = pd.Series([1.0, 2.0, 3.0])
    a = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="not the same"):
        roll(s, a, function=lambda x: float(np.mean(x)), window=2)


def test_up_down_filter_factor_periods_numpy():
    returns = np.array([1.0, 2.0, 3.0, 4.0])
    factor = np.array([1.0, -1.0, 2.0, -2.0])

    def stat(r, f):
        assert isinstance(r, np.ndarray)
        assert isinstance(f, np.ndarray)
        return float(np.sum(r) + np.sum(f))

    up_out = up(returns, factor, function=lambda r, f: stat(r, f) if np.all(f > 0) else np.nan)
    assert up_out == 7.0

    down_out = down(returns, factor, function=lambda r, f: stat(r, f) if np.all(f < 0) else np.nan)
    assert down_out == 3.0
