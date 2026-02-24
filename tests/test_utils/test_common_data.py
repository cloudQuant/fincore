"""Tests for data manipulation utilities in fincore.utils.common_utils."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.utils import common_utils as cu


def test_clip_returns_to_benchmark_clips_when_out_of_range():
    """Test clip_returns_to_benchmark clips returns to benchmark date range."""
    idx_rets = pd.date_range("2020-01-01", periods=5, freq="D")
    idx_b = pd.date_range("2020-01-02", periods=3, freq="D")
    rets = pd.Series(range(5), index=idx_rets, dtype=float)
    bench = pd.Series(range(3), index=idx_b, dtype=float)

    out = cu.clip_returns_to_benchmark(rets, bench)
    assert out.index.equals(idx_b)


def test_clip_returns_to_benchmark_noop_when_already_aligned():
    """Test clip_returns_to_benchmark returns original when already aligned."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    rets = pd.Series(range(3), index=idx, dtype=float)
    bench = pd.Series(range(3), index=idx, dtype=float)
    out = cu.clip_returns_to_benchmark(rets, bench)
    assert out is rets


def test_vectorize_decorator_series_and_dataframe():
    """Test @vectorize decorator handles Series and DataFrame inputs."""
    @cu.vectorize
    def plus_one(s):
        return s + 1

    s = pd.Series([1, 2, 3])
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    out_s = plus_one(s)
    out_df = plus_one(df)

    pd.testing.assert_series_equal(out_s, pd.Series([2, 3, 4]))
    pd.testing.assert_frame_equal(out_df, pd.DataFrame({"a": [2, 3], "b": [4, 5]}))


def test_restride_rolling_window_happy_path_and_errors():
    """Test rolling_window creates correct views and handles errors."""
    a = np.arange(25).reshape(5, 5)
    out = cu.rolling_window(a, 2, mutable=True)
    assert out.shape == (4, 2, 5)

    out[0, 0, 0] = 999
    assert a[0, 0] == 999

    with pytest.raises(ValueError, match="0-length"):
        cu.rolling_window(a, 0)
    with pytest.raises(IndexError, match="scalar"):
        cu.rolling_window(np.array(1.0), 1)
    with pytest.raises(IndexError, match="window length"):
        cu.rolling_window(np.arange(3), 5)


def test_rolling_window_fallback_when_writeable_kwarg_not_supported(monkeypatch):
    """Test rolling_window fallback when writeable kwarg not supported."""
    import numpy as _np
    from numpy.lib.stride_tricks import as_strided as _real_as_strided

    def _as_strided_no_writeable(array, shape, strides, writeable=None):  # noqa: ARG001
        raise TypeError("no writeable")

    a = _np.arange(9).reshape(3, 3)

    def combined_as_strided(array, shape, strides, writeable=None):
        if writeable is None:
            return _real_as_strided(array, shape, strides)
        return _as_strided_no_writeable(array, shape, strides, writeable)

    monkeypatch.setattr(cu, "as_strided", combined_as_strided)

    with pytest.raises(ValueError, match="Cannot create a writable rolling window"):
        cu.rolling_window(a, 2, mutable=True)


def test_standardize_data_center_and_scale():
    """Test standardize_data centers and scales data."""
    x = np.array([1.0, 2.0, 3.0])
    z = cu.standardize_data(x)
    assert abs(float(np.mean(z))) < 1e-12
    assert abs(float(np.std(z)) - 1.0) < 1e-12


def test_to_series_converts_first_column():
    """Test to_series function converts DataFrame first column to Series."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    s = cu.to_series(df)
    pd.testing.assert_series_equal(s, pd.Series([1, 2], name="a"))


def test_analyze_dataframe_differences_prints_for_identical_and_different(capsys):
    """Test analyze_dataframe_differences output for identical and different DataFrames."""
    df1 = pd.DataFrame({"a": [1, 2]}, index=pd.date_range("2024-01-01", periods=2, freq="D"))
    df2 = df1.copy()
    cu.analyze_dataframe_differences(df1, df2)
    out = capsys.readouterr().out
    assert "The DataFrames are identical." in out

    df3 = pd.DataFrame({"a": [1, 999]}, index=df1.index)
    cu.analyze_dataframe_differences(df1, df3)
    out = capsys.readouterr().out
    assert "The DataFrames are not identical" in out


def test_analyze_dataframe_differences_prints_index_columns_dtype_and_metadata_differences(capsys):
    """Test analyze_dataframe_differences prints all types of differences."""
    idx1 = pd.date_range("2024-01-01", periods=2, freq="D")
    idx2 = pd.date_range("2024-01-02", periods=2, freq="D")
    df1 = pd.DataFrame({"a": [1, 2]}, index=idx1)
    df2 = pd.DataFrame({"b": [1.0, 2.0]}, index=idx2)

    cu.analyze_dataframe_differences(df1, df2)
    out = capsys.readouterr().out
    assert "Indices are different" in out
    assert "Columns are different" in out
    assert "Dtypes are different" in out


def test_analyze_series_differences_prints_for_identical_and_different(capsys):
    """Test analyze_series_differences output for identical and different Series."""
    s1 = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, freq="D"))
    s2 = s1.copy()
    cu.analyze_series_differences(s1, s2)
    out = capsys.readouterr().out
    assert "The Series are identical." in out

    s3 = pd.Series([1.0, 999.0], index=s1.index)
    cu.analyze_series_differences(s1, s3)
    out = capsys.readouterr().out
    assert "The Series are not identical" in out


def test_analyze_series_differences_prints_index_dtype_and_metadata_differences(capsys):
    """Test analyze_series_differences prints all types of differences."""
    idx1 = pd.date_range("2024-01-01", periods=2, freq="D")
    idx2 = pd.date_range("2024-01-02", periods=2, freq="D")
    s1 = pd.Series([1, 2], index=idx1, dtype="int64")
    s2 = pd.Series([1.0, 2.0], index=idx2, dtype="float64")
    cu.analyze_series_differences(s1, s2)
    out = capsys.readouterr().out
    assert "Indices are different" in out
    assert "Dtypes are different" in out
    assert "Index frequencies are identical" in out or "Index frequencies are different" in out


def test_analyze_series_differences_different_freq(capsys):
    """Test analyze_series_differences with different index frequencies."""
    idx1 = pd.date_range("2024-01-01", periods=2, freq="D")
    idx2 = pd.date_range("2024-01-01", periods=2, freq="W")
    s1 = pd.Series([1, 2], index=idx1, dtype="int64")
    s2 = pd.Series([1.0, 2.0], index=idx2, dtype="float64")
    cu.analyze_series_differences(s1, s2)
    out = capsys.readouterr().out
    assert "Index frequencies are different" in out

    # Same for DataFrame
    df1 = pd.DataFrame({"a": [1, 2]}, index=idx1)
    df2 = pd.DataFrame({"b": [1.0, 2.0]}, index=idx2)
    cu.analyze_dataframe_differences(df1, df2)
    out = capsys.readouterr().out
    assert "Index frequencies are different" in out
