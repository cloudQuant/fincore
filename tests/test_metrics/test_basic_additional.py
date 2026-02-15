import numpy as np
import pandas as pd
import pytest

from fincore.metrics.basic import (
    aligned_series,
    annualization_factor,
    ensure_datetime_index_series,
    flatten,
    to_pandas,
)


def test_ensure_datetime_index_series_empty_and_passthrough() -> None:
    out = ensure_datetime_index_series([])
    assert isinstance(out, pd.Series)
    assert out.empty

    idx = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    assert ensure_datetime_index_series(s) is s


def test_flatten_series_branch() -> None:
    s = pd.Series([1.0, 2.0, 3.0])
    out = flatten(s)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)


def test_annualization_factor_invalid_period_raises() -> None:
    with pytest.raises(ValueError, match="Period cannot"):
        annualization_factor("not-a-period")


def test_to_pandas_converts_and_errors() -> None:
    s = pd.Series([1.0, 2.0])
    assert to_pandas(s) is s

    assert isinstance(to_pandas(np.array([1.0, 2.0])), pd.Series)
    assert isinstance(to_pandas(np.array([[1.0, 2.0], [3.0, 4.0]])), pd.DataFrame)

    with pytest.raises(ValueError, match="dim > 2"):
        to_pandas(np.zeros((2, 2, 2)))


def test_aligned_series_dataframe_dataframe_and_mixed() -> None:
    idx1 = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
    idx2 = pd.date_range("2024-01-03", periods=5, freq="B", tz="UTC")

    df1 = pd.DataFrame({"A": range(5)}, index=idx1)
    df2 = pd.DataFrame({"A": range(5)}, index=idx2)

    a1, a2 = aligned_series(df1, df2)
    assert isinstance(a1, pd.DataFrame)
    assert isinstance(a2, pd.DataFrame)
    assert a1.index.equals(a2.index)
    assert len(a1) == len(idx1.intersection(idx2))

    # df + series alignment.
    s = pd.Series(range(5), index=idx2)
    adf, aser = aligned_series(df1, s)
    assert isinstance(adf, pd.DataFrame)
    assert isinstance(aser, pd.Series)
    assert adf.index.equals(aser.index)

    aser2, adf2 = aligned_series(s, df1)
    assert isinstance(aser2, pd.Series)
    assert isinstance(adf2, pd.DataFrame)
    assert aser2.index.equals(adf2.index)

    # Optimization for ndarrays of same length.
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([0.0, 0.0, 0.0])
    out = aligned_series(x, y)
    assert out == (x, y)

    # Cover "indices equal" fast path for df+series.
    s_same = pd.Series(range(len(idx1)), index=idx1)
    out_df, out_s = aligned_series(df1, s_same)
    assert out_df is df1
    assert out_s is s_same
