"""Tests for date/time related utilities in fincore.utils.common_utils."""

from __future__ import annotations

import pandas as pd
import pytest

from fincore.utils import common_utils as cu


def test_to_utc_localize_and_convert_branches():
    """Test to_utc handles both naive and timezone-aware DataFrames."""
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    df = pd.DataFrame({"x": [1, 2]}, index=idx)
    out1 = cu.to_utc(df.copy())
    assert str(out1.index.tz) == "UTC"

    df2 = pd.DataFrame({"x": [1, 2]}, index=idx.tz_localize("US/Eastern"))
    out2 = cu.to_utc(df2.copy())
    assert str(out2.index.tz) == "UTC"


def test_make_timezone_aware_all_branches():
    """Test make_timezone_aware handles naive and aware timestamps."""
    ts_naive = pd.Timestamp("2020-01-01")
    ts_aware = pd.Timestamp("2020-01-01", tz="UTC")

    assert cu.make_timezone_aware(ts_naive, "UTC").tz is not None
    assert str(cu.make_timezone_aware(ts_aware, "US/Eastern").tz) == "US/Eastern"
    assert cu.make_timezone_aware(ts_aware, None).tz is None
    assert cu.make_timezone_aware(ts_naive, None).tz is None


def test_get_utc_timestamp_localize_and_convert_branches():
    """Test get_utc_timestamp handles both string and Timestamp inputs."""
    out1 = cu.get_utc_timestamp("2020-01-01")
    assert str(out1.tz) == "UTC"

    out2 = cu.get_utc_timestamp(pd.Timestamp("2020-01-01", tz="US/Eastern"))
    assert str(out2.tz) == "UTC"


def test_1_bday_ago_returns_timestamp():
    """Test _1_bday_ago returns a valid Timestamp."""
    ts = cu._1_bday_ago()
    assert isinstance(ts, pd.Timestamp)


def test_get_month_end_freq_both_branches_via_monkeypatch(monkeypatch):
    """Test get_month_end_freq handles different pandas versions."""
    monkeypatch.setattr(cu.pd, "__version__", "2.1.9", raising=False)
    assert cu.get_month_end_freq() == "M"

    monkeypatch.setattr(cu.pd, "__version__", "2.2.0", raising=False)
    assert cu.get_month_end_freq() == "ME"
