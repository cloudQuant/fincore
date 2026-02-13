"""Tests for report.compute module.

Validates compute_sections with various input combinations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.report.compute import compute_sections


@pytest.fixture
def daily_returns():
    np.random.seed(42)
    return pd.Series(
        np.random.randn(252) * 0.01,
        index=pd.date_range("2020-01-01", periods=252, freq="B"),
        name="strategy",
    )


@pytest.fixture
def benchmark_returns():
    np.random.seed(123)
    return pd.Series(
        np.random.randn(252) * 0.008,
        index=pd.date_range("2020-01-01", periods=252, freq="B"),
        name="benchmark",
    )


class TestComputeSections:
    def test_basic_returns_only(self, daily_returns):
        sections = compute_sections(daily_returns, None, None, None, None, 126)
        assert isinstance(sections, dict)
        assert "cum_returns" in sections or len(sections) > 0

    def test_with_benchmark(self, daily_returns, benchmark_returns):
        sections = compute_sections(daily_returns, benchmark_returns, None, None, None, 126)
        assert isinstance(sections, dict)
        assert "benchmark_cum" in sections

    def test_short_returns(self):
        short = pd.Series(
            [0.01, -0.005, 0.002],
            index=pd.date_range("2020-01-01", periods=3, freq="B"),
        )
        sections = compute_sections(short, None, None, None, None, 126)
        assert isinstance(sections, dict)

    def test_all_zero_returns(self):
        zeros = pd.Series(
            np.zeros(100),
            index=pd.date_range("2020-01-01", periods=100, freq="B"),
        )
        sections = compute_sections(zeros, None, None, None, None, 126)
        assert isinstance(sections, dict)

    def test_sections_keys(self, daily_returns):
        sections = compute_sections(daily_returns, None, None, None, None, 126)
        assert isinstance(sections, dict)
        assert len(sections) > 0
