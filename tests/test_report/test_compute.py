"""Tests for report.compute module.

Validates compute_sections with various input combinations.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from fincore.exceptions import InputContractError
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

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sections = compute_sections(zeros, None, None, None, None, 126)

        assert isinstance(sections, dict)
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0

    def test_sections_keys(self, daily_returns):
        sections = compute_sections(daily_returns, None, None, None, None, 126)
        assert isinstance(sections, dict)
        assert len(sections) > 0

    def test_core_performance_consumes_direct_metric_kernel_once(self, daily_returns, monkeypatch):
        import fincore.report.compute as report_compute

        calls = 0
        original = report_compute.sharpe_ratio

        def recording(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(report_compute, "sharpe_ratio", recording)

        sections = compute_sections(daily_returns, None, None, None, None, 126)

        assert calls == 1
        assert sections["perf_stats"]["Sharpe Ratio"] == original(daily_returns)

    def test_report_uses_context_validation_profile(self, daily_returns):
        invalid = daily_returns.copy()
        invalid.iloc[3] = np.inf

        with pytest.raises(InputContractError, match="finite"):
            compute_sections(invalid, None, None, None, None, 126)

    def test_report_reuses_direct_leverage_and_turnover_once(self, daily_returns, monkeypatch):
        import fincore.report.compute as report_compute

        positions = pd.DataFrame(
            {"AAA": 100.0, "cash": 50.0},
            index=daily_returns.index,
        )
        transactions = pd.DataFrame(
            {"amount": [2.0], "price": [10.0], "symbol": ["AAA"]},
            index=pd.DatetimeIndex([daily_returns.index[3] + pd.Timedelta(hours=10)]),
        )
        calls = {"gross": 0, "turnover": 0}
        original_gross = report_compute.gross_lev
        original_turnover = report_compute.get_turnover

        def recording_gross(*args, **kwargs):
            calls["gross"] += 1
            return original_gross(*args, **kwargs)

        def recording_turnover(*args, **kwargs):
            calls["turnover"] += 1
            return original_turnover(*args, **kwargs)

        monkeypatch.setattr(report_compute, "gross_lev", recording_gross)
        monkeypatch.setattr(report_compute, "get_turnover", recording_turnover)

        sections = compute_sections(daily_returns, None, positions, transactions, None, 126)

        assert calls == {"gross": 1, "turnover": 1}
        assert sections["gross_leverage"] is not None
        assert sections["turnover"] is not None
