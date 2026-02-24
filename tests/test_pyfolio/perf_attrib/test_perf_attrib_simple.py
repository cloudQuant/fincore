"""Tests for simple performance attribution.

Tests perf_attrib with basic scenarios.
Split from test_perf_attrib.py for maintainability.
"""

from __future__ import annotations

import unittest

import pandas as pd

from .conftest import (
    _empyrical_compat_perf_attrib_result,
    create_perf_attrib_stats,
    perf_attrib,
)


class TestPerfAttribSimple(unittest.TestCase):
    """Test simple performance attribution scenarios."""

    def test_perf_attrib_simple(self):
        """Test basic performance attribution calculation."""
        start_date = "2017-01-01"
        periods = 2
        dts = pd.date_range(start_date, periods=periods)
        dts.name = "dt"

        tickers = ["stock1", "stock2"]
        styles = ["risk_factor1", "risk_factor2"]

        returns = pd.Series(data=[0.1, 0.1], index=dts)

        factor_returns = pd.DataFrame(
            columns=styles, index=dts, data={"risk_factor1": [0.1, 0.1], "risk_factor2": [0.1, 0.1]}
        )

        positions = pd.DataFrame(index=dts, data={"stock1": [20, 20], "stock2": [50, 50], "cash": [0, 0]})

        index = pd.MultiIndex.from_product([dts, tickers], names=["dt", "ticker"])

        factor_loadings = pd.DataFrame(
            columns=styles,
            index=index,
            data={"risk_factor1": [0.25, 0.25, 0.25, 0.25], "risk_factor2": [0.25, 0.25, 0.25, 0.25]},
        )

        expected_perf_attrib_output = _empyrical_compat_perf_attrib_result(
            index=dts,
            columns=[
                "risk_factor1",
                "risk_factor2",
                "total_returns",
                "common_returns",
                "specific_returns",
                "tilt_returns",
                "timing_returns",
            ],
            data={
                "risk_factor1": [0.025, 0.025],
                "risk_factor2": [0.025, 0.025],
                "common_returns": [0.05, 0.05],
                "specific_returns": [0.05, 0.05],
                "tilt_returns": [0.05, 0.05],
                "timing_returns": [0.0, 0.0],
                "total_returns": returns,
            },
        )

        expected_exposures_portfolio = pd.DataFrame(
            index=dts,
            columns=["risk_factor1", "risk_factor2"],
            data={"risk_factor1": [0.25, 0.25], "risk_factor2": [0.25, 0.25]},
        )

        exposures_portfolio, perf_attrib_output = perf_attrib(returns, positions, factor_returns, factor_loadings)

        expected_perf_attrib_output.equals(perf_attrib_output)
        expected_exposures_portfolio.equals(exposures_portfolio)

        # test long and short positions
        positions = pd.DataFrame(index=dts, data={"stock1": [20, 20], "stock2": [-20, -20], "cash": [20, 20]})

        exposures_portfolio, perf_attrib_output = perf_attrib(returns, positions, factor_returns, factor_loadings)

        expected_perf_attrib_output = _empyrical_compat_perf_attrib_result(
            index=dts,
            columns=[
                "risk_factor1",
                "risk_factor2",
                "total_returns",
                "common_returns",
                "specific_returns",
                "tilt_returns",
                "timing_returns",
            ],
            data={
                "risk_factor1": [0.0, 0.0],
                "risk_factor2": [0.0, 0.0],
                "common_returns": [0.0, 0.0],
                "specific_returns": [0.1, 0.1],
                "tilt_returns": [0.0, 0.0],
                "timing_returns": [0.0, 0.0],
                "total_returns": returns,
            },
        )

        expected_exposures_portfolio = pd.DataFrame(
            index=dts,
            columns=["risk_factor1", "risk_factor2"],
            data={"risk_factor1": [0.0, 0.0], "risk_factor2": [0.0, 0.0]},
        )

        expected_perf_attrib_output.equals(perf_attrib_output)
        expected_exposures_portfolio.equals(exposures_portfolio)

        perf_attrib_summary, exposures_summary = create_perf_attrib_stats(perf_attrib_output, exposures_portfolio)

        self.assertEqual(
            perf_attrib_summary["Annualized Specific Return"], perf_attrib_summary["Annualized Total Return"]
        )

        self.assertEqual(perf_attrib_summary["Cumulative Specific Return"], perf_attrib_summary["Total Returns"])

        exposures_summary.equals(
            pd.DataFrame(
                0.0,
                index=["risk_factor1", "risk_factor2"],
                columns=["Average Risk Factor Exposure", "Annualized Return", "Cumulative Return"],
            )
        )
