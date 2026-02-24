"""Tests for performance attribution regression.

Tests perf_attrib with real data regression.
Split from test_perf_attrib.py for maintainability.
"""

from __future__ import annotations

import os
import numpy as np
import unittest
import warnings

import pandas as pd

from fincore.empyrical import Empyrical

from .conftest import PerfAttribTestLocation, create_perf_attrib_stats, perf_attrib


class TestPerfAttribRegression(unittest.TestCase, PerfAttribTestLocation):
    """Test performance attribution regression with real data."""

    def test_perf_attrib_regression(self):
        """Test performance attribution with real CSV data."""
        positions = pd.read_csv(os.path.join(self.__location__, "positions.csv"), index_col=0, parse_dates=True)

        positions.columns = [int(col) if col != "cash" else col for col in positions.columns]

        returns = pd.read_csv(os.path.join(self.__location__, "returns.csv"), index_col=0, parse_dates=True, header=None)
        returns = returns.squeeze()  # Manually squeeze if needed
        factor_loadings = pd.read_csv(os.path.join(self.__location__, "factor_loadings.csv"), index_col=[0, 1])
        factor_loadings.index = factor_loadings.index.set_levels(
            pd.to_datetime(factor_loadings.index.levels[0]), level=0
        )

        factor_returns = pd.read_csv(os.path.join(self.__location__, "factor_returns.csv"), index_col=0, parse_dates=True)

        residuals = pd.read_csv(os.path.join(self.__location__, "residuals.csv"), index_col=0, parse_dates=True)

        residuals.columns = [int(col) for col in residuals.columns]

        intercepts = pd.read_csv(os.path.join(self.__location__, "intercepts.csv"), index_col=0, header=None)

        intercepts = intercepts.squeeze()  # Manually squeeze if needed
        risk_exposures_portfolio, perf_attrib_output = perf_attrib(
            returns,
            positions,
            factor_returns,
            factor_loadings,
        )

        specific_returns = perf_attrib_output["specific_returns"]
        common_returns = perf_attrib_output["common_returns"]
        combined_returns = specific_returns + common_returns

        # since all returns are factor returns, common returns should be
        # equivalent to total returns, and specific returns should be 0
        returns.equals(common_returns)

        self.assertTrue(np.isclose(specific_returns, 0).all())

        # specific and common returns combined should equal total returns
        returns.equals(combined_returns)
        # check that residuals + intercepts = specific returns
        self.assertTrue(np.isclose((residuals + intercepts), 0).all())

        # check that exposure * factor returns = common returns
        expected_common_returns = risk_exposures_portfolio.multiply(factor_returns, axis="rows").sum(axis="columns")

        expected_common_returns.equals(common_returns)
        # since factor loadings are ones, portfolio risk exposures
        # should be ones
        pd.testing.assert_frame_equal(
            risk_exposures_portfolio,
            pd.DataFrame(
                np.ones_like(risk_exposures_portfolio),
                index=risk_exposures_portfolio.index,
                columns=risk_exposures_portfolio.columns,
            ),
        )
        risk_exposures_portfolio.equals(
            pd.DataFrame(
                np.ones_like(risk_exposures_portfolio),
                index=risk_exposures_portfolio.index,
                columns=risk_exposures_portfolio.columns,
            )
        )

        perf_attrib_summary, exposures_summary = create_perf_attrib_stats(perf_attrib_output, risk_exposures_portfolio)

        self.assertEqual(Empyrical.annual_return(specific_returns), perf_attrib_summary["Annualized Specific Return"])

        self.assertEqual(Empyrical.annual_return(common_returns), perf_attrib_summary["Annualized Common Return"])

        self.assertEqual(Empyrical.annual_return(combined_returns), perf_attrib_summary["Annualized Total Return"])

        self.assertEqual(Empyrical.sharpe_ratio(specific_returns), perf_attrib_summary["Specific Sharpe Ratio"])

        self.assertEqual(
            Empyrical.cum_returns_final(specific_returns), perf_attrib_summary["Cumulative Specific Return"]
        )

        self.assertEqual(Empyrical.cum_returns_final(common_returns), perf_attrib_summary["Cumulative Common Return"])

        self.assertEqual(Empyrical.cum_returns_final(combined_returns), perf_attrib_summary["Total Returns"])

        avg_factor_exposure = risk_exposures_portfolio.mean().rename("Average Risk Factor Exposure")
        avg_factor_exposure.equals(exposures_summary["Average Risk Factor Exposure"])

        cumulative_returns_by_factor = pd.Series(
            [Empyrical.cum_returns_final(perf_attrib_output[c]) for c in risk_exposures_portfolio.columns],
            name="Cumulative Return",
            index=risk_exposures_portfolio.columns,
        )

        cumulative_returns_by_factor.equals(exposures_summary["Cumulative Return"])
        annualized_returns_by_factor = pd.Series(
            [Empyrical.annual_return(perf_attrib_output[c]) for c in risk_exposures_portfolio.columns],
            name="Annualized Return",
            index=risk_exposures_portfolio.columns,
        )

        annualized_returns_by_factor.equals(exposures_summary["Annualized Return"])
