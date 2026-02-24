"""Tests for performance attribution warnings and edge cases.

Tests perf_attrib with missing data, high turnover, and costs.
Split from test_perf_attrib.py for maintainability.
"""

from __future__ import annotations

import unittest
import warnings

import pandas as pd

from .conftest import generate_toy_risk_model_output, mock_transactions_from_positions, perf_attrib


class TestPerfAttribWarnings(unittest.TestCase):
    """Test performance attribution warning scenarios."""

    def test_missing_stocks_and_dates(self):
        """Test perf_attrib with missing stocks and dates."""
        (returns, positions, factor_returns, factor_loadings) = generate_toy_risk_model_output()

        # factor loadings missing a stock should raise a warning
        factor_loadings_missing_stocks = factor_loadings.drop("TLT", level="ticker")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", UserWarning)

            perf_attrib(returns, positions, factor_returns, factor_loadings_missing_stocks)

            self.assertEqual(len(w), 1)
            self.assertIn("The following assets were missing factor loadings: ['TLT']", str(w[-1].message))
            self.assertIn("Ratio of assets missing: 0.333", str(w[-1].message))

            # missing dates should raise a warning
            missing_dates = ["2017-01-01", "2017-01-05"]
            factor_loadings_missing_dates = factor_loadings.drop(missing_dates)

            exposures, perf_attrib_data = perf_attrib(returns, positions, factor_returns, factor_loadings_missing_dates)

            self.assertEqual(len(w), 2)
            self.assertIn(f"Could not find factor loadings for {len(missing_dates)} dates", str(w[-1].message))

            for date in missing_dates:
                self.assertNotIn(date, exposures.index)
                self.assertNotIn(date, perf_attrib_data.index)

            # perf attrib should work if factor_returns already missing dates
            exposures, perf_attrib_data = perf_attrib(
                returns, positions, factor_returns.drop(pd.DatetimeIndex(missing_dates)), factor_loadings_missing_dates
            )

            self.assertEqual(len(w), 3)
            self.assertIn(f"Could not find factor loadings for {len(missing_dates)} dates", str(w[-1].message))

            for date in missing_dates:
                self.assertNotIn(date, exposures.index)
                self.assertNotIn(date, perf_attrib_data.index)

            # test missing stocks and dates
            factor_loadings_missing_both = factor_loadings_missing_dates.drop("TLT", level="ticker")

            exposures, perf_attrib_data = perf_attrib(returns, positions, factor_returns, factor_loadings_missing_both)

            self.assertEqual(len(w), 5)
            self.assertIn("The following assets were missing factor loadings: ['TLT']", str(w[-2].message))
            self.assertIn("Ratio of assets missing: 0.333", str(w[-2].message))

            self.assertIn(f"Could not find factor loadings for {len(missing_dates)} dates", str(w[-1].message))
            for date in missing_dates:
                self.assertNotIn(date, exposures.index)
                self.assertNotIn(date, perf_attrib_data.index)

            # raise exception if all stocks are filtered out
            empty_factor_loadings = factor_loadings.drop(["AAPL", "TLT", "XOM"], level="ticker")

            with self.assertRaisesRegex(ValueError, "No factor loadings were available"):
                exposures, perf_attrib_data = perf_attrib(returns, positions, factor_returns, empty_factor_loadings)

    def test_high_turnover_warning(self):
        """Test that high turnover triggers a warning."""
        (returns, positions, factor_returns, factor_loadings) = generate_toy_risk_model_output()

        # Mock the positions data to turn over the whole portfolio from
        # one asset to another every day (cycling every 3 days).
        positions.iloc[::3, :] = [100.0, 0.0, 0.0, 0.0]
        positions.iloc[1::3, :] = [0.0, 100.0, 0.0, 0.0]
        positions.iloc[2::3, :] = [0.0, 0.0, 100.0, 0.0]

        transactions = mock_transactions_from_positions(positions)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", UserWarning)

            perf_attrib(returns, positions, factor_returns, factor_loadings, transactions=transactions)

        self.assertEqual(len(w), 1)
        self.assertIn(
            "This algorithm has relatively high turnover of its positions.",
            str(w[-1].message),
        )
