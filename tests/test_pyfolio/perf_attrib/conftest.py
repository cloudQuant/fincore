"""Shared fixtures and test utilities for performance attribution tests."""

import os
import numpy as np
import pandas as pd

from fincore.empyrical import Empyrical


def _empyrical_compat_perf_attrib_result(index, columns, data):
    """Create expected performance attribution result DataFrame."""
    return pd.DataFrame(index=index, columns=columns, data=data)


def generate_toy_risk_model_output(start_date="2017-01-01", periods=10, num_styles=2):
    """
    Generate toy risk model output.

    Parameters
    ----------
    start_date : str
        date to start generating toy data
    periods : int
        number of days for which to generate toy data

    Returns
    -------
    tuple of (returns, factor_returns, positions, factor_loadings)
    """
    dts = pd.date_range(start_date, periods=periods)
    np.random.seed(123)
    tickers = ["AAPL", "TLT", "XOM"]
    styles = [f"factor{i}" for i in range(num_styles)]

    returns = pd.Series(index=dts, data=np.random.randn(periods)) / 100

    factor_returns = pd.DataFrame(columns=styles, index=dts, data=np.random.randn(periods, len(styles))) / 100

    arrays = [dts, tickers]
    index = pd.MultiIndex.from_product(arrays, names=["dt", "ticker"])

    positions = pd.DataFrame(columns=tickers, index=dts, data=np.random.randint(100, size=(periods, len(tickers))))
    positions["cash"] = np.zeros(periods)

    factor_loadings = pd.DataFrame(
        columns=styles, index=index, data=np.random.randn(periods * len(tickers), len(styles))
    )

    return returns, positions, factor_returns, factor_loadings


def mock_transactions_from_positions(positions):
    """Mock transactions from positions DataFrame."""
    # Compute the day-to-day diff of the positions frame, then collapse
    # that into a frame with one row per day per asset.
    transactions = (
        pd.melt(
            positions.diff().dropna().reset_index(),
            id_vars=["index"],
            var_name="symbol",
            value_name="amount",
        )
        .sort_values(["index", "symbol"])
        .set_index("index")
    )

    # Filter out positions that didn't actually change.
    transactions = transactions[transactions.amount != 0]

    # Tack on a price column.
    transactions["price"] = 100.0

    return transactions


class PerfAttribTestLocation:
    """Mixin to provide test data location."""

    @property
    def __location__(self):
        """Get the test data directory location."""
        # The test data is in tests/test_data/
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test_data")


# Export functions for tests
perf_attrib = Empyrical.perf_attrib
create_perf_attrib_stats = Empyrical.create_perf_attrib_stats
_cumulative_returns_less_costs = Empyrical._cumulative_returns_less_costs
