"""Tests for plot_turnover function.

Part of test_transactions_plotting_full_coverage.py split - Turnover tests with P2 markers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _mpl_cleanup():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Create sample returns, positions, and transactions data."""
    np.random.seed(42)
    idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="America/New_York")

    returns = pd.Series(np.random.randn(100) * 0.01, index=idx, name="returns")

    positions = pd.DataFrame(
        {
            "AAPL": np.random.randn(100) * 1000 + 10000,
            "MSFT": np.random.randn(100) * 1000 + 10000,
            "cash": np.random.randn(100) * 1000 + 50000,
        },
        index=idx,
    )

    txn_idx = pd.date_range("2020-01-01", periods=50, freq="B", tz="America/New_York")
    transactions = pd.DataFrame(
        {
            "symbol": np.random.choice(["AAPL", "MSFT"], 50),
            "amount": np.random.randint(-100, 100, 50),
            "price": np.random.uniform(50, 200, 50),
        },
        index=txn_idx,
    )

    return returns, positions, transactions


@pytest.fixture
def empyrical_instance():
    """Create a mock Empyrical instance with needed methods."""
    from fincore.empyrical import Empyrical

    np.random.seed(42)
    idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="America/New_York")
    returns = pd.Series(np.random.randn(100) * 0.01, index=idx)
    return Empyrical(returns=returns)


@pytest.mark.p2
class TestPlotTurnover:
    """Test plot_turnover function."""

    def test_plot_turnover_with_ax_none(self, sample_data, empyrical_instance):
        """Test plot_turnover without providing ax."""
        from fincore.tearsheets.transactions import plot_turnover

        returns, positions, transactions = sample_data
        ax = plot_turnover(empyrical_instance, returns, transactions, positions)

        assert ax is not None
        assert ax.get_title() == "Daily turnover"
        assert ax.get_ylabel() == "Turnover"

    def test_plot_turnover_with_custom_ax(self, sample_data, empyrical_instance):
        """Test plot_turnover with custom ax."""
        import matplotlib.pyplot as plt

        from fincore.tearsheets.transactions import plot_turnover

        returns, positions, transactions = sample_data
        _fig, custom_ax = plt.subplots()

        ax = plot_turnover(empyrical_instance, returns, transactions, positions, ax=custom_ax)

        assert ax is custom_ax

    def test_plot_turnover_with_custom_legend_loc(self, sample_data, empyrical_instance):
        """Test plot_turnover with custom legend location."""
        from fincore.tearsheets.transactions import plot_turnover

        returns, positions, transactions = sample_data
        ax = plot_turnover(
            empyrical_instance,
            returns,
            transactions,
            positions,
            legend_loc="upper right",
        )

        assert ax is not None
