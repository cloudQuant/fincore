"""Tests for plot_txn_time_hist function.

Part of test_transactions_plotting_full_coverage.py split - Time hist tests with P2 markers.
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
    """Create timezone-aware sample data for plot_txn_time_hist."""
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


@pytest.mark.p2
class TestPlotTxnTimeHist:
    """Test plot_txn_time_hist function."""

    def test_plot_txn_time_hist_with_ax_none(self, sample_data):
        """Test plot_txn_time_hist without providing ax."""
        from fincore.tearsheets.transactions import plot_txn_time_hist

        returns, positions, transactions = sample_data
        ax = plot_txn_time_hist(transactions)

        assert ax is not None
        assert "time" in ax.get_title().lower()

    def test_plot_txn_time_hist_with_custom_ax(self, sample_data):
        """Test plot_txn_time_hist with custom ax."""
        import matplotlib.pyplot as plt

        from fincore.tearsheets.transactions import plot_txn_time_hist

        returns, positions, transactions = sample_data
        fig, custom_ax = plt.subplots()

        ax = plot_txn_time_hist(transactions, ax=custom_ax)

        assert ax is custom_ax

    def test_plot_txn_time_hist_with_custom_bin_minutes(self, sample_data):
        """Test plot_txn_time_hist with custom bin_minutes."""
        from fincore.tearsheets.transactions import plot_txn_time_hist

        returns, positions, transactions = sample_data
        ax = plot_txn_time_hist(transactions, bin_minutes=10)

        assert ax is not None

    def test_plot_txn_time_hist_with_custom_tz(self, sample_data):
        """Test plot_txn_time_hist with custom timezone."""
        from fincore.tearsheets.transactions import plot_txn_time_hist

        returns, positions, transactions = sample_data
        ax = plot_txn_time_hist(transactions, tz="UTC")

        assert ax is not None
