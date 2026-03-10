"""Tests for plot_daily_volume and plot_daily_turnover_hist functions.

Part of test_transactions_plotting_full_coverage.py split - Volume tests with P2 markers.
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
class TestPlotDailyVolume:
    """Test plot_daily_volume function."""

    def test_plot_daily_volume_with_ax_none(self, sample_data, empyrical_instance):
        """Test plot_daily_volume without providing ax."""
        from fincore.tearsheets.transactions import plot_daily_volume

        returns, positions, transactions = sample_data
        ax = plot_daily_volume(empyrical_instance, returns, transactions)

        assert ax is not None
        assert ax.get_title() == "Daily trading volume"
        assert ax.get_ylabel() == "Amount of shares traded"

    def test_plot_daily_volume_with_custom_ax(self, sample_data, empyrical_instance):
        """Test plot_daily_volume with custom ax."""
        import matplotlib.pyplot as plt

        from fincore.tearsheets.transactions import plot_daily_volume

        returns, positions, transactions = sample_data
        fig, custom_ax = plt.subplots()

        ax = plot_daily_volume(empyrical_instance, returns, transactions, ax=custom_ax)

        assert ax is custom_ax


@pytest.mark.p2
class TestPlotDailyTurnoverHist:
    """Test plot_daily_turnover_hist function."""

    def test_plot_daily_turnover_hist_with_ax_none(self, sample_data, empyrical_instance):
        """Test plot_daily_turnover_hist without providing ax."""
        from fincore.tearsheets.transactions import plot_daily_turnover_hist

        returns, positions, transactions = sample_data
        ax = plot_daily_turnover_hist(empyrical_instance, transactions, positions)

        assert ax is not None
        assert "turnover" in ax.get_title().lower()

    def test_plot_daily_turnover_hist_with_custom_ax(self, sample_data, empyrical_instance):
        """Test plot_daily_turnover_hist with custom ax."""
        import matplotlib.pyplot as plt

        from fincore.tearsheets.transactions import plot_daily_turnover_hist

        returns, positions, transactions = sample_data
        fig, custom_ax = plt.subplots()

        ax = plot_daily_turnover_hist(empyrical_instance, transactions, positions, ax=custom_ax)

        assert ax is custom_ax
