"""Tests for tearsheets.transactions plotting functions - full coverage.

This file tests the direct plotting functions from fincore.tearsheets.transactions
for uncovered lines (ax=None default parameter branches).
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

    # Create transactions with same timezone-aware datetime index
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


@pytest.fixture
def sample_data_tz_aware():
    """Create timezone-aware sample data for plot_txn_time_hist."""
    np.random.seed(42)

    # Returns and positions for other tests
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

    # Transactions must be timezone-aware for plot_txn_time_hist
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
        fig, custom_ax = plt.subplots()

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


class TestPlotSlippageSweep:
    """Test plot_slippage_sweep function."""

    def test_plot_slippage_sweep_with_ax_none(self, sample_data, empyrical_instance):
        """Test plot_slippage_sweep without providing ax."""
        from fincore.tearsheets.transactions import plot_slippage_sweep

        returns, positions, transactions = sample_data
        ax = plot_slippage_sweep(empyrical_instance, returns, positions, transactions)

        assert ax is not None
        assert "slippage" in ax.get_title().lower()

    def test_plot_slippage_sweep_with_custom_ax(self, sample_data, empyrical_instance):
        """Test plot_slippage_sweep with custom ax."""
        import matplotlib.pyplot as plt

        from fincore.tearsheets.transactions import plot_slippage_sweep

        returns, positions, transactions = sample_data
        fig, custom_ax = plt.subplots()

        ax = plot_slippage_sweep(empyrical_instance, returns, positions, transactions, ax=custom_ax)

        assert ax is custom_ax

    def test_plot_slippage_sweep_with_custom_params(self, sample_data, empyrical_instance):
        """Test plot_slippage_sweep with custom slippage_params."""
        from fincore.tearsheets.transactions import plot_slippage_sweep

        returns, positions, transactions = sample_data
        custom_params = (5, 10, 15, 20)
        ax = plot_slippage_sweep(empyrical_instance, returns, positions, transactions, slippage_params=custom_params)

        assert ax is not None


class TestPlotSlippageSensitivity:
    """Test plot_slippage_sensitivity function."""

    def test_plot_slippage_sensitivity_with_ax_none(self, sample_data, empyrical_instance):
        """Test plot_slippage_sensitivity without providing ax."""
        from fincore.tearsheets.transactions import plot_slippage_sensitivity

        returns, positions, transactions = sample_data
        ax = plot_slippage_sensitivity(empyrical_instance, returns, positions, transactions)

        assert ax is not None
        assert "slippage" in ax.get_title().lower()
        assert ax.get_ylabel() == "Average annual return"

    def test_plot_slippage_sensitivity_with_custom_ax(self, sample_data, empyrical_instance):
        """Test plot_slippage_sensitivity with custom ax."""
        import matplotlib.pyplot as plt

        from fincore.tearsheets.transactions import plot_slippage_sensitivity

        returns, positions, transactions = sample_data
        fig, custom_ax = plt.subplots()

        ax = plot_slippage_sensitivity(empyrical_instance, returns, positions, transactions, ax=custom_ax)

        assert ax is custom_ax
