"""Tests for fincore.tearsheets.round_trips plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from fincore.tearsheets import round_trips as trt


@pytest.fixture(autouse=True)
def _mpl_cleanup():
    yield
    plt.close("all")


def test_plot_round_trip_lifetimes_with_default_ax():
    """Test plot_round_trip_lifetimes when ax=None (line 35)."""
    # Create sample round trips data
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    round_trips = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL", "AAPL", "MSFT"],
            "open_dt": [dates[0], dates[1], dates[2], dates[0], dates[1]],
            "close_dt": [dates[1], dates[2], dates[3], dates[2], dates[3]],
            "long": [True, False, True, True, False],
            "pnl": [100.0, -50.0, 75.0, 200.0, -25.0],
        }
    )

    # Test with ax=None (line 35)
    ax = trt.plot_round_trip_lifetimes(round_trips, ax=None)
    assert ax is not None
    plt.close()


def test_plot_round_trip_lifetimes_with_custom_ax():
    """Test plot_round_trip_lifetimes with custom ax."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    round_trips = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL", "AAPL", "MSFT"],
            "open_dt": [dates[0], dates[1], dates[2], dates[0], dates[1]],
            "close_dt": [dates[1], dates[2], dates[3], dates[2], dates[3]],
            "long": [True, False, True, True, False],
            "pnl": [100.0, -50.0, 75.0, 200.0, -25.0],
        }
    )

    _, ax = plt.subplots()
    result = trt.plot_round_trip_lifetimes(round_trips, ax=ax)
    assert result is ax
    plt.close()


def test_plot_prob_profit_trade_with_default_ax():
    """Test plot_prob_profit_trade when ax=None (line 97)."""
    # Create sample round trips data
    round_trips = pd.DataFrame(
        {
            "pnl": [100, -50, 75, -25, 150, -30, 50, -10, 80, 20],
        }
    )

    # Test with ax=None (line 97)
    ax = trt.plot_prob_profit_trade(round_trips, ax=None)
    assert ax is not None
    plt.close()


def test_plot_prob_profit_trade_with_custom_ax():
    """Test plot_prob_profit_trade with custom ax."""
    round_trips = pd.DataFrame(
        {
            "pnl": [100, -50, 75, -25, 150],
        }
    )

    _, ax = plt.subplots()
    result = trt.plot_prob_profit_trade(round_trips, ax=ax)
    assert result is ax
    plt.close()
