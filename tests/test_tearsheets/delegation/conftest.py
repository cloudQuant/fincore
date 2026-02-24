"""Shared fixtures and test utilities for tearsheet delegation tests."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


def create_test_index(periods=6, freq="B", tz="UTC"):
    """Create a standard test DatetimeIndex."""
    return pd.date_range("2024-01-01", periods=periods, freq=freq, tz=tz)


def create_test_returns(periods=6, index=None):
    """Create a standard test returns Series."""
    if index is None:
        index = create_test_index(periods)
    return pd.Series(np.linspace(0.001, -0.001, len(index)), index=index, name="r")


def create_test_positions(periods=6, index=None):
    """Create a standard test positions DataFrame."""
    if index is None:
        index = create_test_index(periods)
    return pd.DataFrame(
        {"AAA": [10] * periods, "cash": [100] * periods},
        index=index,
    )


def create_test_transactions(index=None):
    """Create a standard test transactions DataFrame."""
    if index is None:
        index = create_test_index(1)
    return pd.DataFrame({"amount": [1], "price": [10.0], "symbol": ["AAA"]}, index=[index[0]])


def create_market_data(index=None):
    """Create standard market data dict."""
    if index is None:
        index = create_test_index(1)
    return {
        "price": pd.DataFrame({"AAA": [10.0]}, index=[index[0]]),
        "volume": pd.DataFrame({"AAA": [100]}, index=[index[0]]),
    }


@pytest.fixture
def close_all_plots():
    """Fixture to close all matplotlib plots after a test."""
    yield
    plt.close("all")
