"""Shared fixtures for stats tests.

This conftest.py contains common test data used across multiple stats test modules.
Fixtures defined here are automatically available to all tests in this directory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.constants import DAILY, MONTHLY, QUARTERLY, WEEKLY, YEARLY

# Random seed for reproducibility
rand = np.random.RandomState(1337)

# Pandas frequency alias compatibility
try:
    pd.date_range("2000-1-1", periods=1, freq="ME")
    MONTH_FREQ = "ME"
    YEAR_FREQ = "YE"
except ValueError:
    MONTH_FREQ = "M"
    YEAR_FREQ = "A"


# ========================================================================
# Common Return Series Fixtures
# ========================================================================

@pytest.fixture
def simple_benchmark():
    """Simple benchmark series with no drawdown."""
    return pd.Series(
        np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )


@pytest.fixture
def positive_returns():
    """All positive returns with small variance."""
    return pd.Series(
        np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )


@pytest.fixture
def negative_returns():
    """Series with negative returns."""
    return pd.Series(
        np.array([0.0, -6.0, -7.0, -1.0, -9.0, -2.0, -6.0, -8.0, -5.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )


@pytest.fixture
def all_negative_returns():
    """Series with all negative returns."""
    return pd.Series(
        np.array([-2.0, -6.0, -7.0, -1.0, -9.0, -2.0, -6.0, -8.0, -5.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )


@pytest.fixture
def mixed_returns():
    """Series with positive and negative returns including max drawdown."""
    return pd.Series(
        np.array([np.nan, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )


@pytest.fixture
def flat_line_1():
    """Flat line returns (constant 1%)."""
    return pd.Series(
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="D"),
    )


@pytest.fixture
def weekly_returns():
    """Weekly returns series."""
    return pd.Series(
        np.array([0.0, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq="W"),
    )


@pytest.fixture
def monthly_returns():
    """Monthly returns series."""
    return pd.Series(
        np.array([0.0, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
        index=pd.date_range("2000-1-30", periods=9, freq=MONTH_FREQ),
    )


@pytest.fixture
def empty_returns():
    """Empty returns series."""
    return pd.Series(np.array([]) / 100, index=pd.date_range("2000-1-30", periods=0, freq="D"))


@pytest.fixture
def one_return():
    """Single return value."""
    return pd.Series(np.array([1.0]) / 100, index=pd.date_range("2000-1-30", periods=1, freq="D"))


# ========================================================================
# Extended Data Series Fixtures
# ========================================================================

@pytest.fixture
def noise():
    """Random noise series (1000 observations)."""
    return pd.Series(
        rand.normal(0, 0.001, 1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )


@pytest.fixture
def noise_uniform():
    """Uniform random noise series."""
    return pd.Series(
        rand.uniform(-0.01, 0.01, 1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )


@pytest.fixture
def inv_noise(noise):
    """Inverted noise series."""
    return noise.multiply(-1)


@pytest.fixture
def flat_line_0():
    """Flat line at zero."""
    return pd.Series(
        np.linspace(0, 0, num=1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )


@pytest.fixture
def flat_line_1_tz():
    """Flat line at 1% with timezone."""
    return pd.Series(
        np.linspace(0.01, 0.01, num=1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )


@pytest.fixture
def pos_line():
    """Positive sloping line from 0 to 1."""
    return pd.Series(
        np.linspace(0, 1, num=1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )


@pytest.fixture
def neg_line():
    """Negative sloping line from 0 to -1."""
    return pd.Series(
        np.linspace(0, -1, num=1000),
        index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC")
    )


@pytest.fixture
def sparse_noise(noise):
    """Noise series with some NaN values sprinkled in."""
    replace_nan = rand.choice(noise.index.tolist(), rand.randint(1, 10))
    return noise.replace(replace_nan, np.nan)


@pytest.fixture
def sparse_flat_line_1_tz(flat_line_1_tz):
    """Flat line with some NaN values."""
    replace_nan = rand.choice(flat_line_1_tz.index.tolist(), rand.randint(1, 10))
    return flat_line_1_tz.replace(replace_nan, np.nan)


# ========================================================================
# DataFrame Fixtures
# ========================================================================

@pytest.fixture
def df_index_simple():
    """Simple daily date range index."""
    return pd.date_range("2000-1-30", periods=8, freq="D")


@pytest.fixture
def df_index_week():
    """Weekly date range index."""
    return pd.date_range("2000-1-30", periods=8, freq="W")


@pytest.fixture
def df_index_month():
    """Monthly date range index."""
    return pd.date_range("2000-1-30", periods=8, freq=MONTH_FREQ)


@pytest.fixture
def df_simple(df_index_simple):
    """Simple DataFrame with two columns."""
    one = [-0.00171614, 0.01322056, 0.03063862, -0.01422057, -0.00489779, 0.01268925, -0.03357711, 0.01797036]
    two = [0.01846232, 0.00793951, -0.01448395, 0.00422537, -0.00339611, 0.03756813, 0.0151531, 0.03549769]
    return pd.DataFrame(
        {"one": pd.Series(one, index=df_index_simple), "two": pd.Series(two, index=df_index_simple)}
    )


@pytest.fixture
def df_week(df_index_week):
    """DataFrame with weekly index."""
    one = [-0.00171614, 0.01322056, 0.03063862, -0.01422057, -0.00489779, 0.01268925, -0.03357711, 0.01797036]
    two = [0.01846232, 0.00793951, -0.01448395, 0.00422537, -0.00339611, 0.03756813, 0.0151531, 0.03549769]
    return pd.DataFrame({"one": pd.Series(one, index=df_index_week), "two": pd.Series(two, index=df_index_week)})


@pytest.fixture
def df_month(df_index_month):
    """DataFrame with monthly index."""
    one = [-0.00171614, 0.01322056, 0.03063862, -0.01422057, -0.00489779, 0.01268925, -0.03357711, 0.01797036]
    two = [0.01846232, 0.00793951, -0.01448395, 0.00422537, -0.00339611, 0.03756813, 0.0151531, 0.03549769]
    return pd.DataFrame({"one": pd.Series(one, index=df_index_month), "two": pd.Series(two, index=df_index_month)})
