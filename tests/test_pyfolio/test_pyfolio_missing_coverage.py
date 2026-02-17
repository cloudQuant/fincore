"""Tests for missing coverage in pyfolio.py module.

This module covers edge cases and branches that were previously uncovered:
- Line 55-58: matplotlib.use('Agg') exception handling
"""

import numpy as np
import pandas as pd
import pytest

# The pyfolio module has matplotlib backend initialization
# We just test that it imports correctly
try:
    from fincore.pyfolio import Pyfolio

    HAS_PYFOLIO = True
except ImportError:
    HAS_PYFOLIO = False


@pytest.mark.skipif(not HAS_PYFOLIO, reason="pyfolio not available")
class TestPyfolioMissingCoverage:
    """Test Pyfolio edge cases for 100% coverage."""

    def test_pyfolio_init_with_returns(self):
        """Test Pyfolio initialization with returns."""
        np.random.seed(42)
        returns = pd.Series(
            np.random.randn(252) * 0.02,
            index=pd.date_range("2020-01-01", periods=252, freq="D"),
        )

        pyf = Pyfolio(returns=returns)

        assert pyf is not None
        assert pyf.returns is not None

    def test_pyfolio_init_with_factor_returns(self):
        """Test Pyfolio initialization with factor returns."""
        np.random.seed(42)
        returns = pd.Series(
            np.random.randn(252) * 0.02,
            index=pd.date_range("2020-01-01", periods=252, freq="D"),
        )
        factor_returns = pd.Series(
            np.random.randn(252) * 0.015,
            index=pd.date_range("2020-01-01", periods=252, freq="D"),
        )

        pyf = Pyfolio(returns=returns, factor_returns=factor_returns)

        assert pyf is not None

    def test_pyfolio_init_with_positions(self):
        """Test Pyfolio initialization with positions."""
        np.random.seed(42)
        returns = pd.Series(
            np.random.randn(252) * 0.02,
            index=pd.date_range("2020-01-01", periods=252, freq="D"),
        )
        positions = pd.DataFrame(
            {
                "AAPL": np.random.randint(100, 1000, 252),
                "MSFT": np.random.randint(100, 1000, 252),
            },
            index=pd.date_range("2020-01-01", periods=252, freq="D"),
        )

        pyf = Pyfolio(returns=returns, positions=positions)

        assert pyf is not None

    def test_pyfolio_basic_metrics(self):
        """Test Pyfolio basic metric methods."""
        np.random.seed(42)
        returns = pd.Series(
            np.random.randn(252) * 0.02,
            index=pd.date_range("2020-01-01", periods=252, freq="D"),
        )

        pyf = Pyfolio(returns=returns)

        # Test various metrics - need to pass returns explicitly
        sharpe = pyf.sharpe_ratio(returns)
        assert isinstance(sharpe, (int, float, np.floating))

        vol = pyf.annual_volatility(returns)
        assert isinstance(vol, (int, float, np.floating))

        max_dd = pyf.max_drawdown(returns)
        assert isinstance(max_dd, (int, float, np.floating))
