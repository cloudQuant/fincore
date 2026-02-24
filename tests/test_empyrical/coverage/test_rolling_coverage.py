"""Tests for Empyrical rolling metrics with factor_returns.

This file tests rolling alpha/beta metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.empyrical import Empyrical


class TestEmpyricalRollingMetrics:
    """Test rolling metrics with factor_returns."""

    def test_roll_alpha_with_factor_returns(self):
        """Test roll_alpha with factor_returns."""
        returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=pd.date_range("2020-01-01", periods=300, freq="D"),
        )
        factor_returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=returns.index,
        )

        emp = Empyrical(returns=returns, factor_returns=factor_returns)

        result = emp.roll_alpha(window=50)

        assert isinstance(result, (pd.Series, pd.DataFrame))

    def test_roll_beta_with_factor_returns(self):
        """Test roll_beta with factor_returns."""
        returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=pd.date_range("2020-01-01", periods=300, freq="D"),
        )
        factor_returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=returns.index,
        )

        emp = Empyrical(returns=returns, factor_returns=factor_returns)

        result = emp.roll_beta(window=50)

        assert isinstance(result, (pd.Series, pd.DataFrame))

    def test_roll_alpha_instance_with_stored_factor(self):
        """Test roll_alpha using stored factor_returns."""
        returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=pd.date_range("2020-01-01", periods=300, freq="D"),
        )
        factor_returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=returns.index,
        )

        emp = Empyrical(returns=returns, factor_returns=factor_returns)

        # Call without passing factor_returns - should use stored
        result = emp.roll_alpha(window=50)

        assert isinstance(result, (pd.Series, pd.DataFrame))

    def test_roll_beta_instance_with_stored_factor(self):
        """Test roll_beta using stored factor_returns."""
        returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=pd.date_range("2020-01-01", periods=300, freq="D"),
        )
        factor_returns = pd.Series(
            np.random.randn(300) * 0.01,
            index=returns.index,
        )

        emp = Empyrical(returns=returns, factor_returns=factor_returns)

        # Call without passing factor_returns - should use stored
        result = emp.roll_beta(window=50)

        assert isinstance(result, (pd.Series, pd.DataFrame))
