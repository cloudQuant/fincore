"""Tests for Value at Risk (VaR) calculations.

This module tests value at risk and conditional value at risk calculations.

Split from test_stats.py to improve maintainability.

Priority Markers:
- P0: Core value_at_risk and conditional_value_at_risk tests
- P1: Edge cases and validation
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from unittest import TestCase

from fincore import empyrical
from fincore.metrics import risk as risk_module

DECIMAL_PLACES = 8

rand = np.random.RandomState(1337)


class BaseTestCase(TestCase):
    """Base test case for VaR tests."""

    pass


class TestValueAtRisk(BaseTestCase):
    """Tests for Value at Risk and Conditional Value at Risk."""

    @property
    def empyrical(self):
        """Get empyrical module instance."""
        return empyrical

    # ========================================================================
    # Value at Risk Tests
    # ========================================================================

    @pytest.mark.p0  # Critical: core risk metric
    def test_value_at_risk(self):
        """Test value at risk calculation at various cutoffs."""
        returns = [1.0, 2.0]
        assert_almost_equal(risk_module.value_at_risk(returns, cutoff=0.0), 1.0)
        assert_almost_equal(risk_module.value_at_risk(returns, cutoff=0.3), 1.3)
        assert_almost_equal(risk_module.value_at_risk(returns, cutoff=1.0), 2.0)

        returns = [1, 81, 82, 83, 84, 85]
        assert_almost_equal(risk_module.value_at_risk(returns, cutoff=0.1), 41)
        assert_almost_equal(risk_module.value_at_risk(returns, cutoff=0.2), 81)
        assert_almost_equal(risk_module.value_at_risk(returns, cutoff=0.3), 81.5)

        # Test a returns stream of 21 data points at different cutoffs.
        returns = rand.normal(0, 0.02, 21)
        for cutoff in (0, 0.0499, 0.05, 0.20, 0.999, 1):
            assert_almost_equal(
                risk_module.value_at_risk(returns, cutoff),
                np.percentile(returns, cutoff * 100),
            )

    # ========================================================================
    # Conditional Value at Risk Tests
    # ========================================================================

    @pytest.mark.p0  # Critical: core risk metric
    def test_conditional_value_at_risk(self):
        """Test conditional value at risk (CVaR) calculation."""
        # A single-valued array will always just have a CVaR of its only value.
        returns = rand.normal(0, 0.02, 1)
        expected_cvar = returns[0]
        assert_almost_equal(
            risk_module.conditional_value_at_risk(returns, cutoff=0),
            expected_cvar,
        )
        assert_almost_equal(
            risk_module.conditional_value_at_risk(returns, cutoff=1),
            expected_cvar,
        )

        # Test a returns stream of 21 data points at different cutoffs.
        returns = rand.normal(0, 0.02, 21)

        for cutoff in (0, 0.0499, 0.05, 0.20, 0.999, 1):
            # Find the VaR based on our cutoff, then take the average of all
            # values at or below the VaR.
            var = risk_module.value_at_risk(returns, cutoff)
            expected_cvar = np.mean(returns[returns <= var])

            assert_almost_equal(
                risk_module.conditional_value_at_risk(returns, cutoff),
                expected_cvar,
            )


# ========================================================================
# Module-level reference
# ========================================================================
EMPYRICAL_MODULE = empyrical
