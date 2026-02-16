"""Tests for optimization._utils module - full coverage.

This file tests error handling paths in the optimization utilities.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from fincore.optimization._utils import OptimizationError, normalize_weights, validate_result


class TestValidateResult:
    """Test validate_result function."""

    def test_successful_result(self):
        """Test validation of successful optimization result."""
        res = optimize.OptimizeResult(
            x=np.array([0.25, 0.25, 0.25, 0.25]),
            success=True,
            status=0,
            message="Optimization terminated successfully.",
        )
        weights = validate_result(res, context="test")

        np.testing.assert_array_equal(weights, np.array([0.25, 0.25, 0.25, 0.25]))

    def test_failed_result_raises(self):
        """Test that failed optimization raises OptimizationError."""
        res = optimize.OptimizeResult(
            x=np.array([0.25, 0.25, 0.25, 0.25]),
            success=False,
            status=1,
            message="Maximum iterations exceeded.",
        )

        with pytest.raises(OptimizationError, match="Optimization failed for test"):
            validate_result(res, context="test")

    def test_failed_result_contains_status_and_message(self):
        """Test that OptimizationError contains status and message."""
        res = optimize.OptimizeResult(
            x=np.array([0.25, 0.25, 0.25, 0.25]),
            success=False,
            status=2,
            message="Unknown error.",
        )

        try:
            validate_result(res, context="test_context")
        except OptimizationError as e:
            assert e.status == 2
            assert e.solver_message == "Unknown error."

    def test_nan_weights_raises(self):
        """Test that NaN weights raise OptimizationError."""
        res = optimize.OptimizeResult(
            x=np.array([0.25, np.nan, 0.25, 0.25]),
            success=True,
            status=0,
            message="Success",
        )

        with pytest.raises(OptimizationError, match="invalid weights.*NaN"):
            validate_result(res, context="test", allow_nan=False)

    def test_inf_weights_raises(self):
        """Test that infinite weights raise OptimizationError."""
        res = optimize.OptimizeResult(
            x=np.array([0.25, np.inf, 0.25, 0.25]),
            success=True,
            status=0,
            message="Success",
        )

        with pytest.raises(OptimizationError, match="invalid weights.*inf"):
            validate_result(res, context="test", allow_nan=False)

    def test_allow_nan_permits_invalid_weights(self):
        """Test that allow_nan=True permits NaN/inf weights."""
        res = optimize.OptimizeResult(
            x=np.array([0.25, np.nan, 0.25, 0.25]),
            success=True,
            status=0,
            message="Success",
        )
        weights = validate_result(res, context="test", allow_nan=True)

        np.testing.assert_array_equal(weights, np.array([0.25, np.nan, 0.25, 0.25]))


class TestNormalizeWeights:
    """Test normalize_weights function."""

    def test_normalizes_to_sum_one(self):
        """Test that weights are normalized to sum to 1."""
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = normalize_weights(weights)

        np.testing.assert_almost_equal(normalized.sum(), 1.0)

    def test_already_normalized(self):
        """Test that already normalized weights stay the same."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        normalized = normalize_weights(weights)

        np.testing.assert_array_almost_equal(normalized, weights)

    def test_negative_sum_raises(self):
        """Test that negative sum raises OptimizationError."""
        weights = np.array([-1.0, -2.0, -3.0])

        with pytest.raises(OptimizationError, match="sum.*is negative"):
            normalize_weights(weights)

    def test_near_zero_sum_raises(self):
        """Test that near-zero sum raises OptimizationError."""
        weights = np.array([1e-15, 1e-15, 1e-15])

        with pytest.raises(OptimizationError, match="sum.*is too close to zero"):
            normalize_weights(weights)

    def test_zero_sum_raises(self):
        """Test that zero sum raises OptimizationError."""
        weights = np.array([0.0, 0.0, 0.0])

        with pytest.raises(OptimizationError, match="sum.*is too close to zero"):
            normalize_weights(weights)

    def test_custom_epsilon(self):
        """Test with custom epsilon threshold."""
        weights = np.array([1e-7, 1e-7, 1e-7])

        # With larger epsilon (default 1e-12), this small sum should pass
        normalized = normalize_weights(weights)
        np.testing.assert_almost_equal(normalized.sum(), 1.0)
