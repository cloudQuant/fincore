"""Tests for Hill estimator functionality.

Tests the Hill estimator for tail index estimation.
Split from test_evt_full_coverage.py for maintainability.
"""

from __future__ import annotations

import numpy as np
import pytest

from fincore.risk.evt import hill_estimator


class TestHillEstimator:
    """Test Hill estimator functionality."""

    def test_upper_tail(self, heavy_tailed_data):
        """Test Hill estimator for upper tail."""
        xi, excesses = hill_estimator(heavy_tailed_data, tail="upper")

        assert isinstance(xi, float)
        assert xi > 0  # Heavy-tailed
        assert len(excesses) > 0

    def test_lower_tail(self, heavy_tailed_data):
        """Test Hill estimator for lower tail."""
        xi, excesses = hill_estimator(heavy_tailed_data, tail="lower")

        assert isinstance(xi, float)
        assert xi > 0  # Heavy-tailed
        assert len(excesses) > 0

    def test_custom_threshold(self, heavy_tailed_data):
        """Test Hill estimator with custom threshold."""
        xi, excesses = hill_estimator(heavy_tailed_data, threshold=0.1, tail="upper")

        assert isinstance(xi, float)
        assert len(excesses) > 0

    def test_invalid_tail(self, heavy_tailed_data):
        """Test that invalid tail raises ValueError."""
        with pytest.raises(ValueError, match="tail must be"):
            hill_estimator(heavy_tailed_data, tail="middle")

    def test_insufficient_exceedances(self):
        """Test that insufficient exceedances raises ValueError."""
        data = np.array([1, 2, 3, 4, 5])  # Too few data points
        with pytest.raises(ValueError, match="Not enough exceedances"):
            hill_estimator(data, threshold=10, tail="upper")
