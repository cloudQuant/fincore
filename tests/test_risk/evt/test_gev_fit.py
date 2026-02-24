"""Tests for GEV (Generalized Extreme Value) fitting functionality.

Tests GEV fitting for block maxima.
Split from test_evt_full_coverage.py for maintainability.
"""

from __future__ import annotations

import numpy as np
import pytest

from fincore.risk.evt import gev_fit


class TestGEVFit:
    """Test GEV fitting functionality."""

    def test_default_block_size(self, heavy_tailed_data):
        """Test GEV fit with default block size."""
        params = gev_fit(heavy_tailed_data)

        assert "xi" in params
        assert "mu" in params
        assert "sigma" in params
        assert "n_blocks" in params

    def test_custom_block_size(self, heavy_tailed_data):
        """Test GEV fit with custom block size."""
        params = gev_fit(heavy_tailed_data, block_size=100)

        assert params["n_blocks"] == len(heavy_tailed_data) // 100
