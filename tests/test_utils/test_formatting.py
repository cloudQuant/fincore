"""Tests for formatting utilities in fincore.utils.common_utils.

Split from test_common_display.py for maintainability.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from fincore.utils import common_utils as cu


@pytest.mark.p2  # Medium: formatting utility tests
class TestFormatters:
    """Test number and percentage formatting functions."""

    def test_one_dec_places_formatter(self):
        """Test the one_dec_places formatter function."""
        assert cu.one_dec_places(1.234, 0) == "1.2"
        assert cu.one_dec_places(5.678, 0) == "5.7"
        assert cu.one_dec_places(0, 0) == "0.0"

    def test_two_dec_places_formatter(self):
        """Test the two_dec_places formatter function."""
        assert cu.two_dec_places(1.234, 0) == "1.23"
        assert cu.two_dec_places(5.678, 0) == "5.68"
        assert cu.two_dec_places(0, 0) == "0.00"

    def test_percentage_formatter(self):
        """Test the percentage formatter function."""
        assert cu.percentage(50, 0) == "50%"
        assert cu.percentage(100.5, 0) == "100%"
        assert cu.percentage(0, 0) == "0%"


@pytest.mark.p2  # Medium: asset formatting tests
class TestAssetFormatting:
    """Test asset formatting utilities."""

    def test_fallback_html_returns_input(self):
        """Test _fallback_html returns input unchanged."""
        assert cu._fallback_html("<b>x</b>") == "<b>x</b>"

    def test_format_asset_returns_input_when_zipline_missing(self):
        """Test format_asset returns input when zipline is not available."""
        assert cu.format_asset("AAPL") == "AAPL"

    def test_format_asset_non_zipline_asset(self):
        """Test format_asset with non-zipline asset types."""
        assert cu.format_asset("AAPL") == "AAPL"
        assert cu.format_asset(123) == 123
        assert cu.format_asset(None) is None

    def test_format_asset_with_zipline_asset_class_via_stub_module(self, monkeypatch):
        """Test format_asset extracts symbol from zipline Asset class."""
        class Asset:
            def __init__(self, symbol: str) -> None:
                self.symbol = symbol

        zipline_assets_mod = SimpleNamespace(Asset=Asset)
        zipline_mod = SimpleNamespace(assets=zipline_assets_mod)
        monkeypatch.setitem(sys.modules, "zipline", zipline_mod)
        monkeypatch.setitem(sys.modules, "zipline.assets", zipline_assets_mod)

        assert cu.format_asset(Asset("AAPL")) == "AAPL"

    def test_format_asset_with_zipline_available_but_not_asset(self, monkeypatch):
        """Test format_asset when zipline is available but asset is not an Asset."""
        class Asset:
            def __init__(self, symbol: str) -> None:
                self.symbol = symbol

        zipline_assets_mod = SimpleNamespace(Asset=Asset)
        zipline_mod = SimpleNamespace(assets=zipline_assets_mod)
        monkeypatch.setitem(sys.modules, "zipline", zipline_mod)
        monkeypatch.setitem(sys.modules, "zipline.assets", zipline_assets_mod)

        assert cu.format_asset("AAPL") == "AAPL"
        assert cu.format_asset(123) == 123
        assert cu.format_asset(None) is None
        assert cu.format_asset(Asset("GOOGL")) == "GOOGL"
