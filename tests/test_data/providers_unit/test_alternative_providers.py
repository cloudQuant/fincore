"""Tests for alternative data providers.

Tests for AlphaVantage, Tushare, and AkShare providers.
Split from test_providers_unit.py for maintainability.
"""

from __future__ import annotations

import pytest


class TestAlphaVantageProviderUnit:
    """Unit tests for Alpha Vantage provider (no network)."""

    def test_provider_creation(self):
        """Test provider can be instantiated with API key."""
        from fincore.data.providers import AlphaVantageProvider

        provider = AlphaVantageProvider(api_key="test_key")
        assert provider is not None
        assert provider.api_key == "test_key"

    def test_provider_requires_api_key(self):
        """Test provider raises error without API key."""
        from fincore.data.providers import AlphaVantageProvider

        with pytest.raises(TypeError):
            AlphaVantageProvider()


class TestTushareProviderUnit:
    """Unit tests for Tushare provider (no network)."""

    def test_provider_creation(self):
        """Test provider can be instantiated with token."""
        try:
            from fincore.data.providers import TushareProvider

            provider = TushareProvider(token="test_token")
            assert provider is not None
        except ImportError:
            pytest.skip("tushare not installed")

    def test_provider_requires_token(self):
        """Test provider raises error without token."""
        try:
            from fincore.data.providers import TushareProvider

            with pytest.raises(TypeError):
                TushareProvider()
        except ImportError:
            pytest.skip("tushare not installed")


class TestAkShareProviderUnit:
    """Unit tests for AkShare provider (no network)."""

    def test_provider_creation(self):
        """Test provider can be instantiated."""
        try:
            from fincore.data.providers import AkShareProvider

            provider = AkShareProvider()
            assert provider is not None
        except ImportError:
            pytest.skip("akshare not installed")
