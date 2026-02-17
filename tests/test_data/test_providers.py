"""Tests for data provider integrations.

This file contains non-networked tests that verify the interface
and basic functionality of data providers. For integration tests
with actual network calls, see test_providers_integration.py.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest


class TestDataProviderInterface:
    """Tests for DataProvider abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Test that DataProvider cannot be instantiated directly."""
        from fincore.data.providers import DataProvider

        with pytest.raises(TypeError):
            DataProvider()

    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods raise TypeError when called on base class."""
        from fincore.data.providers import DataProvider

        # Create a minimal concrete subclass that doesn't implement abstract methods
        class IncompleteProvider(DataProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_concrete_provider_implements_fetch(self):
        """Test that concrete providers implement the fetch method."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()
        # Verify fetch method exists and is callable
        assert hasattr(provider, "fetch")
        assert callable(provider.fetch)

    def test_concrete_provider_implements_fetch_multiple(self):
        """Test that concrete providers implement the fetch_multiple method."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()
        # Verify fetch_multiple method exists and is callable
        assert hasattr(provider, "fetch_multiple")
        assert callable(provider.fetch_multiple)

    def test_concrete_provider_implements_get_info(self):
        """Test that concrete providers implement the get_info method."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()
        # Verify get_info method exists and is callable
        assert hasattr(provider, "get_info")
        assert callable(provider.get_info)

    def test_validate_dates(self):
        """Test date validation."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        # String dates
        start, end = provider.validate_dates("2020-01-01", "2020-12-31")
        assert start == pd.to_datetime("2020-01-01")
        assert end == pd.to_datetime("2020-12-31")

        # Datetime objects
        dt_start = datetime(2020, 1, 1)
        dt_end = datetime(2020, 12, 31)
        start, end = provider.validate_dates(dt_start, dt_end)
        assert start == pd.to_datetime(dt_start)
        assert end == pd.to_datetime(dt_end)

        # Invalid dates
        with pytest.raises(ValueError):
            provider.validate_dates("2020-12-31", "2020-01-01")


class TestYahooFinanceProvider:
    """Tests for Yahoo Finance provider (no network required)."""

    def test_provider_creation(self):
        """Test provider can be instantiated."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()
        assert provider is not None

    def test_to_returns(self):
        """Test converting prices to returns."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        # Create sample price data
        import pandas as pd

        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        prices = pd.DataFrame(
            {"Close": [100, 102, 101, 103, 105]},
            index=dates,
        )

        returns = provider.to_returns(prices)

        assert isinstance(returns, pd.Series)
        assert not returns.empty
        # Returns should have one less element than prices
        assert len(returns) == len(prices) - 1


class TestAlphaVantageProvider:
    """Tests for Alpha Vantage provider (no network required)."""

    def test_provider_creation(self):
        """Test provider can be instantiated with API key."""
        from fincore.data.providers import AlphaVantageProvider

        provider = AlphaVantageProvider(api_key="test_key")
        assert provider is not None


class TestTushareProvider:
    """Tests for Tushare provider (no network required)."""

    def test_provider_creation(self):
        """Test provider can be instantiated with token."""
        try:
            from fincore.data.providers import TushareProvider

            provider = TushareProvider(token="test_token")
            assert provider is not None
        except ImportError:
            pytest.skip("tushare not installed")


class TestAkShareProvider:
    """Tests for AkShare provider (no network required)."""

    def test_provider_creation(self):
        """Test provider can be instantiated."""
        try:
            from fincore.data.providers import AkShareProvider

            provider = AkShareProvider()
            assert provider is not None
        except ImportError:
            pytest.skip("akshare not installed")


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_provider_yahoo(self):
        """Test getting Yahoo provider."""
        from fincore.data.providers import YahooFinanceProvider, get_provider

        provider = get_provider("yahoo")
        assert isinstance(provider, YahooFinanceProvider)

    def test_get_provider_yfinance_alias(self):
        """Test yfinance alias works."""
        from fincore.data.providers import YahooFinanceProvider, get_provider

        provider = get_provider("yfinance")
        assert isinstance(provider, YahooFinanceProvider)

    def test_get_provider_invalid(self):
        """Test invalid provider name raises error."""
        from fincore.data.providers import get_provider

        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("invalid_provider")
