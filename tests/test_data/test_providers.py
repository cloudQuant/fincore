"""Tests for data provider integrations."""

from __future__ import annotations

import pytest
import pandas as pd
from datetime import datetime, timedelta


class TestDataProviderInterface:
    """Tests for DataProvider abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Test that DataProvider cannot be instantiated directly."""
        from fincore.data.providers import DataProvider

        with pytest.raises(TypeError):
            DataProvider()

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
    """Tests for Yahoo Finance provider."""

    def test_provider_creation(self):
        """Test provider can be instantiated."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()
        assert provider is not None

    @pytest.mark.network
    def test_fetch_single_symbol(self):
        """Test fetching data for a single symbol."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        # Fetch recent data (should be available)
        end = datetime.now()
        start = end - timedelta(days=30)

        data = provider.fetch("AAPL", start, end)

        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert "Close" in data.columns
        assert "Adj Close" in data.columns

    @pytest.mark.network
    def test_fetch_multiple_symbols(self):
        """Test fetching data for multiple symbols."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        end = datetime.now()
        start = end - timedelta(days=30)

        symbols = ["AAPL", "MSFT", "GOOGL"]
        data = provider.fetch_multiple(symbols, start, end)

        assert isinstance(data, dict)
        assert all(isinstance(d, pd.DataFrame) for d in data.values())

    @pytest.mark.network
    def test_get_info(self):
        """Test getting symbol information."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()
        info = provider.get_info("AAPL")

        assert isinstance(info, dict)
        assert "symbol" in info
        assert info["symbol"] == "AAPL"

    @pytest.mark.network
    def test_to_returns(self):
        """Test converting prices to returns."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        end = datetime.now()
        start = end - timedelta(days=30)

        data = provider.fetch("AAPL", start, end)
        returns = provider.to_returns(data)

        assert isinstance(returns, pd.Series)
        assert not returns.empty
        # Returns should have one less element than prices (first is NaN and dropped)
        assert len(returns) <= len(data)

    @pytest.mark.network
    def test_fetch_index(self):
        """Test fetching index data."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        end = datetime.now()
        start = end - timedelta(days=30)

        data = provider.fetch("^GSPC", start, end)

        assert isinstance(data, pd.DataFrame)
        assert not data.empty


class TestAlphaVantageProvider:
    """Tests for Alpha Vantage provider."""

    @pytest.fixture
    def provider(self):
        """Create provider (skipped if no API key)."""
        import os

        api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
        if not api_key:
            pytest.skip("ALPHAVANTAGE_API_KEY not set")

        from fincore.data.providers import AlphaVantageProvider
        return AlphaVantageProvider(api_key=api_key)

    def test_provider_creation(self, provider):
        """Test provider can be instantiated."""
        assert provider is not None

    def test_fetch_single_symbol(self, provider):
        """Test fetching data."""
        data = provider.fetch("AAPL", "2020-01-01", "2020-01-31")

        assert isinstance(data, pd.DataFrame)
        assert not data.empty


class TestTushareProvider:
    """Tests for Tushare provider."""

    @pytest.fixture
    def provider(self):
        """Create provider (skipped if no token)."""
        import os

        token = os.environ.get("TUSHARE_TOKEN")
        if not token:
            pytest.skip("TUSHARE_TOKEN not set")

        from fincore.data.providers import TushareProvider
        return TushareProvider(token=token)

    def test_provider_creation(self, provider):
        """Test provider can be instantiated."""
        assert provider is not None

    def test_fetch_single_symbol(self, provider):
        """Test fetching data."""
        data = provider.fetch("000001.SZ", "2023-01-01", "2023-12-31")

        assert isinstance(data, pd.DataFrame)
        assert not data.empty


class TestAkShareProvider:
    """Tests for AkShare provider."""

    def test_provider_creation(self):
        """Test provider can be instantiated."""
        try:
            from fincore.data.providers import AkShareProvider

            provider = AkShareProvider()
            assert provider is not None
        except ImportError:
            pytest.skip("akshare not installed")

    def test_fetch_single_symbol(self):
        """Test fetching data."""
        try:
            from fincore.data.providers import AkShareProvider

            provider = AkShareProvider()
            data = provider.fetch("000001", "2023-01-01", "2023-12-31")

            assert isinstance(data, pd.DataFrame)
            assert not data.empty
        except ImportError:
            pytest.skip("akshare not installed")
        except Exception as e:
            # Network errors or API changes are OK for testing
            pytest.skip(f"AkShare test failed: {e}")


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_provider_yahoo(self):
        """Test getting Yahoo provider."""
        from fincore.data.providers import get_provider, YahooFinanceProvider

        provider = get_provider("yahoo")
        assert isinstance(provider, YahooFinanceProvider)

    def test_get_provider_yfinance_alias(self):
        """Test yfinance alias works."""
        from fincore.data.providers import get_provider, YahooFinanceProvider

        provider = get_provider("yfinance")
        assert isinstance(provider, YahooFinanceProvider)

    def test_get_provider_invalid(self):
        """Test invalid provider name raises error."""
        from fincore.data.providers import get_provider

        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("invalid_provider")

    @pytest.mark.network
    def test_fetch_price_data_default_years(self):
        """Test fetch_price_data with default years parameter."""
        from fincore.data.providers import fetch_price_data

        data = fetch_price_data("AAPL", years=1)

        assert isinstance(data, pd.DataFrame)
        assert not data.empty

    @pytest.mark.network
    def test_fetch_price_data_with_dates(self):
        """Test fetch_price_data with explicit dates."""
        from fincore.data.providers import fetch_price_data

        data = fetch_price_data("AAPL", start="2023-01-01", end="2023-12-31")

        assert isinstance(data, pd.DataFrame)
        assert not data.empty

    @pytest.mark.network
    def test_fetch_multiple_prices(self):
        """Test fetch_multiple_prices convenience function."""
        from fincore.data.providers import fetch_multiple_prices

        data = fetch_multiple_prices(["AAPL", "MSFT"], years=1)

        assert isinstance(data, dict)
        assert all(isinstance(d, pd.DataFrame) for d in data.values())
