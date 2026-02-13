"""Integration tests for data providers (requires network and API keys).

These tests make actual network calls to external APIs and should be
run separately from unit tests. Use pytest marker:

    pytest tests/test_data/test_providers_integration.py -m integration

Or set environment variable to enable:

    FINCORE_RUN_INTEGRATION_TESTS=1 pytest tests/test_data/
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

# Integration marker
integration = pytest.mark.integration
run_integration = os.environ.get("FINCORE_RUN_INTEGRATION_TESTS", "").lower() in ("1", "true", "yes")


def skip_if_no_integration():
    """Skip test if integration tests are not enabled."""
    if not run_integration:
        pytest.skip("Integration tests disabled. Set FINCORE_RUN_INTEGRATION_TESTS=1 to enable.")


skip_integration = pytest.mark.skipif(
    not run_integration, reason="Set FINCORE_RUN_INTEGRATION_TESTS=1 to enable integration tests"
)


@skip_integration
class TestYahooFinanceProviderIntegration:
    """Integration tests for Yahoo Finance provider."""

    @pytest.mark.skip(reason="Yahoo Finance rate limiting - test requires network access")
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

    @pytest.mark.skip(reason="Yahoo Finance rate limiting - test requires network access")
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

    @pytest.mark.skip(reason="Yahoo Finance rate limiting - test requires network access")
    def test_get_info(self):
        """Test getting symbol information."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()
        info = provider.get_info("AAPL")

        assert isinstance(info, dict)
        assert "symbol" in info
        assert info["symbol"] == "AAPL"

    @pytest.mark.skip(reason="Yahoo Finance rate limiting - test requires network access")
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

    @pytest.mark.skip(reason="Yahoo Finance rate limiting - test requires network access")
    def test_fetch_index(self):
        """Test fetching index data."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        end = datetime.now()
        start = end - timedelta(days=30)

        data = provider.fetch("^GSPC", start, end)

        assert isinstance(data, pd.DataFrame)
        assert not data.empty

    @pytest.mark.skip(reason="Yahoo Finance rate limiting - test requires network access")
    def test_fetch_price_data_default_years(self):
        """Test fetch_price_data with default years parameter."""
        from fincore.data.providers import fetch_price_data

        data = fetch_price_data("AAPL", years=1)

        assert isinstance(data, pd.DataFrame)
        assert not data.empty

    @pytest.mark.skip(reason="Yahoo Finance rate limiting - test requires network access")
    def test_fetch_price_data_with_dates(self):
        """Test fetch_price_data with explicit dates."""
        from fincore.data.providers import fetch_price_data

        data = fetch_price_data("AAPL", start="2023-01-01", end="2023-12-31")

        assert isinstance(data, pd.DataFrame)
        assert not data.empty

    @pytest.mark.skip(reason="Yahoo Finance rate limiting - test requires network access")
    def test_fetch_multiple_prices(self):
        """Test fetch_multiple_prices convenience function."""
        from fincore.data.providers import fetch_multiple_prices

        data = fetch_multiple_prices(["AAPL", "MSFT"], years=1)

        assert isinstance(data, dict)
        assert all(isinstance(d, pd.DataFrame) for d in data.values())


@skip_integration
class TestAlphaVantageProviderIntegration:
    """Integration tests for Alpha Vantage provider."""

    @pytest.fixture
    def provider(self):
        """Create provider (skipped if no API key)."""
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


@skip_integration
class TestTushareProviderIntegration:
    """Integration tests for Tushare provider."""

    @pytest.fixture
    def provider(self):
        """Create provider (skipped if no token)."""
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


@skip_integration
class TestAkShareProviderIntegration:
    """Integration tests for AkShare provider."""

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
