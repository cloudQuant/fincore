"""Unit tests for data providers (no network required).

These tests focus on the logic of provider classes without making actual
network calls. For integration tests with real network calls, see
test_providers_integration.py.
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

    def test_validate_dates_string(self):
        """Test date validation with string dates."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        start, end = provider.validate_dates("2020-01-01", "2020-12-31")
        assert start == pd.to_datetime("2020-01-01")
        assert end == pd.to_datetime("2020-12-31")

    def test_validate_dates_datetime(self):
        """Test date validation with datetime objects."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        dt_start = datetime(2020, 1, 1)
        dt_end = datetime(2020, 12, 31)
        start, end = provider.validate_dates(dt_start, dt_end)
        assert start == pd.to_datetime(dt_start)
        assert end == pd.to_datetime(dt_end)

    def test_validate_dates_invalid_range(self):
        """Test date validation rejects invalid date range."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        with pytest.raises(ValueError):
            provider.validate_dates("2020-12-31", "2020-01-01")


class TestYahooFinanceProviderUnit:
    """Unit tests for Yahoo Finance provider (no network)."""

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
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        prices = pd.DataFrame(
            {
                "Close": [100, 102, 101, 103, 105],
                "Adj Close": [98, 100, 99, 101, 103],
            },
            index=dates,
        )

        returns = provider.to_returns(prices)

        assert isinstance(returns, pd.Series)
        assert not returns.empty
        # Returns should have one less element than prices
        assert len(returns) == len(prices) - 1

    def test_to_returns_with_multiple_columns(self):
        """Test converting prices to returns with multiple columns."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        prices = pd.DataFrame(
            {
                "Close": [100, 102, 101, 103, 105],
                "Adj Close": [98, 100, 99, 101, 103],
                "Open": [99, 101, 100, 102, 104],
            },
            index=dates,
        )

        # Should use Adj Close by default
        returns = provider.to_returns(prices)

        assert isinstance(returns, pd.Series)
        assert len(returns) == len(prices) - 1

    def test_fetch_multiple_default_non_strict_returns_empty_on_failure(self, monkeypatch):
        """Default mode should keep compatibility and return empty DataFrame on failures."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        sample = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="D"),
        )

        def fake_fetch(symbol, start, end, interval="1d", adjust=True):  # noqa: ARG001
            if symbol == "BAD":
                raise ValueError("fetch failed")
            return sample

        monkeypatch.setattr(provider, "fetch", fake_fetch)

        results = provider.fetch_multiple(["OK", "BAD"], "2023-01-01", "2023-01-10")
        assert "OK" in results
        assert "BAD" in results
        assert not results["OK"].empty
        assert results["BAD"].empty

    def test_fetch_multiple_strict_raises_batch_error(self, monkeypatch):
        """Strict mode should aggregate failures and raise BatchFetchError."""
        from fincore.data.providers import BatchFetchError, YahooFinanceProvider

        provider = YahooFinanceProvider()

        sample = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="D"),
        )

        def fake_fetch(symbol, start, end, interval="1d", adjust=True):  # noqa: ARG001
            if symbol in {"BAD1", "BAD2"}:
                raise ValueError(f"failed: {symbol}")
            return sample

        monkeypatch.setattr(provider, "fetch", fake_fetch)

        with pytest.raises(BatchFetchError) as exc_info:
            provider.fetch_multiple(["OK", "BAD1", "BAD2"], "2023-01-01", "2023-01-10", strict=True)

        err = exc_info.value
        assert set(err.errors.keys()) == {"BAD1", "BAD2"}
        assert "OK" in err.partial_results
        assert not err.partial_results["OK"].empty


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


class TestConvenienceFunctionsUnit:
    """Unit tests for convenience functions (no network)."""

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

    def test_get_provider_av(self):
        """Test getting Alpha Vantage provider."""
        try:
            from fincore.data.providers import AlphaVantageProvider, get_provider

            provider = get_provider("av", api_key="test_key")
            assert isinstance(provider, AlphaVantageProvider)
        except ImportError:
            pytest.skip("requests not installed")

    def test_get_provider_tushare(self):
        """Test getting Tushare provider."""
        try:
            from fincore.data.providers import TushareProvider, get_provider

            provider = get_provider("tushare", token="test_token")
            assert isinstance(provider, TushareProvider)
        except ImportError:
            pytest.skip("tushare not installed")

    def test_fetch_multiple_prices_strict_passes_through(self, monkeypatch):
        """fetch_multiple_prices should pass strict=True to provider.fetch_multiple."""
        from fincore.data.providers import BatchFetchError, fetch_multiple_prices

        class DummyProvider:
            def fetch_multiple(self, symbols, start, end, interval="1d", adjust=True, strict=False):  # noqa: ARG002
                if strict:
                    raise BatchFetchError("Dummy", errors={"BAD": ValueError("x")}, partial_results={})
                return {s: pd.DataFrame() for s in symbols}

        with pytest.raises(BatchFetchError):
            fetch_multiple_prices(["OK", "BAD"], provider=DummyProvider(), years=1, strict=True)
