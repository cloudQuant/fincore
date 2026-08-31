"""Tests for convenience functions and import errors.

Tests for get_provider function and ImportError handling.
Split from test_providers_unit.py for maintainability.
"""

from __future__ import annotations

import pandas as pd
import pytest


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
            def fetch_multiple(self, symbols, start, end, interval="1d", adjust=True, strict=False):
                if strict:
                    raise BatchFetchError("Dummy", errors={"BAD": ValueError("x")}, partial_results={})
                return {s: pd.DataFrame() for s in symbols}

        with pytest.raises(BatchFetchError):
            fetch_multiple_prices(["OK", "BAD"], provider=DummyProvider(), years=1, strict=True)


class TestProviderImportErrors:
    """Tests for ImportError handling in providers (lines 226-227, 387-388, 530-531)."""

    def test_yahoo_finance_import_error(self, monkeypatch):
        """Test YahooFinanceProvider handles its canonical loader failure gracefully."""
        from fincore.data.providers import YahooFinanceProvider
        from fincore.runtime import validation

        real_import_module = validation.importlib.import_module

        def missing_yfinance(name: str):
            if name == "yfinance":
                raise ModuleNotFoundError("No module named 'yfinance'")
            return real_import_module(name)

        monkeypatch.setattr(validation.importlib, "import_module", missing_yfinance)

        with pytest.raises(ImportError, match="yfinance"):
            YahooFinanceProvider()

    def test_alpha_vantage_import_error(self, monkeypatch):
        """Test AlphaVantageProvider handles its canonical loader failure gracefully."""
        from fincore.data.providers import AlphaVantageProvider
        from fincore.runtime import validation

        real_import_module = validation.importlib.import_module

        def missing_requests(name: str):
            if name == "requests":
                raise ModuleNotFoundError("No module named 'requests'")
            return real_import_module(name)

        monkeypatch.setattr(validation.importlib, "import_module", missing_requests)

        with pytest.raises(ImportError, match="requests is required"):
            AlphaVantageProvider(api_key="test_key")

    def test_tushare_import_error(self, monkeypatch):
        """Test TushareProvider handles its canonical loader failure gracefully."""
        from fincore.data.providers import TushareProvider
        from fincore.runtime import validation

        real_import_module = validation.importlib.import_module

        def missing_tushare(name: str):
            if name == "tushare":
                raise ModuleNotFoundError("No module named 'tushare'")
            return real_import_module(name)

        monkeypatch.setattr(validation.importlib, "import_module", missing_tushare)

        with pytest.raises(ImportError, match="tushare is required"):
            TushareProvider(token="test_token")
