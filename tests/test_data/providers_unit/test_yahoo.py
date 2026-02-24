"""Tests for Yahoo Finance provider unit tests.

Tests for YahooFinanceProvider without network calls.
Split from test_providers_unit.py for maintainability.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest


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


class TestYahooFinanceProviderWithSession:
    """Tests for Yahoo Finance provider with custom session."""

    def test_provider_with_session(self):
        """Test provider creation with custom session (line 269)."""
        from fincore.data.providers import YahooFinanceProvider

        # Create a provider and verify it can be initialized with session
        # The actual session usage is covered in integration tests
        session = MagicMock()
        provider = YahooFinanceProvider(session)
        # Verify session is stored
        assert provider._session is session
