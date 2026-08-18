"""Provider contract tests: injected clients and controlled dependency errors."""

from __future__ import annotations

import builtins
import sys

import pandas as pd
import pytest

from fincore.data.providers import YahooFinanceProvider
from fincore.exceptions import DependencyError


class FakeYahooClient:
    """A fake in-memory transport used to exercise provider logic offline."""

    def fetch(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        interval: str = "1d",
        adjust: bool = True,
    ) -> pd.DataFrame:
        return pd.DataFrame({"Close": [10.0, 11.0]}, index=pd.to_datetime([start, end]))


def test_provider_can_use_an_injected_fake_client_when_sdk_is_unavailable() -> None:
    provider = YahooFinanceProvider(client=FakeYahooClient())

    start, _ = provider.validate_dates("2024-01-01", "2024-01-02")

    assert start.year == 2024


def test_provider_fetch_uses_injected_client_offline() -> None:
    provider = YahooFinanceProvider(client=FakeYahooClient())

    data = provider.fetch("AAPL", "2024-01-01", "2024-01-02")

    assert list(data["Close"]) == [10.0, 11.0]


def test_broken_optional_sdk_raises_dependency_error_not_attribute_error(monkeypatch) -> None:
    """A broken SDK import must surface as a controlled DependencyError."""
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "yfinance":
            raise AttributeError("module 'lib' has no attribute 'GEN_EMAIL'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    monkeypatch.delitem(sys.modules, "yfinance", raising=False)

    with pytest.raises(DependencyError, match="data-yahoo") as excinfo:
        YahooFinanceProvider()

    assert excinfo.value.dependency == "yfinance"
    assert excinfo.value.extra == "data-yahoo"
    assert isinstance(excinfo.value.__cause__, AttributeError)


def test_missing_optional_sdk_raises_dependency_error(monkeypatch) -> None:
    real_import = builtins.__import__

    def missing_import(name, *args, **kwargs):
        if name == "yfinance":
            raise ImportError("No module named 'yfinance'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_import)
    monkeypatch.delitem(sys.modules, "yfinance", raising=False)

    with pytest.raises(DependencyError, match="data-yahoo"):
        YahooFinanceProvider()


def test_dependency_error_remains_an_import_error() -> None:
    """Historical ``except ImportError`` call sites keep working."""
    err = DependencyError("broken", dependency="yfinance", extra="data-yahoo")

    assert isinstance(err, ImportError)
