from __future__ import annotations

import sys
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest


class _DummyResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_batch_fetch_error_message_and_fields() -> None:
    from fincore.data.providers import BatchFetchError

    err = BatchFetchError(
        provider="X",
        errors={"B": ValueError("b"), "A": RuntimeError("a")},
        partial_results={"A": pd.DataFrame()},
    )
    assert err.provider == "X"
    assert set(err.errors.keys()) == {"A", "B"}
    assert "Failed to fetch 2 symbol(s) from X" in str(err)


def test_yahoo_finance_fetch_and_get_info_are_offline(monkeypatch) -> None:
    from fincore.data.providers import YahooFinanceProvider

    idx = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
    sample = pd.DataFrame(
        {"Open": [1.0, 1.1, 1.2], "Close": [1.0, 1.05, 1.1]},
        index=idx,
    )

    class _DummyTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol
            self.info = {
                "longName": "Example Corp",
                "exchange": "XNAS",
                "currency": "USD",
                "industry": "Software",
                "sector": "Technology",
                "marketCap": 123,
                "sharesOutstanding": 456,
            }

        def history(self, **_kwargs):
            return sample

    class _DummyYF:
        def Ticker(self, symbol: str):
            return _DummyTicker(symbol)

    provider = YahooFinanceProvider()
    monkeypatch.setattr(provider, "_yf", _DummyYF())

    df = provider.fetch("AAPL", start="2024-01-01", end="2024-02-01")
    assert not df.empty
    assert "Close" in df.columns

    info = provider.get_info("AAPL")
    assert info["symbol"] == "AAPL"
    assert info["name"] == "Example Corp"
    assert info["exchange"] == "XNAS"
    assert info["currency"] == "USD"


def test_yahoo_finance_fetch_raises_when_empty(monkeypatch) -> None:
    from fincore.data.providers import YahooFinanceProvider

    class _DummyTicker:
        def history(self, **_kwargs):
            return pd.DataFrame()

    class _DummyYF:
        def Ticker(self, _symbol: str):
            return _DummyTicker()

    provider = YahooFinanceProvider()
    monkeypatch.setattr(provider, "_yf", _DummyYF())

    with pytest.raises(ValueError, match="No data found"):
        provider.fetch("AAPL", start="2024-01-01", end="2024-02-01")


def test_alpha_vantage_fetch_success_and_error_are_offline(monkeypatch) -> None:
    from fincore.data.providers import AlphaVantageProvider

    provider = AlphaVantageProvider(api_key="k")

    payload = {
        "Time Series (Daily)": {
            "2024-01-02": {
                "1. open": "1.0",
                "2. high": "1.1",
                "3. low": "0.9",
                "4. close": "1.05",
                "5. volume": "100",
            },
            "2024-01-03": {
                "1. open": "1.05",
                "2. high": "1.2",
                "3. low": "1.0",
                "4. close": "1.1",
                "5. volume": "200",
            },
        }
    }

    monkeypatch.setattr(provider._session, "get", lambda *_args, **_kwargs: _DummyResponse(payload))

    df = provider.fetch("AAPL", start="2024-01-01", end="2024-01-31", interval="1d", adjust=True)
    assert not df.empty
    assert "Adj Close" in df.columns

    err_payload = {"Note": "rate limit"}
    monkeypatch.setattr(provider._session, "get", lambda *_args, **_kwargs: _DummyResponse(err_payload))
    with pytest.raises(ValueError, match="Alpha Vantage error"):
        provider.fetch("AAPL", start="2024-01-01", end="2024-01-31")


def test_alpha_vantage_get_info_offline(monkeypatch) -> None:
    from fincore.data.providers import AlphaVantageProvider

    provider = AlphaVantageProvider(api_key="k")
    info_payload = {
        "Name": "Example Corp",
        "Exchange": "XNAS",
        "Currency": "USD",
        "Industry": "Software",
        "Sector": "Technology",
        "MarketCapitalization": "123",
        "SharesOutstanding": "456",
    }
    monkeypatch.setattr(provider._session, "get", lambda *_args, **_kwargs: _DummyResponse(info_payload))

    info = provider.get_info("AAPL")
    assert info["symbol"] == "AAPL"
    assert info["name"] == "Example Corp"


def test_tushare_provider_offline_fetch_and_info_via_stub_module(monkeypatch) -> None:
    # Create a lightweight tushare stub to avoid external dependency.
    from fincore.data import providers as providers_mod

    class _DummyPro:
        def daily(self, ts_code: str, start_date: str, end_date: str, adj: str):  # noqa: ARG002
            return pd.DataFrame(
                {
                    "trade_date": ["20240102", "20240103"],
                    "open": [1.0, 1.1],
                    "high": [1.2, 1.3],
                    "low": [0.9, 1.0],
                    "close": [1.05, 1.15],
                    "vol": [100, 200],
                    "amount": [1000, 2000],
                }
            )

        def stock_basic(self, ts_code: str, fields: str):  # noqa: ARG002
            return pd.DataFrame([{"ts_code": ts_code, "name": "Example", "industry": "X", "list_date": "20200101"}])

    dummy_ts = SimpleNamespace(pro_api=lambda _token: _DummyPro())

    # Patch sys.modules so __init__ can import tushare.
    monkeypatch.setitem(sys.modules, "tushare", dummy_ts)

    provider = providers_mod.TushareProvider(token="t")

    df = provider.fetch("000001.SZ", start="2024-01-01", end="2024-02-01", adjust=True)
    assert not df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    info = provider.get_info("000001.SZ")
    assert info["symbol"] == "000001.SZ"
    assert info["name"] == "Example"


def test_akshare_provider_offline_fetch_and_info_via_stub_module(monkeypatch) -> None:
    from fincore.data import providers as providers_mod

    def stock_zh_a_hist(symbol: str, period: str, start_date: str, end_date: str, adjust: str):  # noqa: ARG001
        return pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03"],
                "开盘": [1.0, 1.1],
                "最高": [1.2, 1.3],
                "最低": [0.9, 1.0],
                "收盘": [1.05, 1.15],
                "成交量": [100, 200],
                "成交额": [1000, 2000],
            }
        )

    def stock_individual_info_em(symbol: str):  # noqa: ARG001
        return {"item": {"股票简称": "Example", "交易所": "SZ", "行业": "X"}}

    dummy_ak = SimpleNamespace(
        stock_zh_a_hist=stock_zh_a_hist,
        stock_individual_info_em=stock_individual_info_em,
    )

    monkeypatch.setitem(sys.modules, "akshare", dummy_ak)

    provider = providers_mod.AkShareProvider()
    df = provider.fetch("000001", start="2024-01-01", end="2024-02-01")
    assert not df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    info = provider.get_info("000001")
    assert info["name"] == "Example"


def test_fetch_price_data_and_multiple_prices_default_date_logic(monkeypatch) -> None:
    from fincore.data.providers import fetch_multiple_prices, fetch_price_data

    calls: list[tuple[str, datetime, datetime]] = []

    class DummyProvider:
        def fetch(self, symbol: str, start, end, interval="1d", adjust=True):  # noqa: ARG002
            calls.append((symbol, pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime()))
            return pd.DataFrame({"Close": [1.0]}, index=[pd.Timestamp("2024-01-02")])

        def fetch_multiple(self, symbols: list[str], start, end, interval="1d", adjust=True, strict=False):  # noqa: ARG002
            for s in symbols:
                calls.append((s, pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime()))
            return {s: pd.DataFrame({"Close": [1.0]}, index=[pd.Timestamp("2024-01-02")]) for s in symbols}

    # Cover end=None branch with a fixed "today".
    monkeypatch.setattr(pd.Timestamp, "today", classmethod(lambda cls: pd.Timestamp("2024-02-01")))  # type: ignore[arg-type]

    df = fetch_price_data("AAPL", provider=DummyProvider(), start=None, end=None, years=1)
    assert not df.empty

    out = fetch_multiple_prices(["AAPL", "MSFT"], provider=DummyProvider(), start=None, end=None, years=2)
    assert set(out.keys()) == {"AAPL", "MSFT"}
    assert len(calls) >= 3
