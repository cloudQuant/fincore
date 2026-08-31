"""Direct, offline parity scenarios for the data and extensions domains."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest


def _price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": [10.0, 11.0], "High": [11.0, 12.0], "Low": [9.0, 10.0], "Close": [10.5, 11.5]},
        index=pd.date_range("2024-01-02", periods=2, freq="B"),
    )


class _OfflinePriceClient:
    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp, interval: str, adjust: bool) -> pd.DataFrame:
        assert symbol
        assert start < end
        assert interval == "1d"
        assert adjust is True
        return _price_frame()


class _OfflineProvider:
    def fetch(self, symbol: str, start: object, end: object, interval: str = "1d", adjust: bool = True) -> pd.DataFrame:
        assert symbol
        assert start is not None
        assert end is not None
        assert interval == "1d"
        assert adjust is True
        return _price_frame()

    def fetch_multiple(
        self,
        symbols: list[str],
        start: object,
        end: object,
        interval: str = "1d",
        adjust: bool = True,
        strict: bool = False,
    ) -> dict[str, pd.DataFrame]:
        assert start is not None
        assert end is not None
        assert interval == "1d"
        assert adjust is True
        assert strict is False
        return {symbol: _price_frame() for symbol in symbols}


def test_yahoo() -> None:
    from fincore.data.providers import YahooFinanceProvider

    provider = YahooFinanceProvider(client=_OfflinePriceClient())

    frame = provider.fetch("AAPL", "2024-01-01", "2024-02-01")

    pd.testing.assert_frame_equal(frame, _price_frame())


def test_yahoo_finance_provider() -> None:
    from fincore.data.providers import YahooFinanceProvider, get_provider

    provider = get_provider("yahoo", client=_OfflinePriceClient())

    assert isinstance(provider, YahooFinanceProvider)
    assert not provider.fetch("MSFT", "2024-01-01", "2024-02-01").empty


def test_tushare(monkeypatch: pytest.MonkeyPatch) -> None:
    from fincore.data.providers import TushareProvider

    class _Pro:
        def daily(self, **_kwargs: str) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "trade_date": ["20240102", "20240103"],
                    "open": [10.0, 11.0],
                    "high": [11.0, 12.0],
                    "low": [9.0, 10.0],
                    "close": [10.5, 11.5],
                    "vol": [100, 200],
                }
            )

    monkeypatch.setitem(sys.modules, "tushare", SimpleNamespace(pro_api=lambda _token: _Pro()))
    provider = TushareProvider(token="offline")

    assert list(provider.fetch("000001.SZ", "2024-01-01", "2024-02-01").columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]


def test_tushare_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from fincore.data.providers import TushareProvider, get_provider

    monkeypatch.setitem(sys.modules, "tushare", SimpleNamespace(pro_api=lambda _token: object()))

    assert isinstance(get_provider("tushare", token="offline"), TushareProvider)


def test_akshare(monkeypatch: pytest.MonkeyPatch) -> None:
    from fincore.data.providers import AkShareProvider

    def stock_zh_a_hist(**_kwargs: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03"],
                "开盘": [10.0, 11.0],
                "最高": [11.0, 12.0],
                "最低": [9.0, 10.0],
                "收盘": [10.5, 11.5],
                "成交量": [100, 200],
            }
        )

    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(stock_zh_a_hist=stock_zh_a_hist))
    provider = AkShareProvider()

    assert list(provider.fetch("000001", "2024-01-01", "2024-02-01").columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]


def test_get_provider() -> None:
    from fincore.data.providers import get_provider

    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("not-a-provider")


def test_fetch_price_data() -> None:
    from fincore.data.providers import fetch_price_data

    frame = fetch_price_data("AAPL", provider=_OfflineProvider(), start="2024-01-01", end="2024-02-01")

    pd.testing.assert_frame_equal(frame, _price_frame())


def test_fetch_multiple_prices() -> None:
    from fincore.data.providers import fetch_multiple_prices

    frames = fetch_multiple_prices(
        ["AAPL", "MSFT"],
        provider=_OfflineProvider(),
        start="2024-01-01",
        end="2024-02-01",
    )

    assert set(frames) == {"AAPL", "MSFT"}
    assert all(not frame.empty for frame in frames.values())


def test_data_providers_inherit_one_batch_fetch_contract() -> None:
    from fincore.data.providers import BatchFetchError, DataProvider

    class InMemoryProvider(DataProvider):
        def fetch(self, symbol, start, end, interval="1d", adjust=True):
            del start, end, interval, adjust
            if symbol == "BAD":
                raise ValueError("fixture failure")
            return _price_frame()

        def get_info(self, symbol: str) -> dict:
            return {"symbol": symbol}

    provider = InMemoryProvider()

    non_strict = provider.fetch_multiple(["OK", "BAD"], "2024-01-01", "2024-02-01")

    assert not non_strict["OK"].empty
    assert non_strict["BAD"].empty
    with pytest.raises(BatchFetchError) as error:
        provider.fetch_multiple(["OK", "BAD"], "2024-01-01", "2024-02-01", strict=True)
    assert error.value.provider == "InMemory"
    assert set(error.value.partial_results) == {"OK"}


def test_execute_hooks() -> None:
    from fincore.extensions.snapshot import ExtensionHook, ExtensionSnapshot

    snapshot = ExtensionSnapshot(
        hooks=(
            ExtensionHook(event="normalize", callable=lambda value: value + 1, priority=20),
            ExtensionHook(event="normalize", callable=lambda value: value * 2, priority=10),
        )
    )

    assert snapshot.execute_hooks("normalize", 3) == 7


def test_scope() -> None:
    from fincore.extensions.operations import operations
    from fincore.extensions.snapshot import ExtensionSnapshot
    from fincore.runtime import OperationCatalog

    base = OperationCatalog(())
    scoped = base.with_extensions(ExtensionSnapshot())

    assert operations() == ()
    assert base.extension_snapshot is None
    assert scoped.extension_snapshot is not None
    assert base.digest != scoped.digest
