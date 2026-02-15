from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest


def test_alpha_vantage_fetch_multiple_strict_and_nonstrict(monkeypatch) -> None:
    from fincore.data.providers import AlphaVantageProvider, BatchFetchError

    provider = AlphaVantageProvider(api_key="k")

    def fake_fetch(symbol: str, *_args, **_kwargs):
        if symbol == "BAD":
            raise RuntimeError("boom")
        return pd.DataFrame({"Close": [1.0]}, index=[pd.Timestamp("2024-01-02")])

    monkeypatch.setattr(provider, "fetch", fake_fetch)

    out = provider.fetch_multiple(["OK", "BAD"], start="2024-01-01", end="2024-01-31", strict=False)
    assert set(out.keys()) == {"OK", "BAD"}
    assert not out["OK"].empty
    assert out["BAD"].empty

    with pytest.raises(BatchFetchError) as ei:
        provider.fetch_multiple(["OK", "BAD"], start="2024-01-01", end="2024-01-31", strict=True)
    err = ei.value
    assert err.provider == "AlphaVantage"
    assert "BAD" in err.errors
    assert "OK" in err.partial_results


def test_tushare_fetch_multiple_and_get_info_empty(monkeypatch) -> None:
    from fincore.data import providers as providers_mod
    from fincore.data.providers import BatchFetchError

    class _DummyPro:
        def stock_basic(self, ts_code: str, fields: str):  # noqa: ARG002
            return pd.DataFrame()

    dummy_ts = SimpleNamespace(pro_api=lambda _token: _DummyPro())
    monkeypatch.setitem(sys.modules, "tushare", dummy_ts)

    provider = providers_mod.TushareProvider(token="t")

    # Cover fetch_multiple error handling without calling the network.
    def fake_fetch(symbol: str, *_args, **_kwargs):
        if symbol == "BAD":
            raise ValueError("nope")
        return pd.DataFrame({"Close": [1.0]}, index=[pd.Timestamp("2024-01-02")])

    monkeypatch.setattr(provider, "fetch", fake_fetch)

    out = provider.fetch_multiple(["OK", "BAD"], start="2024-01-01", end="2024-01-31", strict=False)
    assert not out["OK"].empty
    assert out["BAD"].empty

    with pytest.raises(BatchFetchError):
        provider.fetch_multiple(["OK", "BAD"], start="2024-01-01", end="2024-01-31", strict=True)

    # Cover get_info empty branch.
    info = provider.get_info("000001.SZ")
    assert info["symbol"] == "000001.SZ"
    assert info["name"] == "N/A"


def test_akshare_fetch_multiple_and_get_info_exception(monkeypatch) -> None:
    from fincore.data import providers as providers_mod
    from fincore.data.providers import BatchFetchError

    def stock_individual_info_em(symbol: str):  # noqa: ARG001
        raise RuntimeError("down")

    dummy_ak = SimpleNamespace(
        stock_zh_a_hist=lambda **_kwargs: pd.DataFrame(),  # not used here
        stock_individual_info_em=stock_individual_info_em,
    )
    monkeypatch.setitem(sys.modules, "akshare", dummy_ak)

    provider = providers_mod.AkShareProvider()

    def fake_fetch(symbol: str, *_args, **_kwargs):
        if symbol == "BAD":
            raise ValueError("nope")
        return pd.DataFrame({"Close": [1.0]}, index=[pd.Timestamp("2024-01-02")])

    monkeypatch.setattr(provider, "fetch", fake_fetch)

    out = provider.fetch_multiple(["OK", "BAD"], start="2024-01-01", end="2024-01-31", strict=False)
    assert not out["OK"].empty
    assert out["BAD"].empty

    with pytest.raises(BatchFetchError):
        provider.fetch_multiple(["OK", "BAD"], start="2024-01-01", end="2024-01-31", strict=True)

    info = provider.get_info("000001")
    assert info["symbol"] == "000001"
    assert info["name"] == "N/A"


def test_fetch_price_data_and_multiple_prices_provider_as_string_and_date_strings(monkeypatch) -> None:
    from fincore.data import providers as providers_mod

    calls: list[tuple[str, pd.Timestamp, pd.Timestamp, bool]] = []

    class _DummyProvider:
        def fetch(self, symbol: str, start, end, interval="1d", adjust=True):  # noqa: ARG002
            calls.append((symbol, pd.Timestamp(start), pd.Timestamp(end), bool(adjust)))
            return pd.DataFrame({"Close": [1.0]}, index=[pd.Timestamp("2024-01-02")])

        def fetch_multiple(self, symbols, start, end, interval="1d", adjust=True, strict=False):  # noqa: ARG002
            for s in symbols:
                calls.append((s, pd.Timestamp(start), pd.Timestamp(end), bool(adjust)))
            if strict:
                raise AssertionError("strict should be passed through but not trigger here")
            return {s: pd.DataFrame({"Close": [1.0]}, index=[pd.Timestamp("2024-01-02")]) for s in symbols}

    monkeypatch.setattr(providers_mod, "get_provider", lambda *_args, **_kwargs: _DummyProvider())

    df = providers_mod.fetch_price_data("AAPL", provider="yahoo", start="2024-01-01", end="2024-01-31")
    assert not df.empty

    out = providers_mod.fetch_multiple_prices(["AAPL", "MSFT"], provider="yahoo", start="2024-01-01", end="2024-01-31")
    assert set(out.keys()) == {"AAPL", "MSFT"}
    assert len(calls) >= 3
