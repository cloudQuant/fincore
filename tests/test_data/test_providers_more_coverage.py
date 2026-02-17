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


def test_yahoo_finance_provider_with_session(monkeypatch) -> None:
    """Test YahooFinanceProvider with a custom session (line 269)."""
    from unittest.mock import MagicMock

    from fincore.data.providers import YahooFinanceProvider

    # Create a mock session
    mock_session = MagicMock()

    # Create a mock yfinance module
    mock_ticker_class = MagicMock()
    mock_history_df = pd.DataFrame(
        {"Close": [100.0, 101.0], "Volume": [1000, 1100]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = mock_history_df
    mock_ticker_class.return_value = mock_ticker_instance

    mock_yf = MagicMock()
    mock_yf.Ticker = mock_ticker_class
    monkeypatch.setitem(sys.modules, "yfinance", mock_yf)

    provider = YahooFinanceProvider(session=mock_session)
    data = provider.fetch("AAPL", start="2024-01-01", end="2024-01-31")

    # Verify the session was passed to history
    mock_ticker_instance.history.assert_called_once()
    call_kwargs = mock_ticker_instance.history.call_args.kwargs
    assert "session" in call_kwargs
    assert call_kwargs["session"] is mock_session
    assert not data.empty


def test_tushare_provider_empty_data_raises(monkeypatch) -> None:
    """Test TushareProvider raises ValueError when data is empty (line 587)."""
    from fincore.data import providers as providers_mod

    class _EmptyPro:
        def daily(self, ts_code: str, start_date: str, end_date: str, adj: str):  # noqa: ARG002
            return pd.DataFrame()  # Empty DataFrame

    dummy_ts = SimpleNamespace(pro_api=lambda _token: _EmptyPro())
    monkeypatch.setitem(sys.modules, "tushare", dummy_ts)

    provider = providers_mod.TushareProvider(token="t")

    with pytest.raises(ValueError, match="No data found for symbol"):
        provider.fetch("000001.SZ", start="2024-01-01", end="2024-01-31")


def test_akshare_provider_import_error(monkeypatch) -> None:
    """Test AkShareProvider raises ImportError when akshare is not available (lines 683-684)."""
    from fincore.data import providers as providers_mod

    # Remove akshare from modules if it exists
    monkeypatch.delitem(sys.modules, "akshare", raising=False)

    # Mock the import to raise ImportError
    def mock_import(name, *args, **kwargs):
        if name == "akshare":
            raise ImportError("No module named 'akshare'")
        return __import__(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    with pytest.raises(ImportError, match="akshare is required"):
        providers_mod.AkShareProvider()


def test_akshare_provider_empty_data_raises(monkeypatch) -> None:
    """Test AkShareProvider raises ValueError when data is empty (line 727)."""
    from fincore.data import providers as providers_mod

    dummy_ak = SimpleNamespace(
        stock_zh_a_hist=lambda **_kwargs: pd.DataFrame()  # Empty DataFrame
    )
    monkeypatch.setitem(sys.modules, "akshare", dummy_ak)

    provider = providers_mod.AkShareProvider()

    with pytest.raises(ValueError, match="No data found for symbol"):
        provider.fetch("000001", start="2024-01-01", end="2024-01-31")
