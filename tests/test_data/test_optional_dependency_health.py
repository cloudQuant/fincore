"""Optional-dependency health: core paths must survive a broken optional SDK.

A broken optional SDK (yfinance/akshare) must never invalidate client-injected
unit tests or leak a raw third-party error out of a constructor.  These tests
run offline and never contact a market-data service.
"""

from __future__ import annotations

import builtins
import sys

import pandas as pd
import pytest

from fincore.data.providers import AkShareProvider, YahooFinanceProvider
from fincore.exceptions import DependencyError


class _FakeClient:
    def fetch(self, symbol, start, end, interval="1d", adjust=True):
        return pd.DataFrame({"Close": [1.0]})


def test_yahoo_provider_survives_broken_sdk_when_client_injected(monkeypatch) -> None:
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "yfinance":
            raise RuntimeError("yfinance native stack is broken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    monkeypatch.delitem(sys.modules, "yfinance", raising=False)

    provider = YahooFinanceProvider(client=_FakeClient())
    assert provider.validate_dates("2024-01-01", "2024-01-02")[0].year == 2024


def test_akshare_broken_sdk_raises_controlled_dependency_error(monkeypatch) -> None:
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "akshare":
            raise AttributeError("module 'lib' has no attribute 'GEN_EMAIL'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    monkeypatch.delitem(sys.modules, "akshare", raising=False)

    with pytest.raises(DependencyError, match="data-cn"):
        AkShareProvider()


def test_dependency_error_names_install_extra() -> None:
    with pytest.raises(DependencyError, match="data-yahoo"):
        raise DependencyError("yfinance missing", dependency="yfinance", extra="data-yahoo")
