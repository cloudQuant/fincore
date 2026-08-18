"""Data provider integrations for fincore.

Provides unified access to financial data from multiple sources:
- Yahoo Finance (via yfinance)
- Alpha Vantage (via alpha-vantage)
- Tushare (Chinese A-share data)
- AkShare (Chinese financial data)

All providers are ``provider_required`` capabilities: each needs its optional
extra and a working transport.  States are declared in
:mod:`fincore.capabilities` and rendered into
``docs/quality/capability-inventory.md``.
"""

from __future__ import annotations

from fincore.data.providers import (
    AkShareProvider,
    AlphaVantageProvider,
    DataProvider,
    TushareProvider,
    YahooFinanceProvider,
    fetch_multiple_prices,
    fetch_price_data,
    get_provider,
)

__all__ = [
    "AkShareProvider",
    "AlphaVantageProvider",
    "DataProvider",
    "TushareProvider",
    "YahooFinanceProvider",
    "fetch_multiple_prices",
    "fetch_price_data",
    "get_provider",
]
