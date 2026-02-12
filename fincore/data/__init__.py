"""Data provider integrations for fincore.

Provides unified access to financial data from multiple sources:
- Yahoo Finance (via yfinance)
- Alpha Vantage (via alpha-vantage)
- Tushare (Chinese A-share data)
- AkShare (Chinese financial data)
"""

from __future__ import annotations

from fincore.data.providers import (
    DataProvider,
    YahooFinanceProvider,
    AlphaVantageProvider,
    TushareProvider,
    AkShareProvider,
    get_provider,
    fetch_price_data,
    fetch_multiple_prices,
)

__all__ = [
    "DataProvider",
    "YahooFinanceProvider",
    "AlphaVantageProvider",
    "TushareProvider",
    "AkShareProvider",
    "get_provider",
    "fetch_price_data",
    "fetch_multiple_prices",
]
