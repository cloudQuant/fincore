# Capability Inventory

> Machine-generated from `fincore.capabilities`. Do not edit by hand.

| id | domain | status | public path | input contract | output contract | docs |
| --- | --- | --- | --- | --- | --- | --- |
| `risk.evt` | risk | `stable` | `fincore.risk.evt` | A returns Series/array (excess or simple) with optional tail probability. | Tail-risk estimates (VaR/CVaR) with the fitted EVT parameters attached. | `api/risk.md` |
| `risk.garch` | risk | `stable` | `fincore.risk.garch` | A returns Series/array and model order parameters (p, q). | Fitted conditional-volatility model with forecast and conditional VaR. | `api/risk.md` |
| `compat.empyrical` | compat | `stable` | `fincore.empyrical` | empyrical 0.6.0 call signatures and return shapes. | Numerically verified empyrical 0.6.0 results (C0-C3 gates). | `development/compatibility.md` |
| `compat.pyfolio` | compat | `stable` | `fincore.pyfolio` | pyfolio 0.9.6 tear-sheet workflow signatures. | pyfolio 0.9.6-profile tear-sheet workflows (C1/C4 gates). | `development/compatibility.md` |
| `compat.alphalens` | compat | `experimental` | `fincore.alphalens` | alphalens 0.4.0 source-shaped call signatures. | Source-shaped alphalens strict façade (Beta integration). | `development/compatibility.md` |
| `attribution.brinson` | attribution | `stable` | `fincore.attribution.brinson_attribution` | Portfolio/benchmark returns and weights of matching shape. | Allocation, selection and interaction effects summing to active return. | `api/attribution.md` |
| `attribution.brinson_hood` | attribution | `not_implemented` | `fincore.attribution.BrinsonAttribution.calculate` | (unavailable) would be portfolio/benchmark returns and weights. | (unavailable) would be Brinson-Hood-Faber attribution by period. | `api/attribution.md` |
| `attribution.fama_french_model` | attribution | `stable` | `fincore.attribution.FamaFrenchModel` | Asset/portfolio returns plus factor returns (3/4/5 factor). | Fitted factor exposures, idiosyncratic risk and R-squared. | `api/attribution.md` |
| `attribution.ff_factor_provider` | attribution | `provider_required` | `fincore.attribution.fetch_ff_factors` | Factor names and a date range; an injected provider is required. | Fama-French factor returns for the requested interval. | `api/attribution.md` |
| `attribution.style_analysis` | attribution | `stable` | `fincore.attribution.style_analysis` | Portfolio returns and optional style factor returns. | Style tilts and regression-based attribution. | `api/attribution.md` |
| `attribution.style_factor_provider` | attribution | `provider_required` | `fincore.attribution.fetch_style_factors` | Style factor identifiers and a date range; an injected provider is required. | Style factor returns for the requested interval. | `api/attribution.md` |
| `data.yahoo` | data | `provider_required` | `fincore.data.YahooFinanceProvider` | Symbol(s), date range; yfinance extra and a working transport. | Price history DataFrame with the provider's price-adjustment convention. | `api/fincore.md` |
| `data.alphavantage` | data | `provider_required` | `fincore.data.AlphaVantageProvider` | Symbol(s), date range and an API key; requests extra. | Price history DataFrame. | `api/fincore.md` |
| `data.tushare` | data | `provider_required` | `fincore.data.TushareProvider` | Symbol(s), date range and a Tushare token. | Chinese A-share price history DataFrame. | `api/fincore.md` |
| `data.akshare` | data | `provider_required` | `fincore.data.AkShareProvider` | Symbol(s) and date range; akshare extra. | Chinese market price history DataFrame. | `api/fincore.md` |
| `report.strategy_report` | report | `stable` | `fincore.report.create_strategy_report` | Returns Series (required) plus optional benchmark/positions/transactions/trades. | An HTML or PDF strategy report at the caller-selected path. | `api/report.md` |
| `factor_analysis.prepare` | factor_analysis | `experimental` | `fincore.factor_analysis.prepare_factor_data` | Factor, prices and optional groupby/quantile configuration. | A PreparedFactorData container with documented loss accounting. | `api/factor-analysis.md` |
| `factor_analysis.analyze` | factor_analysis | `experimental` | `fincore.factor_analysis.analyze_factor` | A PreparedFactorData container. | FactorAnalysisModel with IC, quantile returns and turnover. | `api/factor-analysis.md` |
| `factor_analysis.render` | factor_analysis | `experimental` | `fincore.factor_analysis.tears` | A FactorAnalysisModel or EventAnalysisModel. | Rendered tear-sheet figures with explicit close_owned_figures ownership. | `api/factor-analysis.md` |
