# Changelog

All notable changes to Fincore are documented here. This changelog reports version **0.5.1.dev0**, the current development version; **0.5.0** is the latest release.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and version labels follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-09-01

### Changed

- **Unified core** — Fincore 0.5 exposes direct domain namespaces only:
  `metrics`, `portfolio`, `factor_analysis`, `attribution`, `risk`, `report`,
  `performance`, `optimization`, `simulation`, `data`, `runtime`, `extensions`,
  and `viz`.
- **Compute-once reporting** — portfolio, factor, and risk workflows build a
  shared `ReportDocument`; HTML, Matplotlib, PDF, XLSX, Plotly, and Bokeh
  renderers project that document without recomputing financial values.
- **Capability extras** — install direct extras such as
  `fincore[visualization]`, `fincore[factor-analysis]`,
  `fincore[report-pdf]`, or `fincore[report-xlsx]`.

### Removed

- The `empyrical`, `pyfolio`, and `alphalens` package-shaped APIs, flat
  root metric exports, facade classes, mutable plugin registry, legacy
  dispatcher, and tear-sheet wrappers.
- Compatibility-profile extras and aliases. Use the canonical domain modules
  and direct capability extras instead.

### Migration notes

- `fincore.metrics.ratios.sharpe_ratio` replaces root and Empyrical metric
  calls; choose each leaf module by domain ownership.
- `fincore.report.portfolio.compute.build_portfolio_report` replaces
  Pyfolio-style report/tear-sheet construction.
- `fincore.factor_analysis` and `fincore.attribution` contain the direct
  factor and attribution workflows formerly reached through package facades.

## Historical releases

Pre-0.5 release notes and compatibility claims remain available in Git history.
They do not describe the public surface of this breaking release.
