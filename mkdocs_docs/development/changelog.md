# Changelog

See the full [CHANGELOG.md](https://github.com/cloudQuant/fincore/blob/master/CHANGELOG.md)
in the repository.

## [0.3.0] - unreleased (release candidate)

Current version. Package maturity: **Beta**. No Stable/1.0 claim is made.

### Added

- Strict empyrical 0.6.0 compatibility layer (`fincore.empyrical`): 54/54
  public symbols (C0), 49/49 callables (C1), core callables C3.
- Pyfolio 0.9.6-profile façade (`fincore.pyfolio`): 11 workflows C1,
  risk/returns/perf-attrib/full-sheet main chains C4.
- Functional extras: pyfolio, interactive, report-pdf, report-xlsx, bayesian,
  data-yahoo, data-alphavantage, data-pandas-datareader, data-cn.
- Enhanced validation exceptions and AnalysisContext snapshot semantics
  (`replace_data()`, cache invalidation, immunity to caller-side mutation).
- Machine-generated quality baseline and CI release gates.

### Changed

- Python 3.11+ required (breaking change vs empyrical).
- Flat API remains bound to enhanced `fincore.metrics` semantics in 0.3.x.

### Fixed

- Drawdown tear sheets with fewer drawdowns than top-N (no `NaT
  ConversionError`).
- Wide/stacked perf-attrib equivalence; attribution identity holds across date
  gaps.
- Lossless legacy/canonical transaction normalization.
- No Matplotlib backend mutation on import; no package-dir writes from
  compatibility workflows.

## Historical snapshot (not release evidence)

An earlier planning document labeled `1.0.0 - 2025-02-18` (including a "100%
test coverage" claim) is retained in `CHANGELOG.md` purely as a historical
snapshot. Its numbers are not current acceptance evidence; the repository has
no 1.0.0 release.
