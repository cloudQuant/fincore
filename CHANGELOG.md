# Changelog

All notable changes to this project will be documented in this file.

This changelog reports version **0.3.0** (the current release candidate).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Alphalens migration surfaces** — `fincore.alphalens` provides the
  source-shaped strict namespace, while `fincore.factor_analysis` provides the
  distinct enhanced prepare/analyze/render workflow and typed
  `PyfolioFactorInputs` bridge. Install `fincore[factor-analysis]` for
  compute-only enhanced analysis or `fincore[alphalens]` for the rendering
  stack.
- **Offline executable factor-analysis example** —
  `examples/factor_analysis_quickstart.py` uses fixed-seed synthetic data,
  makes no network requests or default file writes, renders headlessly under
  Agg, and closes its figures explicitly.

### Release blockers

- The human Alphalens license/NOTICE and provenance decision is still pending.
  This Unreleased entry does not create a release claim, a third-party notice,
  or a legal conclusion.

## [0.3.0] - unreleased (release candidate)

Current version, release candidate pending final acceptance. Package maturity:
**Beta** (classifier `Development Status :: 4 - Beta`).
No claim of "Stable" or "100% coverage" is made anywhere in this project.

### Added

- **Strict empyrical compatibility layer** — `fincore.empyrical` pins the frozen
  empyrical 0.6.0 surface (commit `74655e9`): 54/54 public symbols reach C0,
  49/49 callables reach C1 (signature), and the core callables are numerically
  verified (C3). Enforced by `tests/compat/fixtures/` manifests and the
  `tests/compat/` gates (CI job `compat`).
- **Pyfolio functional façade** — `fincore.pyfolio` implements the pinned
  pyfolio 0.9.6 profile of 11 tear-sheet workflows (commit `724bbd7`); all
  entries reach C1 and the risk/returns/perf-attrib/full-sheet main chains
  reach C4 end-to-end. The `Pyfolio` class is the enhanced OO convenience on
  the same workflows and requires the `pyfolio` extra.
- **Functional extras** — `pyfolio`, `interactive`, `report-pdf`,
  `report-xlsx`, `bayesian`, `data-yahoo`, `data-alphavantage`,
  `data-pandas-datareader`, `data-cn`, plus 0.3.x aliases `datareader` and
  `viz`.
- **Enhanced validation exceptions** — `ValidationError`, `NumericalError`,
  `DataAlignmentError`, `DependencyError`, `InvalidPeriodError` on the
  enhanced surfaces.
- **AnalysisContext snapshot semantics** — `replace_data()` atomically swaps
  inputs and invalidates every cached metric; caller-side mutation of the
  original inputs cannot stale the cached snapshot.
- **Machine-generated quality baseline** — `docs/quality/current-baseline.*`
  produced by `scripts/collect_quality_baseline.py`; README quality claims
  refer only to this snapshot.
- **Release gates** — CI jobs `marker-audit`, `coverage-branch` (baseline plus
  changed-lines gate), `compare-nonserial` (JUnit single vs xdist),
  `integration-offline`, wheel-consumer full-profile matrix in `build`.
- **Single version source** — `pyproject.toml` is authoritative; runtime
  resolution prefers installed distribution metadata.

### Changed

- **Python 3.11+ is now required** — a documented breaking change relative to
  empyrical.
- Flat API (`from fincore import ...`) remains bound to enhanced
  `fincore.metrics` semantics throughout 0.3.x; no deprecation is scheduled.
- `fincore.empyrical` strict façade keeps pinned legacy semantics (e.g.
  calendar-year weekly grouping, legacy CVaR tail policy) while the enhanced
  surfaces expose documented divergences (`week_year="iso"`, inclusive
  expected shortfall).
- Compatibility workflows never write into the installed package directory;
  legacy `run_flask_app` parameters remain accepted but display-only.
- `from fincore import Pyfolio` raises `DependencyError` naming
  `pip install fincore[pyfolio]` when the extra is absent.

### Fixed

- Drawdown tear sheets with fewer drawdowns than the top-N no longer raise
  `NaT ConversionError`.
- Wide and stacked perf-attrib inputs are equivalent; date gaps do not crash
  and the attribution identity holds.
- Legacy/canonical transaction inputs normalize losslessly; duplicate
  transaction timestamps are retained in stable order.
- Importing `fincore` or `Pyfolio` no longer changes the Matplotlib backend.
- Tests and compatibility workflows no longer write into site-packages or the
  source tree.

### Removed

- None.

## [0.1.0] - 2024-XX-XX

### Added
- Initial release
- Core financial metrics from empyrical
- Basic tearsheet functionality

---

## Historical snapshots (not release evidence)

### 1.0 planning snapshot (previously labeled `1.0.0 - 2025-02-18`)

The material below is retained as a historical completion/planning snapshot
only. Its counts and completion claims are **not** current acceptance evidence
and must be revalidated before any future release. The repository has no
1.0.0 release.

#### Reported as added in the historical snapshot
- **100% test coverage** - 8177 lines of code with 1800 passing tests
- **Comprehensive test suite** - Edge case coverage for all metrics and modules
- **AnalysisContext API** - Lazy, cached metric computation with `analyze()` function
- **RollingEngine** - Batch rolling metric computation for multiple metrics
- **Pluggable visualization backends** - Matplotlib, HTML, Plotly, Bokeh support
- **Data provider module** - Unified interface for Yahoo Finance, Alpha Vantage, Tushare, AkShare
- **Portfolio optimization module** - Efficient frontier, risk parity, constrained optimization
- **Monte Carlo simulation module** - Bootstrap analysis, scenario testing
- **Performance attribution module** - Brinson decomposition, Fama-French analysis
- **150+ financial metrics** - Comprehensive risk and performance analytics
- **Self-contained HTML reports** - No external dependencies for report generation
- **Three-tier lazy loading** - Fast import (~0.06s) with deferred heavy module loading
- **Registry-based method generation** - Eliminates ~1000 lines of boilerplate code

#### Reported as changed in the historical snapshot
- **Migrated from empyirical to fincore** package name
- **Python version support** - Now requires Python 3.11+
- **Improved NaN handling** - Robust handling of missing data throughout all calculations
- **Vectorized operations** - Performance improvements in rolling metrics
- **Type annotations** - Core modules fully annotated with type hints
- **Documentation** - Comprehensive bilingual (English/Chinese) user guide

#### Reported as deprecated in the historical snapshot
- None

#### Reported as removed in the historical snapshot
- None

#### Reported as fixed in the historical snapshot
- Fixed NaN handling in edge cases for all metrics
- Improved error messages for invalid inputs
- Fixed numerical stability issues in extreme value theory calculations
- Corrected timezone handling in date range calculations

#### Reported security status in the historical snapshot
- None
