# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The package metadata currently reports version **0.3.0**. No 1.0.0 release is
recorded here.

### Current changes

- Pinned static API/signature manifests for empyrical 0.6.0 and a bounded
  pyfolio 0.9.6 compatibility profile.
- Hardened those manifests to read pinned Git blobs, resolve safe constant
  defaults, preserve aliases/star exports, bind optional oracle evidence to the
  pinned checkout, and invalidate human review attestations on evidence drift.
- Bounded static constant parsing by depth, node visits, container/scalar size,
  and numeric magnitude; all Git/oracle subprocesses are now noninteractive
  and time-limited with operation-specific failures.
- Added explicit compatibility matrices and upstream provenance review notes.
- Corrected migration and README claims that previously implied certified
  drop-in compatibility, no breaking changes, or 100% coverage.

### Historical 1.0 planning snapshot (not release evidence)

The material below was previously labeled `1.0.0 - 2025-02-18`. It is retained
as a historical completion/planning snapshot only. Its counts and completion
claims are not current acceptance evidence and must be revalidated before any
future release.

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

## [0.1.0] - 2024-XX-XX

### Added
- Initial release
- Core financial metrics from empyrical
- Basic tearsheet functionality
