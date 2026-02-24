# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-02-18

### Added
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

### Changed
- **Migrated from empyirical to fincore** package name
- **Python version support** - Now requires Python 3.11+
- **Improved NaN handling** - Robust handling of missing data throughout all calculations
- **Vectorized operations** - Performance improvements in rolling metrics
- **Type annotations** - Core modules fully annotated with type hints
- **Documentation** - Comprehensive bilingual (English/Chinese) user guide

### Deprecated
- None

### Removed
- None

### Fixed
- Fixed NaN handling in edge cases for all metrics
- Improved error messages for invalid inputs
- Fixed numerical stability issues in extreme value theory calculations
- Corrected timezone handling in date range calculations

### Security
- None

## [0.1.0] - 2024-XX-XX

### Added
- Initial release
- Core financial metrics from empyrical
- Basic tearsheet functionality
