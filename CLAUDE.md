# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Empyrical is a Python library for calculating common financial risk and performance metrics used in quantitative finance. Originally developed by Quantopian Inc., it provides statistical functions for analyzing investment returns, calculating risk metrics, and performing performance attribution.

## Development Commands

### How to Query Available Virtual Environments

```bash
conda env list
```

### How to Create New Virtual Environments

```bash
conda create -n py37 python=3.7
conda create -n py38 python=3.8
conda create -n py39 python=3.9
conda create -n py310 python=3.10
conda create -n py311 python=3.11
conda create -n py312 python=3.12
conda create -n py313 python=3.13
```

### How to Activate Virtual Environments

```bash
conda activate py37
conda activate py38
conda activate py39
conda activate py310
conda activate py311
conda activate py312
conda activate py313
```

### How to Deactivate Virtual Environments

```bash
conda deactivate
```

### How to Install Dependencies
```bash
# Windows
install_win.bat

# Unix/Linux/MacOS
sh install_unix.sh
```

### How to Run Tests
```bash
# Run all tests in parallel
pytest ./empyrical/tests -n 4

# Run a single test file
pytest ./empyrical/tests/test_stats.py

# Run a specific test
pytest ./empyrical/tests/test_stats.py::test_function_name
```

### How to Set Up Development Environment
```bash
# Install dependencies
pip install -U -r requirements.txt

# Install package in development mode
pip install -U ./empyrical
```

## Architecture Overview

### Core Modules

1. **stats.py** (2,242 lines) - The main module containing all financial metrics calculations. Functions are organized by metric type:
   - Return metrics (annual_return, cagr, cum_returns)
   - Risk metrics (max_drawdown, value_at_risk, downside_risk)
   - Risk-adjusted metrics (sharpe_ratio, sortino_ratio, calmar_ratio)
   - Market relationship metrics (alpha, beta, capture ratios)
   - Rolling versions of metrics (prefixed with `roll_`)

2. **utils.py** (576 lines) - Utility functions for data validation, NaN handling, and return calculations. Key functions include:
   - `nanmean`, `nanstd`, `nanmax`, `nanmin` - NaN-aware statistical operations (with bottleneck optimization)
   - `simple_returns`, `cum_returns` - Calculate returns from prices
   - `rolling_window` - Create rolling windows for calculations
   - `_adjust_returns` - Adjust returns data type and handle required_return parameter
   - `up`, `down` - Identify up/down market periods

3. **periods.py** (20 lines) - Constants defining annualization factors:
   - DAILY=252, WEEKLY=52, MONTHLY=12, QUARTERLY=4, YEARLY=1
   - ANNUALIZATION_FACTORS dict for period conversion

4. **perf_attrib.py** (161 lines) - Performance attribution functionality for decomposing returns

5. **deprecate.py** (44 lines) - Deprecation warning utilities

6. **__init__.py** (83 lines) - Module exports and public API surface

### Design Patterns

- **Consistent Function Signatures**: Most metrics follow the pattern `metric(returns, risk_free=0, period=DAILY, **kwargs)`
- **NaN Handling**: Functions use custom NaN-aware operations from utils.py with bottleneck optimization when available
- **Rolling Calculations**: Rolling versions of metrics use `_create_unary_vectorized_roll_function` and `_create_binary_vectorized_roll_function` factory functions
- **Pandas Integration**: All functions accept pandas Series/DataFrame and numpy arrays
- **Aligned Calculations**: Many metrics have both regular and `_aligned` versions for handling misaligned data

### Testing Approach

- Tests use CSV files in `tests/test_data/` for realistic financial data:
  - `returns.csv` - Price returns data
  - `factor_returns.csv` - Benchmark/factor returns
  - `positions.csv`, `factor_loadings.csv`, `residuals.csv` - Performance attribution data
- Each metric has comprehensive tests covering edge cases (empty data, all NaN, single values)
- Tests verify both correctness and consistency with expected financial calculations
- Uses parameterized tests for multiple test cases
- Test data includes 2 years of daily returns (2016-2017)

### Important Implementation Notes

1. **Annualization**: Many metrics use annualization factors from periods.py. Always check if a metric needs annualization.

2. **Risk-Free Rate**: Most risk-adjusted metrics accept a `risk_free` parameter that can be a scalar or Series

3. **Return Type Handling**: The library assumes simple returns (not log returns) as input

4. **DataFrame Support**: When DataFrames are passed, metrics are calculated for each column independently

5. **Period Detection**: Some functions attempt to infer the period from pandas DatetimeIndex if not specified

6. **Optional Dependencies**:
   - `bottleneck` provides optimized NaN-aware operations for performance
   - `pandas-datareader` is deprecated and being removed in future versions

7. **Performance**: Vectorized operations throughout, with special rolling window implementations for efficiency
