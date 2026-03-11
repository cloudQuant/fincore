---
project_name: fincore
user_name: cloud
date: 2026-03-09T13:20:00Z
sections_completed: ['technology_stack', 'language_rules', 'framework_rules', 'testing_rules', 'code_quality', 'workflow_rules', 'critical_rules']
existing_patterns_found: 5
---

# Project Context for AI Agents

_This file contains critical rules and patterns that AI agents must follow when implementing code in this project. Focus on unobvious details that agents might otherwise miss._

---

## Technology Stack & Versions

### Core Language
- **Python**: 3.11+ (supports 3.11, 3.12, 3.13)
- **License**: Apache-2.0

### Core Dependencies
- numpy >= 1.17.0
- pandas >= 0.25.0
- scipy >= 1.3.0
- pytz >= 2023.3
- packaging >= 21.0

### Optional Dependencies
- **viz**: matplotlib >= 3.3, seaborn >= 0.11, ipython >= 7.0
- **bayesian**: pymc >= 5.0
- **datareader**: pandas-datareader >= 0.8.0

### Development Tools
- pytest >= 6.0 with plugins: xdist, cov, benchmark, sugar
- ruff >= 0.4.0 (linting & formatting)
- mypy >= 1.5 (type checking)
- bottleneck >= 1.3.0 (performance optimization)
- parameterized >= 0.7 (parameterized tests)

### Version Constraints
- Minimum Python version: 3.11 (enforced in pyproject.toml)
- CI tests on all three versions: 3.11, 3.12, 3.13

---

## Critical Implementation Rules

### 1. Lazy Loading Architecture (CRITICAL)

This project uses a **three-tier lazy loading architecture** to optimize import time (~0.06s). Agents MUST understand and follow this pattern:

**Tier 1 - Top-level `__init__.py`:**
- Uses `__getattr__` to defer loading of `Empyrical`, `Pyfolio`, `analyze`, `create_strategy_report`
- Flat API functions (e.g., `sharpe_ratio`) loaded via `_FLAT_API` dict mapping
- NEVER import heavy submodules at top level

**Tier 2 - Metrics submodules:**
- Located in `fincore.metrics/` with 17+ modules
- Load via `__getattr__` in `metrics/__init__.py`
- Each module is only imported when accessed

**Tier 3 - Empyrical class methods:**
- Methods generated from `_registry.py` using `_LazyMethod` descriptor
- `_resolve_module()` lazy-loads metric modules only when method is first called
- Subsequent calls use cached staticmethod (zero overhead)

**Implementation Rules:**
```python
# ✅ CORRECT: Lazy import in __getattr__
def __getattr__(name: str):
    if name == "Empyrical":
        from .empyrical import Empyrical
        globals()["Empyrical"] = Empyrical
        return Empyrical

# ❌ WRONG: Top-level import
from .empyrical import Empyrical  # This defeats lazy loading!
```

**When adding new metrics:**
1. Implement in appropriate `fincore.metrics.*` module
2. Add to `_FLAT_API` dict in `__init__.py` if commonly used
3. Add to appropriate registry in `_registry.py` (CLASSMETHOD_REGISTRY or STATIC_METHODS)
4. NEVER import the metric module at top level

### 2. Registry-Based Method Generation

The `_registry.py` module eliminates ~1000 lines of boilerplate delegation code. Agents MUST use this pattern:

**Four Registries:**
1. **STATIC_METHODS** - Utility helpers (e.g., `align`)
2. **CLASSMETHOD_REGISTRY** - Simple forwarding to module functions
3. **DUAL_RETURNS_REGISTRY** - Auto-fills `returns` from instance
4. **DUAL_RETURNS_FACTOR_REGISTRY** - Auto-fills both `returns` AND `factor_returns`

**Pattern:**
```python
# In _registry.py
CLASSMETHOD_REGISTRY = {
    "sharpe_ratio": ("_ratios", "sharpe_ratio"),
    "max_drawdown": ("_drawdown", "max_drawdown"),
    # ... more mappings
}

# In empyrical.py - methods auto-generated via decorator
@_populate_from_registry
class Empyrical:
    # sharpe_ratio, max_drawdown, etc. are automatically attached
    pass
```

**When adding new methods:**
1. Implement function in appropriate metrics module
2. Add mapping to correct registry
3. Method appears automatically on Empyrical class
4. NO manual delegation code needed

### 3. Dual Method Pattern

The `@_dual_method` descriptor allows methods to work both ways:

```python
# Class-level usage
Empyrical.sharpe_ratio(returns, risk_free=0.02)

# Instance-level usage
emp = Empyrical(returns=returns)
emp.sharpe_ratio()  # returns auto-filled from instance
```

**Implementation Rules:**
- Use `@_dual_method` for methods that need both calling patterns
- Access instance attributes via `self.returns`, `self.factor_returns`
- Maintain exact same signature for both call patterns
- Cache bound methods to avoid repeated wrapper creation

### 4. NaN Handling (CRITICAL)

All functions use **custom NaN-aware operations** from `fincore.utils`:

**Rules:**
- Use `nanmean`, `nanstd`, `nansum` from utils, NOT numpy directly
- Bottleneck optimization used when available
- All metrics must handle NaN gracefully
- Return `np.nan` for invalid inputs, never raise exceptions for edge cases

**Example:**
```python
# ✅ CORRECT: Use utils for NaN handling
from fincore.utils import nanmean, nanstd

def sharpe_ratio(returns):
    return nanmean(returns) / nanstd(returns)

# ❌ WRONG: Direct numpy usage
import numpy as np
def sharpe_ratio(returns):
    return np.mean(returns) / np.std(returns)  # NaN not handled!
```

### 5. Return Type Requirements

**CRITICAL**: Library expects **simple returns** (not log returns) as input.

```python
# ✅ CORRECT: Simple returns
simple_returns = (price_t / price_t_1) - 1

# ❌ WRONG: Log returns
log_returns = np.log(price_t / price_t_1)
```

**DataFrame Support:**
- When DataFrames passed, metrics calculate for each column independently
- Preserve DataFrame structure in output
- Align indexes automatically using `align` function from utils

---

## Language-Specific Rules

### Python Import Rules

**Circular Import Prevention:**
```python
# ✅ CORRECT: Import inside function
def some_function():
    from fincore.metrics.ratios import sharpe_ratio
    return sharpe_ratio(returns)

# ❌ WRONG: Top-level circular import
from fincore.metrics.ratios import sharpe_ratio  # May cause circular dependency
```

**Type Annotations:**
- Core modules (`core.*`, `metrics.*`) use `disallow_untyped_defs: false` in mypy
- New code SHOULD include type hints for clarity
- Use `TYPE_CHECKING` for imports that are type-only:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame
```

### Error Handling Patterns

**Graceful Degradation:**
- Return `np.nan` for edge cases, never crash
- Log warnings for unexpected conditions
- Use `try/except` sparingly, prefer validation

```python
# ✅ CORRECT: Graceful handling
def max_drawdown(returns):
    if len(returns) == 0:
        return np.nan
    # ... calculation

# ❌ WRONG: Raising exceptions
def max_drawdown(returns):
    if len(returns) == 0:
        raise ValueError("Empty returns")  # Too aggressive!
```

### Performance Patterns

**Vectorization:**
- Prefer vectorized numpy/pandas operations over loops
- Use `np.where`, `np.select` for conditional logic
- Avoid Python loops for array operations

**Bottleneck Optimization:**
- Automatically uses bottleneck if installed
- Custom implementations in utils for NaN-aware ops
- Benchmark critical paths using pytest-benchmark

---

## Framework-Specific Rules

### Financial Metrics Implementation

**Annualization:**
- Most risk-adjusted metrics use annualization factors from `fincore.constants`
- Default period is DAILY (252 trading days)
- Always check if metric needs annualization

```python
from fincore.constants import DAILY, WEEKLY, MONTHLY

def annual_volatility(returns, period=DAILY):
    annualization_factor = get_annualization_factor(period)
    return returns.std() * np.sqrt(annualization_factor)
```

**Alignment:**
- Use `align` function to align returns with factor_returns
- Handle different frequencies automatically
- Preserve index alignment throughout calculations

**Date Handling:**
- All dates should be timezone-aware (use pytz)
- Business day calculations use 252 days/year
- Handle missing data points gracefully

### Visualization Backend System

**Pluggable Backend Pattern:**
- `VizBackend` protocol defines interface
- Three backends: matplotlib, html, interactive (plotly/bokeh)
- Backends implement: `plot_returns()`, `plot_drawdown()`, etc.

```python
from fincore.viz import get_backend

backend = get_backend('matplotlib')
backend.plot_returns(returns)
```

**When adding visualizations:**
1. Add method to `VizBackend` protocol in `viz/base.py`
2. Implement in all backend classes
3. Use protocol methods, never backend-specific code

---

## Testing Rules

### Test Organization

**File Structure:**
```
tests/
├── test_core/           # Core functionality tests
├── test_empyrical/      # Empyrical metrics tests
├── test_pyfolio/        # Pyfolio tearsheet tests
├── test_data/           # Test fixtures (CSV files)
└── index_data/          # Global market index data
```

**Naming Conventions:**
- Files: `test_*.py`
- Classes: `Test*`
- Functions: `test_*`

### Test Execution

**Parallel Testing:**
```bash
# Run all tests with 4 workers
pytest tests/ -n 4

# Auto-detect workers
pytest tests/ -n auto --dist=loadscope
```

**Coverage Requirements:**
```bash
pytest tests/ --cov=fincore --cov-report=term-missing
```

### Test Priority Markers

Use pytest markers to categorize tests:
- `@pytest.mark.p0` - Critical (core metrics, revenue-critical, security)
- `@pytest.mark.p1` - High (frequently used features)
- `@pytest.mark.p2` - Medium (secondary features)
- `@pytest.mark.p3` - Low (rarely used, cosmetic)

```python
@pytest.mark.p0
def test_sharpe_ratio_critical():
    # Core metric test
```

### Test Data Fixtures

**Location:** `tests/test_data/`
- `returns.csv` - Price returns data
- `factor_returns.csv` - Benchmark/factor returns
- `positions.csv`, `factor_loadings.csv`, `residuals.csv` - Attribution data

**Usage:**
```python
import pandas as pd

def test_with_fixtures():
    returns = pd.read_csv('tests/test_data/returns.csv', index_col=0, parse_dates=True)
```

### Benchmarking

Use pytest-benchmark for performance tests:
```python
def test_sharpe_performance(benchmark):
    returns = generate_test_returns(10000)
    result = benchmark(sharpe_ratio, returns)
    assert not np.isnan(result)
```

---

## Code Quality & Style Rules

### Ruff Configuration

**Settings** (from pyproject.toml):
- Line length: 120 characters
- Target version: py311
- Selected rules: E, F, I, UP, B, SIM

**Ignored Rules** (project-specific):
- `E402` - Module-level import not at top (intentional for lazy loading)
- `E501` - Line too long (handled by ruff format)
- `F401` - Unused import (many are dynamically accessed via __getattr__)

**Running Ruff:**
```bash
ruff check fincore/
ruff format fincore/
```

### MyPy Configuration

**Settings:**
- Python version: 3.11
- `warn_return_any: true`
- `ignore_missing_imports: true` (for third-party libs)

**Module Overrides:**
- Core/metrics modules: `disallow_untyped_defs: false`
- New code should include type hints despite this setting

**Running MyPy:**
```bash
mypy fincore/
```

### Code Organization

**Module Structure:**
```
fincore/
├── __init__.py          # Lazy imports, flat API
├── empyrical.py         # Main facade class
├── pyfolio.py           # Tearsheet extensions
├── _registry.py         # Method registry
├── _types.py            # Type definitions
├── core/                # AnalysisContext, RollingEngine
├── metrics/             # 17+ metric modules
├── viz/                 # Visualization backends
├── constants/           # Period constants
├── utils/               # Helper utilities
├── report/              # HTML/PDF report generation
├── attribution/         # Factor attribution
├── optimization/        # Portfolio optimization
├── risk/                # Risk models (GARCH, EVT)
└── simulation/          # Monte Carlo, bootstrap
```

### Naming Conventions

**Functions/Methods:**
- Snake_case: `sharpe_ratio`, `max_drawdown`
- Descriptive verbs: `calculate_*`, `compute_*`, `generate_*`

**Classes:**
- PascalCase: `Empyrical`, `AnalysisContext`, `RollingEngine`

**Constants:**
- UPPER_SNAKE_CASE: `DAILY`, `WEEKLY`, `MONTHLY`

**Private functions:**
- Leading underscore: `_resolve_module`, `_populate_from_registry`

### Documentation

**Docstrings:**
- Use Google-style docstrings
- Include examples for complex functions
- Document parameters and return types

```python
def sharpe_ratio(returns, risk_free=0.0, period=DAILY):
    """Calculate the Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free: Risk-free rate (default 0.0)
        period: Period for annualization (default DAILY)

    Returns:
        float: Sharpe ratio

    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01])
        >>> sharpe_ratio(returns)
        2.123
    """
```

---

## Development Workflow Rules

### Git Workflow

**Branch Naming:**
- Feature: `feature/add-new-metric`
- Bugfix: `fix/sharpe-calculation`
- Refactor: `refactor/lazy-loading`

**Commit Messages:**
- Use conventional commits format
- Be specific about what changed

```
feat: add Sortino ratio implementation
fix: handle NaN in max_drawdown calculation
refactor: implement lazy loading for metrics
test: add edge case tests for alpha_beta
```

### Pull Request Checklist

Before creating PR:
- [ ] Run all tests: `pytest tests/ -n 4`
- [ ] Check coverage: `pytest --cov=fincore`
- [ ] Run linter: `ruff check fincore/`
- [ ] Run formatter: `ruff format fincore/`
- [ ] Type check: `mypy fincore/`
- [ ] Update documentation if needed
- [ ] Add tests for new functionality

### Code Review Standards

**Review Checklist:**
1. Does it follow lazy loading architecture?
2. Are NaN cases handled properly?
3. Is it using utils for NaN-aware operations?
4. Does it maintain DataFrame compatibility?
5. Are there sufficient tests?
6. Is it documented?

---

## Critical Don't-Miss Rules

### Anti-Patterns to Avoid

**1. Breaking Lazy Loading:**
```python
# ❌ WRONG: Import at top level
from fincore.metrics.ratios import sharpe_ratio

# ✅ CORRECT: Lazy import
def get_sharpe():
    from fincore.metrics.ratios import sharpe_ratio
    return sharpe_ratio
```

**2. Direct numpy for NaN operations:**
```python
# ❌ WRONG
import numpy as np
mean = np.mean(returns)  # Doesn't handle NaN

# ✅ CORRECT
from fincore.utils import nanmean
mean = nanmean(returns)
```

**3. Using log returns:**
```python
# ❌ WRONG
log_returns = np.log(prices.pct_change() + 1)

# ✅ CORRECT
simple_returns = prices.pct_change()
```

**4. Raising exceptions for edge cases:**
```python
# ❌ WRONG
if len(returns) == 0:
    raise ValueError("Empty returns")

# ✅ CORRECT
if len(returns) == 0:
    return np.nan
```

**5. Ignoring DataFrame columns:**
```python
# ❌ WRONG: Only works for Series
def metric(returns):
    return returns.mean()

# ✅ CORRECT: Works for both Series and DataFrame
def metric(returns):
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: _metric_single(col))
    return _metric_single(returns)
```

### Edge Cases to Handle

**1. Empty returns:**
```python
if len(returns) == 0:
    return np.nan
```

**2. All NaN returns:**
```python
if returns.isna().all():
    return np.nan
```

**3. Single value:**
```python
if len(returns) == 1:
    return np.nan  # Or appropriate default
```

**4. Zero volatility:**
```python
vol = returns.std()
if vol == 0:
    return np.nan  # Avoid division by zero in Sharpe
```

**5. Mismatched indexes:**
```python
# Always align before calculations
returns, factor_returns = align(returns, factor_returns)
```

### Security Considerations

**1. Input Validation:**
- Validate return values are numeric
- Check for infinite values
- Handle extremely large/small numbers

**2. Numerical Stability:**
- Use `np.clip` to avoid overflow
- Check for division by zero
- Use `np.finfo` for float limits

### Performance Gotchas

**1. Avoid repeated calculations:**
```python
# ❌ WRONG: Calculate multiple times
def sharpe_ratio(returns):
    return returns.mean() / returns.std()

def sortino_ratio(returns):
    downside = returns[returns < 0]
    return returns.mean() / downside.std()

# ✅ CORRECT: Calculate once
def performance_metrics(returns):
    mean = returns.mean()
    vol = returns.std()
    sharpe = mean / vol
    # ... use cached values
```

**2. DataFrame iteration:**
```python
# ❌ WRONG: Loop over DataFrame
for col in df.columns:
    result[col] = metric(df[col])

# ✅ CORRECT: Vectorized
result = df.apply(metric)
```

**3. Large array operations:**
- Use chunking for very large datasets
- Consider memory usage for rolling windows
- Profile with pytest-benchmark

---

## Quick Reference for AI Agents

### When Adding New Metrics

1. **Implement in correct module** (`fincore.metrics.*`)
2. **Handle NaN properly** (use utils functions)
3. **Add to registry** (`_registry.py`)
4. **Write comprehensive tests** (p0/p1 priority)
5. **Document with examples**
6. **Check DataFrame compatibility**
7. **Benchmark performance**

### When Modifying Existing Code

1. **Preserve lazy loading** (no top-level imports)
2. **Maintain backward compatibility**
3. **Update tests if behavior changes**
4. **Check all callers** (class-level and instance-level)
5. **Run full test suite**
6. **Update documentation**

### When Fixing Bugs

1. **Add failing test first**
2. **Fix the minimal necessary code**
3. **Check for similar issues elsewhere**
4. **Update edge case handling**
5. **Document the fix in commit message**

---

_Remember: This library is used for quantitative finance. Accuracy and correctness are paramount. Always prioritize correctness over performance, but maintain efficiency where possible._
