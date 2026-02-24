# API Stability Policy

This document describes the API stability guarantees for fincore.

## Stable APIs

The following APIs are considered **stable** and will follow [Semantic Versioning](https://semver.org/):

### Top-Level Imports
```python
from fincore import (
    Empyrical,
    Pyfolio,
    analyze,
    create_strategy_report,
)
```

### Flat API Functions
All functions in `fincore.__all__` are stable:

```python
from fincore import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    annual_return,
    annual_volatility,
    cum_returns,
    cum_returns_final,
    alpha,
    beta,
    alpha_beta,
    calmar_ratio,
    omega_ratio,
    information_ratio,
    stability_of_timeseries,
    tail_ratio,
    value_at_risk,
    capture,
    downside_risk,
    simple_returns,
    aggregate_returns,
)
```

### AnalysisContext
The `AnalysisContext` class and its public methods are stable:

```python
import fincore

ctx = fincore.analyze(returns, factor_returns=benchmark)
ctx.sharpe_ratio
ctx.max_drawdown
ctx.perf_stats()
ctx.to_dict()
ctx.to_json()
ctx.to_html()
```

### RollingEngine
The `RollingEngine` class is stable:

```python
from fincore.core.engine import RollingEngine

engine = RollingEngine(returns, factor_returns=benchmark, window=60)
engine.compute(['sharpe', 'volatility', 'max_drawdown', 'beta'])
```

## Internal APIs

Modules and functions prefixed with `_` (underscore) are **internal** and may change
without notice between versions:

```python
from fincore import _registry  # Internal, may change
from fincore.metrics import _basic  # Internal, may change
```

## Versioning Policy

- **Major (X.0.0)**: Breaking changes to stable APIs
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, backward compatible

## Deprecation Process

If a stable API needs to be changed:

1. The old API will be marked as deprecated in documentation
2. A warning will be added (if applicable)
3. The old API will remain functional for at least one minor version
4. The old API will be removed in a major version update

## Python Version Support

fincore supports the following Python versions:
- Python 3.11
- Python 3.12
- Python 3.13

Unsupported Python versions may be removed in a major version update.

## Third-Party Dependencies

Third-party dependencies are considered part of the stable API. Changes to required
dependencies will only occur in minor or major versions, not patch versions.

Optional dependencies (viz, bayesian, datareader) may have their version requirements
updated in patch versions.

## Feedback

If you find an API that is incorrectly labeled (should be stable or internal), please
open an issue on GitHub.

## Questions?

For questions about API stability, please open a discussion on GitHub.
