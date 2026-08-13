# Quick Start

Every code block below is executed verbatim by `tests/docs/test_examples.py`.
Define inputs in each block; later blocks do not inherit earlier variables.

## AnalysisContext (Recommended)

```python
import pandas as pd
import numpy as np
import fincore

dates = pd.bdate_range('2020-01-01', periods=252)
returns = pd.Series(np.random.default_rng(0).normal(0.001, 0.02, 252), index=dates)
benchmark = pd.Series(np.random.default_rng(1).normal(0.0005, 0.015, 252), index=dates)

ctx = fincore.analyze(returns, factor_returns=benchmark)

print(f"Sharpe Ratio:     {ctx.sharpe_ratio:.4f}")
print(f"Max Drawdown:     {ctx.max_drawdown:.4f}")
print(f"Annual Return:    {ctx.annual_return:.4f}")

# Export
ctx.to_json(path="report.json")
ctx.to_html(path="report.html")
```

## Flat API (Function Style)

The flat API is bound to enhanced `fincore.metrics` semantics.

```python
import pandas as pd

import fincore

returns = pd.Series([0.01, -0.005, 0.002, 0.004])

sr = fincore.sharpe_ratio(returns)
md = fincore.max_drawdown(returns)
ar = fincore.annual_return(returns)

print(sr, md, ar)
```

## Strict Compatibility Module

For empyrical 0.6.0-shaped calls, import the strict surface explicitly.

```python
import pandas as pd

from fincore import empyrical

returns = pd.Series([0.01, -0.005, 0.002, 0.004])

print(empyrical.sharpe_ratio(returns))
print(empyrical.max_drawdown(returns))
```

## Classic API (Empyrical Class)

```python
import pandas as pd

from fincore import Empyrical

returns = pd.Series([0.01, -0.005, 0.002, 0.004])
benchmark = pd.Series([0.003, 0.002, -0.001, 0.005])

sharpe = Empyrical.sharpe_ratio(returns, risk_free=0.02/252)
alpha, beta = Empyrical.alpha_beta(returns, benchmark)

print(sharpe, alpha, beta)
```

## Instance API (State-Bound)

```python
import pandas as pd

from fincore import Empyrical

returns = pd.Series([0.01, -0.005, 0.002, 0.004])

emp = Empyrical(returns=returns)
print(emp.sharpe_ratio())
print(emp.max_drawdown())
```

## RollingEngine

```python
import numpy as np
import pandas as pd

from fincore.core.engine import RollingEngine

rng = np.random.default_rng(7)
index = pd.date_range("2024-01-02", periods=60, freq="B")
returns = pd.Series(rng.normal(0.001, 0.02, 60), index=index)
benchmark = pd.Series(rng.normal(0.0005, 0.015, 60), index=index)

engine = RollingEngine(returns, factor_returns=benchmark, window=30)
results = engine.compute(['sharpe', 'volatility', 'max_drawdown', 'beta'])

print(results.keys())
```
