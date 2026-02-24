# Quick Start

## AnalysisContext (Recommended)

```python
import pandas as pd
import numpy as np
import fincore

dates = pd.bdate_range('2020-01-01', periods=252)
returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
benchmark = pd.Series(np.random.normal(0.0005, 0.015, 252), index=dates)

ctx = fincore.analyze(returns, factor_returns=benchmark)

print(f"Sharpe Ratio:     {ctx.sharpe_ratio:.4f}")
print(f"Max Drawdown:     {ctx.max_drawdown:.4f}")
print(f"Annual Return:    {ctx.annual_return:.4f}")

# Export
ctx.to_json()
ctx.to_html(path="report.html")
```

## Flat API (Function Style)

```python
import fincore

sr = fincore.sharpe_ratio(returns)
md = fincore.max_drawdown(returns)
ar = fincore.annual_return(returns)
```

## Classic API (Empyrical Class)

```python
from fincore import Empyrical

sharpe = Empyrical.sharpe_ratio(returns, risk_free=0.02/252)
alpha, beta = Empyrical.alpha_beta(returns, benchmark)
```

## RollingEngine

```python
from fincore.core.engine import RollingEngine

engine = RollingEngine(returns, factor_returns=benchmark, window=60)
results = engine.compute(['sharpe', 'volatility', 'max_drawdown', 'beta'])
```
