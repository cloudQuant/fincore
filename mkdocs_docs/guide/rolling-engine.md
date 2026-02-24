# RollingEngine

Batch rolling metric computation in a single call.

## Usage

```python
from fincore.core.engine import RollingEngine

engine = RollingEngine(returns, factor_returns=benchmark, window=60)
results = engine.compute(['sharpe', 'volatility', 'max_drawdown', 'beta'])

for name, series in results.items():
    print(f"{name}: {len(series)} observations")
```

## Available Metrics

- `sharpe` — Rolling Sharpe ratio
- `volatility` — Rolling annualized volatility
- `max_drawdown` — Rolling maximum drawdown (vectorized)
- `beta` — Rolling beta vs benchmark (requires `factor_returns`)
- `sortino` — Rolling Sortino ratio
- `mean_return` — Rolling annualized mean return

## API Reference

::: fincore.core.engine.RollingEngine
