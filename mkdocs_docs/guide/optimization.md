# Portfolio Optimization

## Efficient Frontier

```python
from fincore.optimization import efficient_frontier

ef = efficient_frontier(returns, n_points=50)
print(f"Min volatility: {ef['min_variance']['volatility']:.4f}")
print(f"Max Sharpe: {ef['max_sharpe']['sharpe']:.4f}")
```

## Risk Parity

```python
from fincore.optimization import risk_parity

rp = risk_parity(returns)
print(f"Weights: {rp['weights']}")
```

## Constrained Optimization

```python
from fincore.optimization import optimize

# Max Sharpe ratio
w = optimize(returns, objective="max_sharpe")

# Target return
w = optimize(returns, objective="target_return", target_return=0.15)

# Target risk
w = optimize(returns, objective="target_risk", target_volatility=0.12)
```

## API Reference

::: fincore.optimization.frontier.efficient_frontier

::: fincore.optimization.risk_parity.risk_parity

::: fincore.optimization.objectives.optimize
