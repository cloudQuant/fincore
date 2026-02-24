# Migration from empyrical

## Import Changes

```python
# Before (empyrical)
from empyrical import sharpe_ratio, max_drawdown, alpha_beta

# After (fincore) — drop-in replacement
from fincore import sharpe_ratio, max_drawdown, alpha_beta
```

## New Features in fincore

| Feature | empyrical | fincore |
|---------|-----------|---------|
| Core Metrics | ✅ | ✅ |
| Lazy Loading | ❌ | ✅ |
| AnalysisContext | ❌ | ✅ |
| RollingEngine | ❌ | ✅ |
| Data Providers | ❌ | ✅ |
| Portfolio Optimization | ❌ | ✅ |
| Pluggable Viz | ❌ | ✅ |
| Python 3.13 | ❌ | ✅ |

## Migration Steps

1. Replace `empyrical` with `fincore>=1.0.0` in `requirements.txt`
2. Find-and-replace: `from empyrical import` → `from fincore import`
3. Run your existing tests — most should work without changes
4. Gradually adopt `AnalysisContext` for better performance
