# fincore 0.5

fincore is a unified Python platform for quantitative performance, portfolio,
factor, attribution, and risk analysis. Version **0.5.0** reorganises
those capabilities into focused canonical domains.

This is a breaking release: upstream-shaped Empyrical, Pyfolio, and Alphalens
facades, root-level metric calls, and compatibility extras are retired. Import
each operation from the owning module instead.

```python
import pandas as pd

from fincore.metrics.drawdown import max_drawdown
from fincore.metrics.ratios import sharpe_ratio

returns = pd.Series([0.01, -0.005, 0.002, 0.004])
print(sharpe_ratio(returns), max_drawdown(returns))
```

Start with [installation](getting-started/installation.md), the
[quick start](getting-started/quickstart.md), and the
[0.5 migration guide](getting-started/migration.md).

## Canonical domains

- `fincore.metrics`: return, drawdown, ratio, rolling, and statistical kernels.
- `fincore.performance`: cash-flow-aware return semantics and disclosures.
- `fincore.portfolio` and `fincore.report`: portfolio inputs, immutable report
  documents, and renderer-specific artifacts.
- `fincore.factor_analysis`: factor preparation, analysis, inference, costs,
  portfolios, and optional rendering.
- `fincore.attribution`, `risk`, `optimization`, and `simulation`: specialised
  financial analysis domains.
- `fincore.data`, `extensions`, `runtime`, and `viz`: platform capabilities.

The [API reference](api/index.md) gives the single public implementation path
for each area.
