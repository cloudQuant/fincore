# `fincore.factor_analysis`

Factor analysis is a first-class canonical domain. Its layers are explicit:

| Module | Responsibility |
| --- | --- |
| `data` | factor cleaning and forward-return preparation |
| `analysis` | factor model calculation |
| `performance` | returns, information coefficient, turnover, weights |
| `portfolio` | typed portfolio inputs |
| `costs` | transaction-cost, borrow, and capacity ledger |
| `inference` | post-analysis and Fama-MacBeth inference |
| `pit` | causal point-in-time factor materialisation |
| `render_matplotlib` and `tears` | optional explicit renderers |

```python
from fincore.factor_analysis.analysis import analyze_factor
from fincore.factor_analysis.data import get_clean_factor_and_forward_returns
from fincore.factor_analysis.performance import mean_return_by_quantile
```

Install `fincore[visualization]` only when invoking matplotlib rendering.
