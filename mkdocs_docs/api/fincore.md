# `fincore` root namespace

The root package is an intentionally small namespace index. It exposes the
version and canonical domains only:

```python
import fincore

print(fincore.__version__)
print(fincore.metrics)
print(fincore.report)
```

It does not re-export metrics, façade classes, stateful contexts, or upstream
package compatibility modules. Import executable APIs from their focused leaf
module, for example:

```python
from fincore.metrics.ratios import sharpe_ratio
from fincore.report.portfolio.compute import build_portfolio_report
```
