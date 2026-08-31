# `fincore.performance`

The performance domain owns return semantics that depend on valuations, cash
flows, fees, currency, or inference policy. It is distinct from scalar metric
kernels in `fincore.metrics`.

```python
from fincore.performance.cashflows import cashflow_adjusted_returns, cashflow_adjusted_twr
from fincore.performance.returns import twr
```

The `cashflows`, `returns`, `inference`, and `disclosures` modules make timing,
assumptions, and output status explicit.
