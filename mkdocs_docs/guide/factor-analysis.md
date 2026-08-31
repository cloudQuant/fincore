# Factor analysis

Factor analysis combines data preparation, model calculation, inference,
portfolio construction, and optional rendering in one canonical domain. There
is no source-shaped compatibility route in 0.5.

```python
from fincore.factor_analysis.analysis import analyze_factor
from fincore.factor_analysis.data import get_clean_factor_and_forward_returns

prepared = get_clean_factor_and_forward_returns(factor, prices, periods=(1, 5))
model = analyze_factor(prepared.data, periods=("1D", "5D"))
```

Inspect preparation loss reports, select cost/borrow/capacity assumptions
explicitly, and use the point-in-time APIs where input availability must be
audited. The bundled `examples/factor_analysis_quickstart.py` is deterministic
and runs offline; rendering requires `fincore[visualization]`.
