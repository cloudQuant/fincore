# Migration from empyrical

The current fincore version is **0.3.0**. There is no current 1.0.0 release, so
do not require `fincore>=1.0.0`.

> **Breaking change:** fincore requires **Python 3.11+**; empyrical supports
> older interpreters.

## The three API surfaces

1. **Strict compatibility** — `fincore.empyrical` is the frozen empyrical
   0.6.0 surface (54 public symbols, 49 callables): all symbols C0, all
   callables C1, core callables C3.
2. **pyfolio façade** — `fincore.pyfolio` implements the frozen pyfolio 0.9.6
   profile of 11 workflows: all entries C1, risk/returns/perf-attrib/full-sheet
   main chains C4. `from fincore import Pyfolio` requires `fincore[pyfolio]`.
3. **Enhanced semantics** — `fincore.metrics`, the flat API, and
   `AnalysisContext` are fincore's own interfaces with documented divergences.
   Recommended for new code; not evidence of empyrical equality.

Details and the full C0–C4 matrix: [Compatibility](../development/compatibility.md).
Frozen manifests: `tests/compat/fixtures/`; executable gates: `tests/compat/`
(CI job `compat`).

## 0.3.x imports

Existing flat imports remain mapped to enhanced `fincore.metrics` functions in
0.3.x:

```python
import pandas as pd

from fincore import max_drawdown, sharpe_ratio

returns = pd.Series([0.01, -0.005, 0.002, 0.004])
print(sharpe_ratio(returns))
print(max_drawdown(returns))
```

Equal names do not imply equal empyrical signatures or edge-case behavior.
For empyrical-shaped calls, import the strict module explicitly:

```python
from fincore import empyrical

empyrical.sharpe_ratio(returns)
```

Do not blindly replace package imports. Inventory each used symbol, check its
compatibility level in the frozen manifest, and run differential tests on
production-shaped inputs before migrating it.

The generated flat-API migration manifest records every current target,
recommended future target, and deprecation state. No switch is scheduled; any
change requires a deprecation period and a future major release.

## Recommended destination: AnalysisContext

```python
import pandas as pd

import fincore

index = pd.date_range("2024-01-02", periods=5, freq="B")
returns = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001], index=index)
benchmark = pd.Series([0.008, -0.003, 0.001, 0.002, 0.0], index=index)

ctx = fincore.analyze(returns, factor_returns=benchmark)
ctx.sharpe_ratio
ctx.to_json(path="report.json")
```

## License/provenance review

Pyfolio's pinned checkout has MIT text in its root `LICENSE` and Apache-2.0
headers in inspected source files. Human/license review is pending. The project
does not infer a legal conclusion or generate a third-party notice until that
review decides what is required.
