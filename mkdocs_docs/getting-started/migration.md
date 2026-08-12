# Migration from empyrical

The current fincore version is **0.3.0**. There is no current 1.0.0 release, so
do not require `fincore>=1.0.0`.

## Current compatibility boundary

Fincore is not yet certified as a drop-in empyrical replacement. The project
has frozen two targets:

- empyrical 0.6.0: 54 public symbols, including 49 callables;
- pyfolio 0.9.6: 11 functional tear-sheet workflows.

The frozen manifests describe upstream source and signatures. Their C0–C4
implementation statuses remain unverified until executable compatibility tests
are completed. See the repository's
[empyrical matrix](https://github.com/cloudQuant/fincore/blob/main/docs/compatibility/empyrical-0.6.0.md)
and [pyfolio profile](https://github.com/cloudQuant/fincore/blob/main/docs/compatibility/pyfolio-0.9.6.md).

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

Equal names do not yet establish equal empyrical signatures or edge-case
behavior. Do not blindly replace package imports. Inventory each used symbol,
check its compatibility level, and run differential tests on production-shaped
inputs before migrating it.

The generated flat-API migration manifest records every current target,
recommended future target, and deprecation state. No switch is scheduled; any
change requires a deprecation period and a future major release.

## License/provenance review

Pyfolio's pinned checkout has MIT text in its root `LICENSE` and Apache-2.0
headers in inspected source files. Human/license review is pending. The project
does not infer a legal conclusion or generate a third-party notice until that
review decides what is required.
