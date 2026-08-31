# `fincore.attribution`

Attribution capabilities are direct kernels grouped by analytical method:

- `brinson`: allocation and selection attribution;
- `fama_french`: factor-model attribution;
- `style`: style analysis;
- `performance`: portfolio performance attribution;
- `operations`: direct operation registration.

```python
from fincore.attribution.performance import perf_attrib
```

Inputs and outputs retain labelled pandas structures so reporting and downstream
analysis can reconcile them without a façade-specific object.
