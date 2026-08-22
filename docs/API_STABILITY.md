# API Stability Policy

This document describes the API stability guarantees for fincore 0.4.0.dev0.

Stability is claimed **only** for surfaces whose compatibility level (C0-C4)
has been verified by the executable gates in `tests/compat/`. A surface not
listed below, or listed with a partial level, carries no broader guarantee.

## Stable surfaces

### Top-level imports

```python
from fincore import (
    Empyrical,
    Pyfolio,                # requires the pyfolio extra
    analyze,
    create_strategy_report,
)
```

`from fincore import Pyfolio` raises `DependencyError` naming
`pip install fincore[pyfolio]` when the extra is absent.

### Strict compatibility module — `fincore.empyrical`

The frozen empyrical 0.6.0 surface is stable at the verified levels:

- C0: all 54 public symbols;
- C1: all 49 callable signatures (constants: not applicable);
- C3: the core callables covered by the numeric contract suites.

The nine rolling callables created by upstream factories remain flagged
`needs_dynamic_review=true` in the frozen manifest until an isolated oracle
run is reviewed by a person. C2/C3 is not claimed for symbols the contract
suites do not exercise.

### pyfolio façade — `fincore.pyfolio`

The frozen pyfolio 0.9.6 profile of 11 workflows is stable at C1 for every
entry and C4 for the risk/returns/perf-attrib/full-sheet main chains. The
`Pyfolio` class is enhanced OO convenience over the same workflows; its
workflow methods keep the frozen signatures.

### Flat API functions

The flat API (`from fincore import ...`) is stable within the current pre-1.0 series **as an
enhanced surface** — bound to `fincore.metrics` implementations, not to
empyrical equality:

```python
from fincore import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    annual_return,
    annual_volatility,
    cum_returns,
    cum_returns_final,
    alpha,
    beta,
    alpha_beta,
    calmar_ratio,
    omega_ratio,
    information_ratio,
    stability_of_timeseries,
    tail_ratio,
    value_at_risk,
    capture,
    downside_risk,
    simple_returns,
    aggregate_returns,
)
```

The complete generated mapping (including `current_target`,
`recommended_target`, `deprecate_in`, `remove_or_switch_in`) lives in
`tests/compat/fixtures/fincore-flat-api-migrations.json`.

### AnalysisContext

The `AnalysisContext` class and its public methods are stable:

```python
import fincore

ctx = fincore.analyze(returns, factor_returns=benchmark)
ctx.sharpe_ratio
ctx.max_drawdown
ctx.perf_stats()
ctx.to_dict()
ctx.to_json()                 # text payload
ctx.to_json(path="report.json")
ctx.to_html(path="report.html")
ctx.plot(backend="matplotlib")  # -> ReportArtifacts
ctx.replace_data(returns=new_returns)  # atomic swap + cache invalidation
```

### RollingEngine

The `RollingEngine` class is stable:

```python
from fincore.core.engine import RollingEngine

engine = RollingEngine(returns, factor_returns=benchmark, window=60)
engine.compute(['sharpe', 'volatility', 'max_drawdown', 'beta'])
```

## In-development Alphalens migration surfaces

`fincore.alphalens` is the strict, source-shaped compatibility namespace for
the pinned cloudQuant-local Alphalens snapshot at commit
`3fa17ad4c3edb025d1410de7aeba9673cba7791c`. The separate
`fincore.factor_analysis` namespace is the enhanced API for new code:
prepare data once, analyze it once, then render explicitly owned artifacts.

These surfaces are **Beta and not Stable**. The strict public-path and
signature checks and the enhanced kernel/workflow tests define the only
current compatibility claims; they do not certify the entire standalone
Alphalens package. In particular, fincore does not support top-level
`import alphalens`, a notebook/HTML workflow, or an interactive rendering
backend in this first integration. Install `fincore[factor-analysis]` for
compute-only enhanced analysis and `fincore[alphalens]` for plotting or strict
Alphalens migration workflows.

The historical source reports conflicting version evidence (`v0.4.0` in
Versioneer and `1.0.0+dev` in `setup.py`); the full commit above is the only
identity used by this policy. The pending human license/NOTICE decision is a
release blocker.

## Not covered by this policy

- `Empyrical`/`Pyfolio` methods beyond the frozen verified surface;
- equality between enhanced `fincore.metrics` behavior and empyrical behavior
  (documented divergences exist by design);
- broad standalone Alphalens equivalence beyond the executable tests attached
  to the strict façade and enhanced workflow;
- modules and functions prefixed with `_` (internal, may change without
  notice):
  ```python
  from fincore import _registry  # Internal, may change
  from fincore.metrics import _basic  # Internal, may change
  ```

## Versioning Policy

- **Major (X.0.0)**: Breaking changes to stable APIs
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, backward compatible

## Deprecation Process

If a stable API needs to be changed:

1. The old API will be marked as deprecated in documentation
2. A warning will be added (if applicable)
3. The old API will remain functional for at least one minor version
4. The old API will be removed in a major version update

## Python Version Support

fincore requires **Python 3.11+** — a documented breaking change relative to
empyrical. Currently exercised versions:

- Python 3.11
- Python 3.12
- Python 3.13

Unsupported Python versions may be removed in a major version update.

## Third-Party Dependencies

Third-party dependencies are considered part of the stable API. Changes to
required dependencies will only occur in minor or major versions, not patch
versions.

Optional dependencies (the functional extras: `pyfolio`, `interactive`,
`report-pdf`, `report-xlsx`, `bayesian`, `data-*`) may have their version
requirements updated in patch versions. The pre-1.0 compatibility aliases `viz` and
`datareader` are retained for at least one documented minor cycle.

## Feedback

If you find an API that is incorrectly labeled (should be stable or internal), please
open an issue on GitHub.

## Questions?

For questions about API stability, please open a discussion on GitHub.
