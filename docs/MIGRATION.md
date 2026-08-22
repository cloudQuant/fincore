# Migrating from empyrical to fincore 0.4.0.dev0

Fincore's current package version is **0.4.0.dev0**. Do not add
`fincore>=1.0.0`: that release does not exist in the repository's current
version history.

> **Breaking change:** fincore requires **Python 3.11+**. empyrical supports
> older interpreters; upgrading an application to fincore may force an
> interpreter upgrade as well.

## The three API surfaces

Migration decisions depend on which surface a call uses. Equal function names
do not imply equal semantics across surfaces:

1. **Strict compatibility** — `fincore.empyrical` is the frozen empyrical
   0.6.0 surface (54 public symbols, 49 callables). All symbols reach **C0**,
   all callables reach **C1** (signature), and the core callables are
   numerically verified (**C3**). The class `Empyrical` exposes the same
   metrics as class methods (explicit `returns` argument) and instance methods
   (state-bound).
2. **pyfolio façade** — `fincore.pyfolio` implements the frozen pyfolio 0.9.6
   profile of 11 tear-sheet workflows. All entries reach **C1**; the
   risk/returns/perf-attrib/full-sheet main chains reach **C4** end-to-end.
   The `Pyfolio` class is enhanced OO convenience over the same workflows and
   requires the `pyfolio` extra.
3. **Enhanced semantics** — `fincore.metrics`, the flat API, and
   `AnalysisContext` are fincore's own interfaces with documented divergences
   (see below). They are the recommended API for new code, but they are not
   evidence of empyrical/pyfolio compatibility.

References:

- [Empyrical 0.6.0 matrix](compatibility/empyrical-0.6.0.md)
- [Pyfolio 0.9.6 profile](compatibility/pyfolio-0.9.6.md)
- [Upstream provenance and license-review register](upstream-provenance.md)

## Compatibility levels

Compatibility levels are C0 (public path), C1 (signature), C2 (structural and
exception behavior), C3 (numeric behavior), and C4 (cross-layer workflows).
The verified status of each frozen entry is enforced by the executable gates
in `tests/compat/` (CI job `compat`); the frozen manifests are in
`tests/compat/fixtures/`.

Verified for the empyrical surface:

| Level | Status |
| --- | --- |
| C0 | 54/54 public symbols resolve in `fincore.empyrical` |
| C1 | 49/49 callable signatures match the frozen manifest (constants: not applicable) |
| C2 | Covered for the callables exercised by the structural contract suites |
| C3 | Core callables verified numerically (CVaR, annual_volatility, cum_returns, rolling family, alignment, `out` buffers, perf-attrib) |
| C4 | Workflow level covered by the pyfolio façade chains |

Verified for the pyfolio profile:

| Level | Status |
| --- | --- |
| C0/C1 | All 11 frozen workflows resolve with the frozen signatures |
| C4 | risk, returns, perf-attrib, and full-sheet main chains run compute-plot-sheet end-to-end |

A C2/C3 result exists only for the callables the contract suites exercise; it
is not a blanket certification of every symbol. Check the matrix for your
symbols before migrating a production call.

## Installation

```bash
pip install fincore

# Optional capability extras
pip install "fincore[pyfolio]"     # Pyfolio tear sheets
pip install "fincore[interactive]" # Plotly/Bokeh backends
pip install "fincore[report-pdf]"  # PDF rendering
```

For repository development, install the checked-out 0.4.0.dev0 source:

```bash
pip install -e ".[dev,viz]"
```

## Pre-1.0 flat API policy

The existing `from fincore import ...` mappings remain unchanged throughout
the current pre-1.0 series. They point to enhanced `fincore.metrics` implementations and are not
automatically identical to empyrical signatures or edge-case behavior. The
complete generated mapping, including `current_target`, `recommended_target`,
`deprecate_in`, and `remove_or_switch_in`, is in
[`tests/compat/fixtures/fincore-flat-api-migrations.json`](../tests/compat/fixtures/fincore-flat-api-migrations.json).

No flat API deprecation is scheduled. A switch to strict-compatible targets is
only a candidate for an unscheduled future major release and requires an
explicit deprecation period first.

## Safe migration workflow

1. Pin your current empyrical version and record the symbols you use.
2. Find each symbol in the empyrical compatibility matrix and frozen JSON.
3. Prefer `fincore.empyrical` imports for empyrical-shaped calls; prefer
   `AnalysisContext`/`fincore.metrics` for new code.
4. Verify the matrix row for each production call: normally C3 for a metric
   and C4 for a report workflow.
5. Run differential tests on your own NaN, Inf, timezone, empty-input, and
   alignment cases.
6. Do not assume that an equal function name means equal signature or behavior
   across the strict and enhanced surfaces.

For code that intentionally uses the current enhanced flat API, this example
executes on fincore 0.4.0.dev0:

```python
import pandas as pd

from fincore import max_drawdown, sharpe_ratio

returns = pd.Series([0.01, -0.005, 0.002, 0.004])
print(sharpe_ratio(returns))
print(max_drawdown(returns))
```

## Enhanced fincore APIs

The following interfaces are fincore features. They are useful migration
destinations, but they are not evidence of empyrical or pyfolio compatibility.

### AnalysisContext

`AnalysisContext` groups a return series, optional benchmark, positions, and
transactions. Metrics are computed lazily and reused by export/render methods.
Inputs are defensively snapshotted, so mutating the caller's series does not
stale cached results; `replace_data()` atomically swaps inputs and invalidates
every cache entry.

```python
import pandas as pd

import fincore

index = pd.date_range("2024-01-02", periods=5, freq="B")
returns = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001], index=index)
benchmark = pd.Series([0.008, -0.003, 0.001, 0.002, 0.0], index=index)

ctx = fincore.analyze(returns, factor_returns=benchmark)
print(ctx.sharpe_ratio)
print(ctx.max_drawdown)
json_text = ctx.to_json()
ctx.to_json(path="report.json")        # write a file
ctx.replace_data(returns=returns + 0.001)
```

`ctx.to_html(path="report.html")` and `ctx.plot(backend="matplotlib")` are
enhanced output operations; `ctx.plot()` returns `ReportArtifacts`. Install
`fincore[viz]` before using visualization backends.

### Strict module access

`fincore.empyrical` is importable as a plain module and carries the frozen
0.6.0 surface:

```python
from fincore import empyrical

print(empyrical.sharpe_ratio(returns))
```

### RollingEngine

```python
from fincore.core.engine import RollingEngine

engine = RollingEngine(returns, factor_returns=benchmark, window=3)
rolling = engine.compute(["sharpe", "volatility", "max_drawdown", "beta"])
```

The metric names and result dictionary are fincore contracts, not legacy
empyrical rolling signatures.

### Pyfolio main chain

```python
from fincore import Pyfolio  # requires fincore[pyfolio]

pyfolio = Pyfolio(returns=returns, benchmark_rets=benchmark)
pyfolio.create_returns_tear_sheet(returns, benchmark_rets=benchmark)
```

### Data providers

Provider integrations are optional and may require extra packages, credentials,
network access, or provider-specific configuration:

```python
from fincore.data import YahooFinanceProvider

provider = YahooFinanceProvider()
prices = provider.fetch("AAPL", start="2024-01-01", end="2024-02-01")
```

This is an integration example, not an offline Quick Start. The provider may
raise an actionable import error when its optional SDK is absent.

### Portfolio optimization

```python
import pandas as pd

from fincore.optimization import efficient_frontier, optimize, risk_parity

asset_returns = pd.DataFrame(
    {
        "asset_a": [0.01, -0.005, 0.004, 0.002],
        "asset_b": [0.003, 0.002, -0.001, 0.005],
    }
)
frontier = efficient_frontier(asset_returns, n_points=5)
parity = risk_parity(asset_returns)
maximum_sharpe = optimize(asset_returns, objective="max_sharpe")
```

### Visualization backends

The enhanced context API accepts `matplotlib`, `html`, `plotly`, or `bokeh`
backend names when their optional dependencies are installed:

```python
figure_or_document = ctx.plot(backend="matplotlib")
```

Backend output types are fincore contracts and are not part of the pinned
pyfolio profile.

## Documented enhanced divergences

The enhanced surfaces intentionally differ from the pinned legacy semantics.
Each divergence has an executable registration in `tests/compat/`:

- **Weekly aggregation** — `fincore.metrics` offers `week_year="iso"`
  (ISO year + ISO week, one group across the 2019/2020 boundary). The strict
  `fincore.empyrical.aggregate_returns` keeps the pinned calendar-year plus
  ISO-week grouping.
- **CVaR ties** — the legacy façade keeps the pinned fixed-tail-count
  order-statistic policy; the enhanced CVaR keeps the threshold-inclusive
  expected shortfall.
- **Validation exceptions** — enhanced surfaces raise
  `ValidationError`/`NumericalError`/`DataAlignmentError` instead of silently
  tolerating invalid input.
- **Alignment/timezone policy** — enhanced binary metrics default to strict
  label alignment, reject mixed naive/aware indices, and support explicit
  `normalize_tz`; the strict façade keeps pinned legacy alignment and
  exception surfaces.
- **`print_table`/`run_flask_app`** — legacy `run_flask_app` parameters are
  accepted but display-only and never write files implicitly; the enhanced
  `export=` keyword requires a caller-owned destination.

## Frequently asked questions

### Is fincore a drop-in replacement for empyrical?

The frozen surface is verified at C0/C1 for every symbol, with C3 numeric
verification for the core callables, and the pyfolio main chains at C4. That
is strong evidence for the verified callables, but C2/C3 is not claimed for
every symbol. Migrate symbol by symbol after checking the matrix row.

### Can empyrical and fincore be installed together?

Yes. Keeping both during differential testing is often useful. Use explicit
module imports so the implementation under test is unambiguous.

### Which Python versions are supported?

Current package metadata requires Python 3.11 or newer. Validate the exact
wheel/environment you plan to deploy.

### Where should compatibility bugs be reported?

Open an issue with the pinned upstream version, the fincore version, a minimal
input, actual output, expected output, and whether the discrepancy concerns
C0, C1, C2, C3, or C4.

## Known migration boundaries

- `fincore.empyrical` is the strict-compatibility surface; nine rolling
  callables created by upstream factories carry `needs_dynamic_review=true`
  until an isolated oracle run is reviewed by a person.
- `fincore.metrics` and the pre-1.0 flat API are enhanced surfaces with
  semver-managed differences.
- The pyfolio profile covers only the 11 listed functional workflows, not the
  entire upstream package or the enhanced `Pyfolio` class.
- Compatibility workflows intentionally must not write into the installed
  package directory. This safety difference is tracked rather than hidden.
- Upstream pyfolio license metadata is inconsistent: its root `LICENSE` has MIT
  text while inspected source files have Apache-2.0 headers. Human/license
  review remains required; this guide makes no legal determination.

## Existing applications

Do not perform a blind replacement such as `import empyrical` to
`import fincore`. First add differential tests, then migrate individual imports
only after the relevant matrix rows reach the required level. `AnalysisContext`,
`RollingEngine`, optimization, and visualization APIs are fincore features, not
empyrical compatibility guarantees.

## Migration from Alphalens

The Alphalens integration has two intentionally separate choices. The strict
namespace is for source-shaped migrations; the enhanced namespace is for new
work that benefits from explicit preparation, immutable analysis models, and
caller-owned plot artifacts. Neither choice installs a standalone package, so
top-level `import alphalens` is unsupported.

| Standalone | Fincore strict | Fincore enhanced |
| --- | --- | --- |
| `alphalens.utils.get_clean_factor_and_forward_returns` | `fincore.alphalens.utils.get_clean_factor_and_forward_returns` | `fincore.factor_analysis.prepare_factor_data` |
| `alphalens.performance.factor_information_coefficient` | `fincore.alphalens.performance.factor_information_coefficient` | `fincore.factor_analysis.factor_information_coefficient` |
| `alphalens.tears.create_full_tear_sheet` | `fincore.alphalens.tears.create_full_tear_sheet` | `analyze_factor` + `render_full_tear_sheet` |
| `create_pyfolio_input` + external pyfolio | strict tuple + `fincore.pyfolio` | `PyfolioFactorInputs` |

For the enhanced full-sheet row, `render_full_tear_sheet` means
`fincore.factor_analysis.create_full_tear_sheet(model)`: first create the
model with `analyze_factor`, then render, inspect, show, or close the returned
artifacts explicitly. The enhanced Pyfolio bridge returns
`PyfolioFactorInputs`; it does not import or invoke external Pyfolio.

### Offline quickstart

Install the rendering extra and run the checked executable example:

```bash
pip install "fincore[alphalens]"
MPLBACKEND=Agg python examples/factor_analysis_quickstart.py
```

The example uses fixed-seed local synthetic data, has no network path and no
default output-file path. It demonstrates the strict quickstart, enhanced
`prepare_factor_data` plus `analyze_factor`, the Pyfolio bridge, and an Agg
summary tear sheet whose figures are closed after use. If the rendering stack
is absent, its `DependencyError` names the exact repair:

```text
pip install fincore[alphalens]
```

### Behavioral differences and migration checks

- The strict façade follows the pinned cloudQuant-local source snapshot at
  commit `3fa17ad4c3edb025d1410de7aeba9673cba7791c`; the source's `v0.4.0`
  Versioneer string and `setup.py` fallback `1.0.0+dev` are conflicting
  evidence, not release identities. Pin and report the commit, not either
  version string.
- `filter_zscore=20` is a source-shaped default on the strict cleanup call,
  but it filters using the full forward-return distribution and can introduce
  look-ahead bias. Prefer `filter_zscore=None` unless a research protocol
  explicitly justifies the filter; the executable example does so.
- Factor and price timestamps must use compatible timezones. Preserve the
  source calendar/frequency and test sessions, holidays, and any naive/aware
  conversion rather than assuming a daily calendar.
- `max_loss` is an acceptance threshold, not a cosmetic warning. Inspect the
  enhanced `PreparedFactorData.loss_report` and choose a research-justified
  limit before increasing it.
- Use `fincore[factor-analysis]` for compute-only enhanced workflows. Use
  `fincore[alphalens]` when rendering or migrating strict Alphalens calls;
  `fincore[pyfolio]` remains the separate extra for Pyfolio tear sheets.
- The first integration has no notebook, HTML-report, or interactive-backend
  promise. It is not a claim of full standalone Alphalens compatibility.

Compatibility wording is deliberately limited to the executable strict-path,
signature, kernel, and workflow tests currently present in this repository.
See `docs/compatibility/alphalens-0.4.0-cloudquant.md` for the pinned source
identity and scope. The human license/NOTICE decision remains an advisory
follow-up, not a CI/CD gate or automated legal approval; this guide makes no
legal conclusion.
