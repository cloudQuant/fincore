# Compatibility

This page describes the C0–C4 compatibility status of fincore 0.3.0 against
its three frozen upstream targets: empyrical 0.6.0, pyfolio 0.9.6, and the
cloudQuant-local Alphalens snapshot. Everything claimed here is enforced by
executable gates in `tests/compat/` (CI jobs `compat` and `compat-alphalens`)
against the frozen manifests in `tests/compat/fixtures/` — nothing is asserted
by hand.

## Compatibility levels

| Level | Meaning | Enforced by |
| --- | --- | --- |
| C0 | The public path resolves in fincore | `test_public_api.py` suites |
| C1 | Parameter name, order, kind, and default match the frozen signature | `test_signatures.py`, `test_public_api.py` |
| C2 | Input immutability, type/shape/index/dtype, and exception surface match | structural contract suites |
| C3 | Numeric, NaN/Inf, timezone, and boundary behavior match | numeric contract suites |
| C4 | Cross-layer workflow and output contract match | end-to-end chain suites |

## Empyrical 0.6.0 — `fincore.empyrical`

Frozen upstream: empyrical **0.6.0** at commit
`74655e974ed2935563820c548c339731f1fe0621`.

Machine-readable source of truth:
[`tests/compat/fixtures/empyrical-0.6.0-api.json`](https://github.com/cloudQuant/fincore/blob/master/tests/compat/fixtures/empyrical-0.6.0-api.json).

| Surface | Status |
| --- | --- |
| Public symbols | **54/54 C0** — every frozen symbol resolves in `fincore.empyrical` |
| Callables | **49/49 C1** — every frozen signature matches (constants: not applicable) |
| Core callables | **C3** — numeric contracts for the CVaR family, `annual_volatility`, `cum_returns`, the rolling family, `out` buffers, alignment, and perf-attrib |
| Rolling factory callables | Nine callables created by upstream factories carry `needs_dynamic_review=true` in the manifest until an isolated oracle run is reviewed by a person |

C2/C3 is claimed only for the callables the contract suites exercise; it is
not a blanket certification of all 49 callables. Check the frozen manifest for
any symbol you migrate.

```python
import pandas as pd

from fincore import empyrical

returns = pd.Series([0.01, -0.005, 0.002, 0.004])
empyrical.sharpe_ratio(returns)
```

## Pyfolio 0.9.6 profile — `fincore.pyfolio`

Frozen upstream: pyfolio **0.9.6** at commit
`724bbd7dbed9a88bb47e1057f2ca29b3409d8e7a` — a bounded profile of 11
tear-sheet workflows, not the entire upstream package.

Machine-readable source of truth:
[`tests/compat/fixtures/pyfolio-0.9.6-api.json`](https://github.com/cloudQuant/fincore/blob/master/tests/compat/fixtures/pyfolio-0.9.6-api.json).

| Workflow | C1 | C4 main chain |
| --- | --- | --- |
| `create_full_tear_sheet` | verified | verified (real subsheet chain) |
| `create_returns_tear_sheet` | verified | verified |
| `create_risk_tear_sheet` | verified | verified |
| `create_perf_attrib_tear_sheet` | verified | verified |
| `create_position_tear_sheet` | verified | — |
| `create_txn_tear_sheet` | verified | — |
| `create_round_trip_tear_sheet` | verified | — |
| `create_interesting_times_tear_sheet` | verified | — |
| `create_capacity_tear_sheet` | verified | — |
| `create_bayesian_tear_sheet` | verified | — |
| `create_simple_tear_sheet` | verified | — |

`from fincore import Pyfolio` requires the `pyfolio` extra
(`pip install fincore[pyfolio]`) and raises `DependencyError` otherwise. The
`Pyfolio` class is enhanced OO convenience driven by the same workflows; its
workflow methods keep the frozen signatures.

```python
import pandas as pd

from fincore import Pyfolio  # requires fincore[pyfolio]

index = pd.date_range("2024-01-02", periods=5, freq="B")
returns = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001], index=index)
benchmark = pd.Series([0.008, -0.003, 0.001, 0.002, 0.0], index=index)

pyfolio = Pyfolio(returns=returns, benchmark_rets=benchmark)
pyfolio.create_returns_tear_sheet(returns, benchmark_rets=benchmark)
```

## Alphalens migration — strict and enhanced routes

The pinned cloudQuant-local Alphalens source identity is commit
`3fa17ad4c3edb025d1410de7aeba9673cba7791c`; its historical `v0.4.0` and
`1.0.0+dev` strings are conflicting evidence, not release identities.
`fincore.alphalens` is the strict source-shaped façade, while
`fincore.factor_analysis` is the enhanced prepare/analyze/render API.

The documented boundary is only what current executable tests cover: strict
public paths and signatures plus targeted enhanced kernels and workflows. It
is not a full standalone compatibility claim. No top-level `import alphalens`,
notebook/HTML, or interactive-backend workflow is supported in this first
integration. Use `fincore[factor-analysis]` for compute-only analysis and
`fincore[alphalens]` for rendering or strict migration calls. The human
license/NOTICE decision remains a release blocker.

| Surface | Status |
| --- | --- |
| Public definitions | **64/64 C0** — every frozen symbol resolves in `fincore.alphalens` |
| Callables / constructors | **C1** — frozen signatures, hidden `set_context` call grammar, dual `quantize_factor` signature |
| utils kernels (forward returns, cleaning, quantization) | **C2/C3** — 36/36 pinned upstream `test_utils.py` source cases rewritten with `pd.testing` assertions |
| performance kernels (IC, weights, returns, turnover, events, alpha/beta) | **C2/C3** — 81/81 pinned upstream `test_performance.py` source cases |
| plotting API | **21/21** with structural/data assertions |
| tear sheets | **7/7 C4** real compute → model → render → sheet chains; 24/24 dormant upstream rows and 96/96 invocations mapped 1:1 |
| Pyfolio bridge | `create_pyfolio_input` output runs the real `fincore.pyfolio` workflow |

## Intentional divergence registrations

The enhanced surfaces (`fincore.metrics`, flat API, `AnalysisContext`) may
differ from the pinned legacy semantics **by design**. Every registered
divergence has an executable test — in `tests/compat/` unless noted in the
table:

| Divergence | Legacy (strict façade) | Enhanced (opt-in) |
| --- | --- | --- |
| Weekly aggregation | `aggregate_returns(..., "weekly")` keeps the pinned calendar-year + ISO-week grouping (two groups across the 2019/2020 ISO boundary) | `week_year="iso"` uses ISO year + ISO week (one group) |
| CVaR quantile ties | Fixed tail-count order-statistic policy | Threshold-inclusive expected shortfall |
| `print_table` export | Display-only; `run_flask_app` accepted but never writes files implicitly | Keyword-only `export=` with a caller-owned XLSX destination recorded on `ReportArtifacts` (test: `tests/test_utils/test_export_destination.py`) |
| `run_flask_app` | Retained for pinned legacy callers; rendering stays display-only | — |
| Timezone/alignment policy | Pinned legacy alignment and exception surfaces | Strict identical-label alignment, mixed naive/aware rejected by default, explicit `normalize_tz` opt-in |
| Input validation | Legacy NaN tolerance where pinned | `ValidationError` / `NumericalError` / `DataAlignmentError` fail fast |

## How the gates enforce this

- Frozen JSON manifests: `tests/compat/fixtures/empyrical-0.6.0-api.json`,
  `pyfolio-0.9.6-api.json`, `alphalens-0.4.0-cloudquant-api.json`,
  `fincore-flat-api-migrations.json` — generated by
  `scripts/generate_compat_manifest.py` from the pinned upstream commits.
- Executable suites: `tests/compat/empyrical/`, `tests/compat/pyfolio/`, and
  `tests/compat/alphalens/` plus `tests/test_factor_analysis/`.
- Pinned upstream-test migration audit:
  `scripts/check_alphalens_upstream_test_migration.py` against
  `tests/compat/fixtures/alphalens-0.4.0-cloudquant-upstream-test-{inventory,migration}.json`.
- Manifest integrity: `tests/compat/test_manifest_integrity.py`.
- CI jobs: `compat` and `compat-alphalens` in `.github/workflows/ci.yml` (both
  are release-blocking inputs of the `build` job and the publish gate).

Run locally:

```bash
pytest -o addopts='' tests/compat -q --maxfail=0

MPLBACKEND=Agg pytest -o addopts='' tests/compat/alphalens tests/test_factor_analysis -q --maxfail=0
```

The internal matrices
([empyrical-0.6.0.md](https://github.com/cloudQuant/fincore/blob/master/docs/compatibility/empyrical-0.6.0.md),
[pyfolio-0.9.6.md](https://github.com/cloudQuant/fincore/blob/master/docs/compatibility/pyfolio-0.9.6.md))
describe the frozen target and regeneration procedure in more detail.
