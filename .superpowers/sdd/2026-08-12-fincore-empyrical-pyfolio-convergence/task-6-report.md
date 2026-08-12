# Task 6 Report: Pyfolio workflow convergence

## Outcome

Task 6 restores the frozen Pyfolio 0.9.6 functional workflow surface and
closes the performance-attribution, drawdown, and full tear-sheet paths.

- `fincore.pyfolio` now exposes all 11 frozen module-level
  `create_*_tear_sheet` functions with their exact pinned signatures.  The
  wrappers bind arguments before delegation, resolve the workflow registry
  lazily, and keep the pinned return projections: full and simple return
  `None`; the remaining workflows preserve the underlying `run_flask_app`
  result.
- `WorkflowSpec` records the independently versioned surface, public name,
  signature key, lazy workflow and adapter references, validation profile,
  result contract, and projection.  The strict registry is keyed by
  `(surface, public_name, variant)` and contains all 11 workflows.
- The former public implementation moved to `fincore/_pyfolio_impl.py`.
  Ordinary import and signature introspection do not load it, plotting
  modules, or optional scientific dependencies, and do not change the active
  Matplotlib backend.  Explicit `Pyfolio` resolution and the first workflow
  call load the implementation without changing the backend.
- The public full workflow retains its pinned explicit `set_context`
  parameter and does not expose `run_flask_app`.  It is not wrapped by the
  context decorator, so the caller's `set_context` value reaches each child
  workflow.  The private Flask/server path is never used.
- Wide performance-attribution positions once again support
  `stack_positions` and `pos_in_dollars`.  Dollar positions use the pinned net
  asset denominator including cash before cash is removed.  Already-stacked
  input is not silently renormalized.  The enhanced `regression_style`
  parameter now accepts the implemented `"OLS"` mode and rejects unsupported
  values instead of ignoring them.
- The ten-row padded drawdown table remains compatible, while plotting skips
  `NaT` padding rows.  Cumulative-return data and peak/recovery coordinates
  are copied and normalized to one UTC-naive plotting representation, leaving
  caller indexes untouched and eliminating the prior mixed-timezone warning.
- The two affected box plots use the installed Matplotlib
  `orientation="horizontal"` API.  The former 11 timezone deprecations and 2
  box-plot pending deprecations are absent from the complete Pyfolio suite.
- Display helpers retain the public `HAS_IPYTHON`, `display`, and `HTML`
  contracts but load `IPython.display` only on use.  A top-level availability
  probe avoids importing the IPython parent package during module import.
- Plot helpers that receive an explicit local series now bypass stored-state
  rebinding for real `Empyrical`/`Pyfolio` instances through class-level metric
  dispatch.  Duck-typed test or extension objects still use their instance
  methods.  This closes the real `Pyfolio(returns=...)` attribution and
  slippage workflows without weakening the reviewed Task 3 binding contract.
- Previously inert `.equals()` expressions in the requested attribution,
  transaction, and round-trip tests are real pandas assertions.  Tests coupled
  to the former private public-module globals now patch the private
  implementation directly; those heavy helpers are not re-exported.

## Lazy and return-contract boundary

The isolated import contract covers ordinary `import fincore.pyfolio` plus
introspection of all 11 workflow signatures.  Explicit
`from fincore.pyfolio import Pyfolio` is also backend-neutral but intentionally
loads the private plotting implementation.  `Pyfolio` remains in `__all__` for
the existing public export contract, so star import also resolves the class;
whether a future core wheel should change that semver boundary is deferred to
Task 11.

The strict façade does not expose enhanced `ReportResult` or export-directory
options.  `run_flask_app=True` returns the pinned in-memory Figure where the
upstream workflow does so, and the no-write test proves that the workflow does
not create or mutate package files.  Enhanced result/export lifecycle work
remains assigned to Task 8.

## TDD evidence

The requested assertions were strengthened before production changes.  The
first exact run was:

```text
four legacy assertion modules: 4 failed, 9 passed
```

The initial compatibility RED groups were:

```text
public API, lazy import, workflow registry, drawdown: 26 failed, 2 passed
performance attribution, full workflow, no-write: 7 failed
```

The failures covered all 11 missing module workflows and delegation paths,
the absent workflow contract, eager backend/dependency import, missing wide
position controls, ignored regression mode, padded `NaT` plotting, full-chain
return behavior, and package-write protection.

Broad testing then exposed the reviewed stored-state/local-series integration
case.  Two real `Pyfolio(returns=...)` regressions were added first:

```text
stored-state attribution and slippage plots: 2 failed
```

The explicit import probe also caught a dotted-spec side effect before its
fix:

```text
common_utils isolated import: 1 failed (117 IPython modules loaded)
```

Focused results after implementation are:

```text
frozen public API and lazy boundary: 28 passed
performance attribution compatibility: 4 passed
drawdown compatibility: 2 passed
full/returns workflow and stored-state chain: 5 passed
no source writes: 1 passed
warning-as-error timezone/box-plot gate: 4 passed
complete tests/test_pyfolio: 90 passed, 9 expected business warnings
```

The nine remaining warnings are the explicit Pyfolio business warning for
returns that do not overlap a named interesting period.  The real Task 6 full
chain uses overlapping data and emits none of those warnings.

### Independent-review follow-up

The first independent review found two P1 boundary defects.  Regression tests
were added before either production fix; the exact review RED was:

```text
DST drawdown + optional-dependency boundary: 3 failed, 1 passed
```

The DST case uses exact `America/Sao_Paulo` timestamps spanning the skipped
midnight on 2018-11-04.  Reconstructing the date-only compatibility table
value raised a nonexistent-time `ValueError`.  Drawdown shading now consumes
the exact peak/valley/recovery timestamps from `get_top_drawdowns`, converts
those instants to a UTC-naive plotting copy, preserves unrecovered-period
fallback and color cardinality, and leaves the ten-row table unchanged.

The optional-dependency case proved both resolution-time and call-time
failures.  `invoke_workflow` now translates recognized `ModuleNotFoundError`
instances across the complete resolve-and-call boundary.  Real isolated
subprocess blockers verify implementation-import failure for `matplotlib` and
Bayesian call-time failure for `pymc`; the actionable installation commands
are the actual published extras `fincore[viz]` and
`fincore[viz,bayesian]`.  An unrelated internal missing module remains the
original `ModuleNotFoundError` object.

Review-focused and final impact results are:

```text
four review cases after minimal fixes: 4 passed
isolated dependency + public/drawdown + legacy dummy: 36 passed
Task 6 impact selector: 888 passed, 9 expected business warnings
```

### Second-review follow-up

The second independent review found one empty-input P1 and requested a P2
strengthening of the DST regression.  Tests were changed before production;
after correcting the endpoint assertion to use the axis' active date units,
the exact RED was:

```text
empty drawdown + exact shaded endpoints: 1 failed, 1 passed
```

`plot_drawdown_periods` now returns its correctly titled and labelled axis
before metric evaluation when given an empty returns series.  The result has
no shading or warnings and the caller's empty `DatetimeIndex` is unchanged.
The Sao Paulo regression also asserts that the shaded polygon begins and ends
at the exact UTC-naive peak and recovery instants, rather than proving only
that plotting does not crash.

Fresh second-review gates are:

```text
empty drawdown + exact shaded endpoints: 2 passed
public/drawdown/legacy focused selector: 37 passed
Task 6 impact selector: 889 passed, 9 expected business warnings
```

## Regression and migration gates

The Task 6 impact selector includes all Pyfolio compatibility tests, the full
legacy Pyfolio and tear-sheet suites, and complete metrics and utility suites.
Its final authoritative result is recorded after the last import-boundary
change in the progress ledger.

Cross-task gates are green:

```text
Task 4 context-impact selector: 291 passed, 3 pinned Pandas warnings
Task 5 manifest/domain selector: 159 passed
Task 3 frozen public selector: 203 passed
manifest integrity/idempotence: 26 passed
```

The three Task 4 warnings are the documented pinned outer-concat behavior and
are unrelated to Task 6.  An AST move audit compared
`816e128:fincore/pyfolio.py` with the private implementation: both contain 69
`Pyfolio` methods, with no missing or added method names.

## Scope and handoff

The implementation includes the planned workflow, attribution, drawdown, and
public/private façade files; the approved lazy display boundary; the approved
stored-state plot dispatch in attribution and transaction tear sheets; and
the directly affected legacy and compatibility tests.  Task 4 and Task 5
contracts were not changed.  Task 6 is awaiting final independent re-review
after the second-review follow-up.
