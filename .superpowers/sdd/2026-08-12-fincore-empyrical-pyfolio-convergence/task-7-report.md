# Task 7 Report: unified validation and AnalysisContext contracts

## Outcome

Task 7 establishes one registry-selected validation boundary for enhanced
metrics and one immutable lifecycle boundary for analysis/report inputs.
Strict Empyrical and Pyfolio compatibility paths deliberately remain outside
that boundary and continue to execute the frozen upstream oracle.

- `fincore.contracts.validation` defines defensive schemas for returns,
  positions, transactions, factors, and price/volume market data. Enhanced
  returns reject empty, non-numeric, non-finite, unsorted, or duplicate input;
  canonical portfolio schemas enforce timestamp, cash, required-column, and
  overlap contracts. Nullable Pandas numeric dtypes classify `pd.NA` as the
  same `NumericalError` finite-value failure as NumPy `NaN`/`Inf`.
- Timezone validation includes the datetime level of stacked MultiIndexes.
  `AnalysisContext` explicitly accepts wide positions only, so a stacked panel
  cannot silently bypass overlap validation; callers may normalize supported
  time-indexed inputs explicitly to UTC.
- Enhanced flat functions, enhanced `Empyrical` methods, and direct
  `fincore.metrics.*` imports resolve exact `MetricSpec` entries and bind both
  positional and keyword arguments before validation. Binary metrics align
  the retained values before finite checking, preserving the reviewed Task 4
  alignment/error ordering. Internal kernel composition and strict wrappers
  enter a context-local raw guard, not a process-global bypass.
- Direct metrics modules expose signature-preserving enhanced wrappers while
  their module globals retain raw functions for internal composition. Import
  order, monkeypatch restoration, aliases, and module reload are covered.
  Wrapper caching now includes the resolved kernel and adapter identities, so
  reload cannot retain an old kernel; materialized aliases such as `cagr`
  cannot fall back to an unvalidated raw function.
- `AnalysisContext` takes defensive copies, validates replacement inputs
  before mutating state, invalidates all cached properties atomically, and
  computes alpha/beta through one shared `alpha_beta` invocation. Positions
  and transactions contribute cached gross-leverage and turnover outputs.
- Report computation constructs one context, consumes its canonical snapshot,
  and reuses its leverage and turnover values. `plot()` returns a
  backend-neutral `ReportArtifacts`; `to_json(path=...)` writes the exact
  returned payload.
- `ReportArtifacts.close()` deduplicates axes belonging to one figure,
  attempts every owned resource, records successful closes, retries only
  failures, is idempotent, and supports context-manager cleanup.

## TDD evidence

The initial Task 7 contract matrix was written before implementation and
reported:

```text
151 failed, 3 passed
```

The first schema slice then reached `19 passed`. Subsequent dispatch/context
work exposed raw-oracle ownership and cross-module import-order failures. The
bounded dependency selector ultimately reached:

```text
1625 passed, 3 pinned Pandas warnings
```

During final edge review, module reload proved two related bugs. A reloaded
`yearly.cagr` retained a raw alias, and the registry-key-only callable cache
retained the prior module's kernel. The real reload test was RED before each
minimal fix and now proves that the public wrapper's `__wrapped__` object is
the current module kernel and that invalid input still reaches enhanced
validation:

```text
reload alias/kernel identity: 1 failed -> 1 passed
```

The final schema pressure test added nullable numeric data and stacked
MultiIndex timezone/context cases. Before implementation it reported three
failures and one already-correct control; after the fixes:

```text
nullable + stacked timezone/context matrix: 4 passed
```

The stale drawdown-composition assertion also failed against the renamed raw
`_cum_returns` reference, then passed after the test was migrated to prove the
actual raw callable rather than a removed private name.

## Fresh acceptance evidence

Final Task 7 and cross-task gates on the staged implementation are:

```text
tests/contracts: 232 passed
tests/contracts + tests/test_core + tests/test_report: 351 passed
Task 3 strict public/signature/state/out selector: 203 passed
Task 4 five compatibility modules: 260 passed, 3 pinned Pandas warnings
Task 5 domain plus manifest selector: 114 passed
Task 6 exact Pyfolio selector: 354 passed, 9 expected business warnings
```

The three Task 4 warnings are the frozen outer-concat behavior. The nine
Task 6 warnings are the existing named-interesting-period warning for
non-overlapping synthetic returns.

Static gates for all Task 7-owned Python paths are clean:

```text
scoped mypy --follow-imports=skip: Success, 13 source files
scoped ruff check: All checks passed
scoped ruff format --check: 37 files already formatted
git diff --check: clean
```

The import-following mypy audit also exposed unrelated transitive baseline
errors in optimization, visualization, positions, and the existing Pyfolio
wrapper. Those are not hidden by configuration changes and remain assigned to
the plan's Task 12 typed batches; the explicitly owned Task 7 scope is zero.

## Scope and handoff

The controlled migration fixture changed only its recorded source digest; the
entry list, targets, and semantic policies are unchanged. Raw numerical tests
under `tests/test_empyrical/stats` and `tests/test_metrics` explicitly declare
kernel-oracle ownership, while all enhanced public validation behavior lives
under `tests/contracts`. One Pyfolio attribution warning test uses a single
explicit raw metric resolver rather than weakening the entire Pyfolio subtree.

Task 8 may build the compute-once report model and offline renderers on the
`ReportArtifacts` lifecycle introduced here. Task 12 retains ownership of the
full-package type-error baseline and must rerun it after Tasks 8–11.
