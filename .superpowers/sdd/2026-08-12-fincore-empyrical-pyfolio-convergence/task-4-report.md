# Task 4 Report: numeric, rolling, and time-series semantics

## Outcome

Task 4 converges the pinned empyrical 0.6.0 numeric and rolling result
contracts without changing the enhanced metrics kernels globally.

- Strict CVaR uses the upstream fixed-count order-statistics tail; the tie case
  `[-.2, -.1, -.1, -.1, 1]` at `cutoff=.25` is `-.15`. The enhanced direct
  metric retains its threshold-inclusive `-.125` result.
- Strict factory-generated `roll_*` functions use `min(len, window)`, preserve
  ndarray/Series shape and labels, and mutate supplied `out` buffers. The
  independently implemented capture family retains its pinned empty short
  result, while enhanced `rolling_*` APIs retain full pandas-shaped output.
- The shared enhanced alignment contract makes `strict`, `inner`, and
  `outer_dropna` policies explicit, rejects duplicate labels, rejects mixed
  timezone awareness by default, and permits explicit UTC normalization.
  Legacy metric alignment remains unchanged.
- Strict weekly aggregation keeps calendar year plus ISO week. The enhanced
  `week_year="iso"` option uses ISO year plus ISO week; the intentional
  divergence is recorded in `docs/compatibility/empyrical-0.6.0.md`.
- Performance attribution intersects by actual date labels and never assigns
  an index merely because lengths match. Each output day satisfies
  `total_returns = common_returns + specific_returns`.
- `AnalysisContext` only applies the Task 4 timezone contract. With no
  normalization it preserves existing input identity and partial-index
  behavior; defensive copies and general strict schema validation remain
  outside this task.

Strict result projection is implemented in the strict façade adapters in
`fincore/_registry.py` and `fincore/empyrical.py`. Consequently the enhanced
`risk.py`, `rolling.py`, and shared `basic.py` kernels required no behavioral
change.

## TDD evidence

The four requested test modules were added before production edits. The exact
initial run produced:

```text
34 failed, 29 passed in 0.60s
```

The failures grouped as 5 numeric, 12 rolling, 11 index/timezone, and 6
attribution contracts. From the first run, strict legacy weekly grouping was
already two groups and capture-family short windows were already empty.

After the first context implementation copied and strictly aligned all inputs,
the identity plus partial-index check was captured as a focused two-test RED:

```text
2 failed in 1.30s
```

The implementation was then narrowed to timezone validation/normalization.
Final focused results are:

```text
four Task 4 compatibility modules: 64 passed in 0.65s
four Task 4 modules + complete context suite: 95 passed in 1.24s
```

## Provenance and manifest evidence

The generator now freezes the pinned `empyrical/utils.py` blob because its
rolling factory logic is numerical C2/C3 evidence. The manifest test resolves
that blob through the generator's bounded, noninteractive `PinnedGitSource`;
it does not introduce an unbounded test subprocess.

Regeneration added only `utils.py` with SHA256
`aff1a9d686b576ad971e7985b22a24f0460100a90e4cb2ab6c7b7f8ca6dc76d9`.
All 49 callable signatures are unchanged. All 54 symbol review flags and all
oracle review flags remain `false`; evidence keys were naturally recomputed
after the source-file evidence set changed. The generator idempotence and
integrity suite is green:

```text
tests/compat/test_manifest_integrity.py: 26 passed in 3.75s
```

## Regression gates

The Task 4 brief impact gate, excluding only the concurrently owned Task 5
file `tests/test_metrics/test_positions_metrics.py`, is green:

```text
tests/compat/empyrical + tests/test_empyrical/stats + tests/test_metrics:
1088 passed in 2.44s
```

The unexcluded run is expected to contain two Task 5 RED tests:
`test_get_long_short_pos_returns_normalized_long_short_and_net_exposure` and
`test_get_long_short_notional_keeps_the_previous_amount_summary`. Task 4 did
not change or stage that file.

## Deferred integration ledger

`tests/integration/test_workflows.py` currently has 3 failures and 12 passes:

- `TestCompleteAnalysisWorkflow::test_workflow_with_empyrical_class` passes a
  Series positionally to a stored-state `max_drawdown` descriptor; under the
  reviewed Task 3 contract it binds to `out` and raises `KeyError`.
- `TestDataConsistencyWorkflow::test_empyrical_vs_flat_api` passes stored
  returns positionally to `sharpe_ratio`; it binds to `risk_free`, producing
  `NaN` instead of `0.18588774419182566`.
- `TestDataConsistencyWorkflow::test_context_vs_empyrical` contains the same
  stale positional call and compares the context value with `NaN`.

These tests predate Task 3's independently reviewed stored-state positional
binding contract. They are recorded for the Task 12 offline integration gate;
Task 4 deliberately does not weaken Task 3 production behavior or edit those
tests.

## Scope and handoff

Only Task 4-owned production, compatibility tests, provenance generator and
fixture, compatibility documentation, ledger, and this report are included.
Task 5-owned pyfolio compatibility files and
`tests/test_metrics/test_positions_metrics.py` are excluded from staging.
