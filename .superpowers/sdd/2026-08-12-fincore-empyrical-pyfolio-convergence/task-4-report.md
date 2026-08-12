# Task 4 Report: numeric, rolling, and time-series semantics

## Outcome

Task 4 converges the pinned empyrical 0.6.0 numeric and rolling result
contracts while making enhanced time-series policies explicit at their public
boundaries.

- Strict CVaR uses the upstream fixed-count order-statistics tail; the tie case
  `[-.2, -.1, -.1, -.1, 1]` at `cutoff=.25` is `-.15`. The enhanced direct
  metric retains its threshold-inclusive `-.125` result. Empty input and
  invalid `NaN`, infinite, and `None` cutoffs now retain the pinned exception
  and warning behavior instead of taking a façade shortcut.
- Strict factory-generated `roll_*` functions now reproduce the pinned unary
  and binary branches separately: unary invalid windows raise, binary invalid
  or empty input follows the empty/`out=nan` contract, each binary operand uses
  its own effective window, Series labels project from the left operand, and
  supplied `out` buffers retain upstream mutation semantics. Strict rolling
  max drawdown treats window-local `NaN` as zero return, while strict rolling
  Sharpe retains the pinned `Inf` behavior.
- The independently implemented strict capture family now follows pinned
  `utils.roll`: inputs must have the same concrete type, Series windows use
  `iloc`, ndarray windows are positional, keyword arguments such as `period`
  are forwarded, and short/invalid windows retain their upstream result shape.
- The shared enhanced alignment contract makes `strict`, `inner`, and
  `outer_dropna` policies explicit, rejects duplicate labels, validates
  `normalize_tz` before inspecting index type, rejects mixed timezone awareness
  by default, and permits explicit UTC normalization. The low-level legacy
  `metrics.basic.aligned_series` outer-join shim remains unchanged.
- The second review closes the previously recorded binary-metric gap. All 33
  direct enhanced entry points in `alpha_beta`, `ratios`, `risk`, `rolling`,
  `stats`, `timing`, and `yearly` now expose keyword-only
  `alignment="inner"` and `normalize_tz=None`. Labelled operands go through the
  shared contract before every short/empty branch; equal-length ndarrays remain
  positional, unequal arrays and mixed ndarray/pandas operands are rejected,
  and dependent public callers forward the selected policy. The five
  conditional/derived callers (`up_alpha_beta`, `down_alpha_beta`,
  `treynor_ratio`, `roll_up_down_capture`, and
  `annual_active_return_by_year`) expose the same enhanced options.
- The frozen strict module does not expose either new keyword. Its alpha/beta
  and beta-fragility adapters retain pinned outer alignment, while the capture
  family retains the pinned independent/filtering semantics rather than being
  blanket-prealigned. Strict rolling capture also preserves the exact pinned
  unknown-keyword error text.
- Strict weekly aggregation keeps calendar year plus ISO week. The enhanced
  `week_year="iso"` option uses ISO year plus ISO week; the intentional
  divergence is recorded in `docs/compatibility/empyrical-0.6.0.md`.
- Enhanced performance attribution intersects by actual date labels and never
  assigns an index merely because lengths match. `outer_dropna` uses the real
  per-day completeness mask, factor-column `strict` and `inner` policies are
  explicit, and dates without usable ticker exposure cannot become false zero
  attribution. The strict `fincore.empyrical.perf_attrib` adapter separately
  reproduces pinned outer label/column broadcasting and all-NaN sum behavior.
- `AnalysisContext` checks all four time-index inputs (`returns`, factors,
  positions, and transactions). With no normalization it preserves established
  object identity and partial labels; explicit UTC normalization covers all
  inputs while preserving duplicate transaction timestamps and event order.
  Defensive copies and general schema validation remain outside this task.

Strict result projection is implemented in the strict façade adapters in
`fincore/_registry.py` and `fincore/empyrical.py`. Enhanced attribution,
timezone behavior, and binary alignment are implemented at their explicit
enhanced/context entry points. The shared legacy `basic.py` shim is unchanged.

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

The review-fix cycle added source-derived boundary tests before each production
change. Its initial RED groups were:

```text
strict CVaR + rolling/capture boundary matrix: 18 failed, 46 passed
enhanced attribution + context timezone matrix: 10 failed, 23 passed
duplicate transaction timestamp preservation: 2 failed
```

The final expanded focused gates are:

```text
four Task 4 compatibility modules: 109 passed, 3 warnings
four Task 4 modules + complete context suite: 140 passed, 3 warnings
enhanced attribution-only selection: 12 passed, 4 deselected, 0 warnings
```

The three warnings belong only to strict pinned `perf_attrib`: pandas emits a
`Pandas4Warning` for the upstream-style outer `pd.concat` of differently
ordered `DatetimeIndex` values. Silencing that warning by changing sort or
label semantics would stop mirroring the pinned path, so it is recorded rather
than folded into the warning-free enhanced contract.

The second review began with a signature/error-message test written before the
production change:

```text
33 enhanced signatures + 3 rolling-capture errors: 36 failed
```

The behavior matrix was then expanded before implementation:

```text
central policy/routing, seven module families, dependent callers, strict
differentials, and legacy shim: 54 failed, 2 passed
```

The final alignment module contains 132 passing cases. It parameterizes all 33
direct sites for the public signature, duplicate-label routing, and rejection
of mixed positional/labelled inputs. Central cases additionally cover
partial/disjoint `strict`/`inner`/`outer_dropna`, successful strict ordering,
timezone rejection and UTC normalization, invalid timezone options, ndarray
length/identity, immutability, seven representative module families, nested
policy forwarding, pinned strict numeric constants, and the unchanged legacy
outer-join shim.

### Intentional enhanced behavior changes

| Surface | Partial or disjoint labels | Timezones | Public parameters |
| --- | --- | --- | --- |
| Enhanced direct, flat, and class convenience APIs | Default label intersection; disjoint annual alpha/beta produce empty yearly output | Mixed awareness fails unless `normalize_tz="UTC"` | Keyword-only `alignment` and `normalize_tz` |
| Frozen strict `fincore.empyrical` module | Pinned per-function behavior: outer alpha/beta alignment; independent capture/filtering | Pinned legacy behavior | Signatures unchanged; new keywords rejected |
| `metrics.basic.aligned_series` | Existing outer join with NaN rows | Existing shim behavior | Unchanged |

Accordingly, old enhanced weekly/monthly tracking, Treynor, and annual-active
tests now compare the convenience API with the corresponding direct enhanced
kernel under the same period instead of freezing obsolete positional/outer
numbers. Old annual alpha/beta no-overlap tests now assert empty output. Tests
that mocked a removed private alignment alias were replaced with observable
disjoint/common-year behavior; no strict façade expectation was migrated.

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

The expanded Task 4 context-impact gate is green:

```text
five Task 4 compatibility modules + complete context suite:
272 passed, 3 strict-pinned warnings
```

The broad Task 4 brief impact gate, excluding only the Task 5-owned positions
implementation from this change, is green:

```text
tests/compat/empyrical + tests/test_empyrical/stats + tests/test_metrics:
1278 passed, 3 strict-pinned warnings
```

The independently reviewed Task 3 strict surface remains green:

```text
public API + signatures + state binding + out contract: 203 passed
```

The final independently reviewed Task 5 selector also remains green after the
Task 4 changes:

```text
manifest + pyfolio + positions/transactions/risk/common compatibility: 159 passed
```

That selector is cross-task regression evidence only. Task 5 is complete at
commits `575a040`, `d686b11`, and `fd03bf6`, with final review CLEAN.

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

### Second-review alignment closure

The former Task 7 alignment ledger is closed by this Task 4 review fix. No
direct `aligned_series` call remains in the seven enhanced metric modules. The
single contract helper is exercised directly and through every one of the 33
public sites, while strict adapters deliberately retain the heterogeneous
pinned oracle behavior described above.

## Scope and handoff

The second review follow-up contains only Task 4-owned production, strict
adapter/registry changes needed to protect the frozen surface, the new
compatibility matrix, six explicitly migrated obsolete enhanced tests, the
ledger, and this report. No Task 5-owned pyfolio, portfolio,
positions/transactions, sheets, generator, manifest, fixture, or compatibility
test path is changed. Implementation is complete and awaiting independent
review; this report does not mark Task 4 finally accepted.
