# Task 5 Report: portfolio, transaction, and risk workflow contracts

## Outcome

Task 5 restores the pinned Pyfolio 0.9.6 portfolio and transaction semantics
while keeping the enhanced internal API explicit.

- Internal sector/cap computations now return frozen `ExposureBundle` values
  with named `long`, `short`, `gross`, and `net` DataFrames. Volume computation
  returns a frozen `VolumeExposureBundle`. The `Pyfolio` facade alone projects
  these results to the pinned ordered 4/4/3 tuples.
- The false-unpacking cases are covered with exactly four asset columns and
  exactly three dates. Projection rejects missing, unexpected, or duplicate
  categories, and panel computation rejects duplicate dates or asset columns.
- Style factors exclude cash, align by date and asset label, and normalize by
  gross exposure across every non-cash position asset. Sector and cap
  denominators likewise retain position assets missing from metadata while
  numerator metadata is reindexed to position columns; extra metadata assets
  are ignored. Sector, cap, and volume results preserve pinned category order
  and return finite empty/zero/all-cash boundary results. Inclusive cap
  endpoints deliberately double count exactly like pinned Pyfolio.
- `get_long_short_pos` now returns the pinned normalized `long`, `short`, and
  `net exposure` DataFrame. The previous absolute amount summary remains
  available as `get_long_short_notional` on the `Empyrical` and `Pyfolio`
  class surfaces without expanding the strict pinned `fincore.empyrical`
  module API.
- Transaction normalization accepts flat lists, canonical DataFrames, pandas
  date-to-list Series, and plain date-to-list mappings. All paths return the
  fixed eight-column schema, preserve nested sid/symbol, order, commission and
  duplicate timestamps, sort by `dt` stably, recompute `txn_dollars`, and reject
  missing, duplicate canonical columns, booleans, Decimal values, or non-finite
  required numeric values with `ValidationError`. Mapping keys are ignored in
  favor of each transaction's embedded `dt`; commission remains an opaque
  pinned field and may be `None`.
- The real risk sheet now computes volume exposure from `shares_held`, never
  dollar positions. A headless real compute-to-plot-to-sheet test renders the
  expected eight axes without replacing the computation or plotting chain
  with fakes. Sheet alignment uses the shared label/timezone policy and only
  includes panels consumed by active sections, so an unused disjoint shares
  panel cannot suppress a sector-only sheet.

`fincore/tearsheets/risk.py` required no production edit: its plotting
functions already accept the ordered facade projections. The fix belongs at
the typed compute boundary, the facade projection, and the sheet call site.

## TDD evidence

The initial Task 5 compatibility run was captured before production edits:

```text
38 failed, 6 passed
```

The failures grouped around untyped sector/cap/volume results, the 4/4/3 false
unpacking traps, wrong long/short semantics, transaction schema/protocol gaps,
and the real risk-sheet chain. The explicitly migrated old tests added two
long/short RED cases and one real common-Zipline RED case. Stable sorting was
then captured independently as two failing flat/Zipline cases.

The final review-added boundary matrix was also driven through RED before its
implementation:

```text
plain date-to-list Mapping + numeric transaction values + duplicate panels:
11 failed, 8 passed
same focused matrix after implementation: 19 passed
```

Duplicate dates were already rejected by the shared Task 4
`align_time_series` contract; the eleven failures were the missing mapping,
numeric, and duplicate-column contracts.

The independent review follow-up was also test-driven. The first focused run
of the 27 new decision/regression tests recorded:

```text
17 failed, 10 passed
```

The failure clusters were the three partial-metadata denominator cases, one
duplicate canonical transaction-column case, six non-finite amount/price
cases, one missing class-surface method, two duplicate sector-name cases, two
finite-zero volume boundaries, and two risk-sheet active-alignment cases. The
already-green decision tests pinned Decimal rejection, opaque `None`
commission, active no-overlap behavior, inclusive cap endpoints, ignored
Zipline mapping keys, and all three invalid bundle-projection branches. The
generated-fixture divergence assertion was separately captured RED before the
controlled generator was extended.

After implementation, the 27-test matrix plus the fixture assertion is green:

```text
28 passed
```

## Provenance and generated contract fixture

`scripts/generate_compat_manifest.py` now generates
`pyfolio-0.9.6-portfolio-contracts.json` directly from the pinned Git blobs at
commit `724bbd7dbed9a88bb47e1057f2ca29b3409d8e7a`:

- `risk.py`: `aed4a2b58dcdbf6823dacda8c97fe6429511ef9bad95b2def450418aebb7b937`
- `pos.py`: `8c836b40c01ab0d4c2bd9624bc4ddd014a8a2ea79db6920af24f86d5b011f5c1`
- `txn.py`: `83c746b2d432cd686e8f8ae6562d8e67628f69e492e3830c70156f527fe68c1a`

The generator statically extracts complete `SECTORS` and `CAP_BUCKETS` values
and order; the non-finite upper cap boundary is represented portably as the
string `"Infinity"`. Numeric behavior is checked by independent regression
oracles in `tests/compat/pyfolio`. No dynamic upstream import was used, and
the fixture and all six golden-case entries remain explicitly
`reviewed: false`. The volume golden case records the intentional enhancement
that zero-share and no-asset rows return finite zero rather than pinned NaN.
Regeneration changed only this generated portfolio fixture.

Generator integrity and two-pass byte-idempotence are green:

```text
tests/compat/test_manifest_integrity.py: 26 passed
```

## Regression gates

The final plan-specified Task 5 domain gate, including the migrated old risk,
positions, transactions, common-Zipline, and review-follow-up tests, is green:

```text
118 passed in 1.93s
```

The manifest plus Task 5 combined authoritative gate is green:

```text
144 passed in 5.42s
```

The existing Task 4 context-impact gate remains green at `140 passed` with
three expected pinned `perf_attrib` Pandas warnings, and the Task 3 public API
regression gate remains green at `203 passed`. `ruff check`, `ruff format
--check`, and `git diff --check` are clean for every Task 5 follow-up Python
path.

The isolated mypy audit of the three core Task 5 modules has no errors in new
Task 5 code. Four pre-existing errors remain in untouched implementations:
`get_percent_alloc`, `stack_positions`, `get_txn_vol`, and `get_turnover`.

## Full-suite attribution

One full-suite run completed with 32 failures and 14 skips. It is advisory,
not an authoritative Task 5 gate, because Task 4 tests and `_registry.py`
changed concurrently while it was running. No failure was in a Task 5 test:

- 18 failures were newly added Task 4 numeric/rolling boundary tests observed
  against an intermediate concurrent Task 4 registry implementation.
- 3 failures were the previously recorded Task 3 stored-state positional-call
  conflicts in `tests/integration/test_workflows.py`.
- 11 further failures have the same Task 3 attribution: old Pyfolio/slippage
  tests pass returns positionally to stored-state Empyrical methods
  (`annual_volatility`, `cum_returns`, or `annual_return`). They bind to the
  next public parameter under the accepted Task 3 contract.

The controller will run the stable combined/full gate after both Task 4 and
Task 5 commits land. No second unstable long run was started here.

## Scope and handoff

Only Task 5 production, compatibility tests, explicitly migrated old tests,
the controlled portfolio fixture/generator extension, this report, and the
progress ledger are included. No Task 4 empyrical test, fixture, compatibility
document, or review artifact is staged.
