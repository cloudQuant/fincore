# Iteration49 G0: factor target membership

This round checks requested target assets against the unique values actually
present in the factor index. It preserves requested-target error order and
does not count unused MultiIndex levels as observed assets. The input panel,
numeric validation and public return contract are unchanged.

The development base is `dev` at
`e04bd082b85374d58e304ec15d43dd134b7fe77b`. Scope is
`fincore/factor_analysis/research.py`, its existing research test file, and this
record. Development uses the existing checkout.

Validation on 2026-10-23:

- Research suite: 28 tests passed using CPython 3.11 and
  `python -m pytest -p no:asyncio tests/factor_analysis/test_factor_analysis_research.py -q`.
  Added regressions cover repeated assets, unused index levels and missing-target order.
- Actual 500-stock, 243-date, seven-column input: the complete 025/026/027
  examples were compared in three serial fresh-process AB/BA/AB pairs, together
  with the Backtrader G0 changes. Factor values, missing-value positions, all
  evaluation tables, evaluation IDs/result fingerprints and report models match.
- The separate 026 diagnostic measured `_validate_factors` at 1.074058 seconds
  before and 0.020965 seconds after this change (98.05% reduction). These are
  single diagnostic observations, separate from the formal whole-flow pairs.
- Combined G0 whole-flow paired median reductions are 6.09%, 4.10%, and 4.71%
  for 025/026/027. These aggregate figures include Backtrader changes and must
  not be attributed solely to fincore. Peak working sets remain approximately
  922 MB / 1242 MB / 493 MB; the final 1 GB full-universe target is not met.

The shared input manifest, paired timing/resource/source records, exact output
comparisons and final scoped commit hashes are maintained with Backtrader's
`docs/_internal/opts/requirements/迭代49-因子分析性能优化/` round records.
No full-universe run, reduced-output mode, fingerprint-format change, release
or deployment is accepted by this round. Continue with bounded, byte-identical
v1 fingerprint encoding after the accepted G0 source commit.
