# Iteration49 G4: reuse report preparation statistics and portfolio outputs

The report previously computed per-date quantile mean/std/count twice to
produce aggregate and dated views. Research evaluation also grouped the
same valid rows again to recover bucket counts. This round shares the exact
per-date aggregation, preserving its ordering, observed=False behavior,
demeaning and the existing two-stage aggregate over per-date means. By-group
and whole-universe groupings remain distinct; unequal groups are not averaged
to manufacture aggregate results.

Report portfolio inputs now project the first selected period's existing
cumulative returns and raw positions. Public portfolio builders retain their
signatures, numerical expressions, benchmark computation and defensive copies.
No metric, figure or report field is removed.

Root focused tests for quantile reuse, report reuse and research completed
with 56 passing markers and exit zero (414f8f, 9.61 seconds). Existing factor
analysis/report/portfolio parity and factor invariants completed with 13
passing markers and exit zero (230a07/de3415). The extra CLI quiet flag
suppressed numeric pytest footers. The four UP037 lint findings in portfolio
parameter annotations are inherited unchanged from parent 743e45c; no new
finding was reported. Independent review of the composed source delta was GO.

Fixed-500 full-output AB/BA/AB performance acceptance is pending. Full5471,
80% wall reduction and 1 GB process-tree peak remain separate final gates.

## Measured fixed-500 acceptance

Root source QA: 56 focus plus 13 compatibility passing markers, exit0 (414f8f,230a07,de3415); numeric pytest footer suppressed by double quiet. Eighteen complete fixed500 runs, nine unique comparisons and three complete verifiers PASS. 026 paired wall reduction 2.572%, dispersion4.557%; CPU reduction2.618%, dispersion3.566%. 025 wall +2.123% and 027 -3.178% are unstable observations, not accepted speedups. All paired-median timing changes remain within the 5% nonregression screen, and peak memory does not regress. Full5471 and combined80%/1GB acceptance remain pending.

| 示例 | A/B墙钟中位数(s) | 墙钟配对变化 | CPU配对变化 | A/B峰值中位数(MB) |
|---|---:|---:|---:|---:|
| 025 | 23.398/24.045 | +2.123% | +1.357% | 314.245/312.762 |
| 026 | 52.050/49.722 | -2.572% | -2.618% | 352.342/350.597 |
| 027 | 35.193/33.867 | -3.178% | -2.861% | 307.687/305.226 |

Actual complete processes: 18; unique output comparisons: 9; full verifiers: 3. All output comparisons passed. Positive timing changes mean slower. Peak combines OS main-process high-water and 0.5-second sampled process-tree RSS.

Exact measured source, config, raw runs and verifier receipts are in Backtrader's `measurements/7col-500-g4-report-pairs.json` and its paired raw/source archives. Final combined fixed-500 and full5471 targets remain pending.
