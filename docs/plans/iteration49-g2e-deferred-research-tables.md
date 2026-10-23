# Iteration49 G2e: defer research output tables

The evaluator previously allocated complete `labels` and `signals` buffers
before evaluating periods, stability, combinations and coverage. On the
5,471-asset, 243-date, three-factor/three-period example, each long table has
3,988,359 rows. Their column buffers alone total about 383 MB, before pandas
overhead. Keeping those buffers alive during metric preparation needlessly
increases overlap with temporary matrices.

This round retains the per-label and per-factor vectors needed by the
calculations, releases completed price/matrix/mask intermediates, and builds
the two complete public tables only after metric computation. Labels retain
period-block ordering; signals retain factor-block ordering and all columns,
dtypes, values and directions. Signal multiplication uses the same expression
on the evaluator's private input copy. No report or public result is omitted.
The result tables still coexist with one another; this change does not claim
to eliminate their required storage.

Root's focused `test_factor_analysis_research.py` run completed with 30
passing markers, 100% progress and exit zero (tool `ac44a2`, 8.56 seconds,
no summary footer). The tests retain the independent former concat oracle,
exact output and input-isolation checks, and verify that all eight long-table
column allocations occur after period/stability/correlation/coverage work,
including the combination path. Independent static review returned GO.
An earlier command referenced a nonexistent second test file and ran no
candidate tests; that harness error is not a candidate failure or acceptance.

Output allocation failures can now surface later, after metric computation;
no partial result is returned. Source QA is complete. Fixed-500 paired results are recorded below.
The full-universe 1 GB ceiling and 80% wall-time target remain unproven.

## G2e fixed-500 paired acceptance

Both arms used Backtrader snapshot `dc52995b7372d8e19423b2d7122bd8d6ec20e3d0`; FinCore A was `d40301518259501eb166e19657f23ea0e583a5e8`, B was that dev checkout plus the deferred-table implementation. Examples 025/026 used five AB/BA/AB/BA/AB pairs; 027 used three. The 26 actual runs and 13 unique output comparisons passed. Five complete verifiers include two reused 027 control slots, not extra timing samples.

| Example | Wall reduction % | Wall dispersion % | CPU reduction % | Peak reduction % | Candidate peak median bytes |
|---|---:|---:|---:|---:|---:|
| 025 | 6.693 | 13.808 | 6.246 | 4.933 | 312643584 |
| 026 | -3.188 | 23.880 | -1.188 | 4.908 | 354189312 |
| 027 | 1.985 | 26.744 | 2.136 | 1.099 | 318005248 |

Reduction is `100*(1-median(B/A))`; dispersion is `(max(B/A)-min(B/A))/median(B/A)`. The noisy wall/CPU results do not establish stable speed improvement. The candidate is retained for equivalent outputs and reduced allocation overlap/memory, subject to the final combined fixed-500 and full-universe gates. No full-universe 80%/1 GB acceptance is claimed. Peak is the greater of OS main-process peak working set and 0.5-second sampled process-tree RSS; both raw fields are retained.

Backtrader evidence: `measurements/7col-500-g2e-pairs.json`, `7col-500-g2e-raw.zip`, `7col-500-g2e-measured-source.zip`.

`7col-500-g2e-raw.zip`: 3467088 bytes / SHA-256 `9160f8084c62b1a4902e1eb36aae98f8396b1a89b570c21293425b28f4fef797`.

`7col-500-g2e-measured-source.zip`: 167237 bytes / SHA-256 `77afb9000677e99fbe5c8e93ecc175baa9824aa75e6c96e1ee4a8544a4cac318`.
