# Iteration 49 G11: bounded positions active-window reduction

G11 replaces the per-timestamp `pd.concat` reduction for ordinary float64
weights with a bounded active-window matrix. The active rows remain in the
same chronological order, and the existing single-expiry-pop rule is kept.
The implementation is enabled only on pandas 2.2.3, whose float64 reduction
layout was source-audited: concat's
homogeneous block is active-row by asset in C order; `DataFrame.sum(axis=1)`
reduces its transposed asset-by-active view, and pandas copies that view in C
order only when NaNs are present. Gross and normalization remain the original
pandas Series operations. Every other pandas version keeps the original
implementation until its reduction layout is independently audited.

The fast path is limited to an exact base Series, a unique datetime index,
unique exact-string asset labels, one float64 NumPy block, and a zero-copy
frame view. Other pandas versions, extension arrays, subclasses, other
labels/dtypes/layouts, and an active matrix above the 8 MiB scratch bound use
the legacy-compatible path.
The returned DataFrame is built once from a float64 output array; NaN results
are zeroed to match the previous final `fillna(0.0)`.

`tests/factor_analysis/test_positions_active_window.py` keeps an independent
copy of the pre-G11 algorithm as a bitwise oracle. It covers a 500-asset
cancellation panel, NaN/infinity behavior, timezone-aware intraday timestamps,
colliding weekend expiries, budget and public-type fallbacks, and duplicate
date/asset errors. No performance result is claimed here; the bounded
fast-path's real-payload gain and process-tree memory remain for the Root's
serial performance round.


## Root source and fixed-500 closeout (2026-10-24)

Root final focused QA passed 12 tests in 1.05 seconds on the exact candidate; three known calendar/Hypothesis warnings were recorded. All 18 full-profile 500 x 243 A/B processes completed, with nine unique comparisons and three complete output verifiers passing. Float64 values, missingness, semantic IDs, all evaluation tables and full report models remain equal; CSV/PNG/HTML outputs remain complete.

| Example | Paired wall reduction | Wall ratio dispersion | Candidate median peak MB |
|---|---:|---:|---:|
| 025 | 2.782% | 11.900% | 309.998 |
| 026 | 10.755% | 5.533% | 348.279 |
| 027 | 1.877% | 2.057% | 299.782 |

These are observed incremental results against G9 using the frozen BT G8 snapshot, not final B0 cumulative gains. 025/026 fail the 5% ratio-stability threshold; 027's small gain is within its observed dispersion. No stable complete-example speed percentage is accepted. The instrumented positions stage dropped by 81.876% / 82.275% / 82.798% at the paired median, with ratio dispersion 10.452% / 9.015% / 9.060%; the direction and large size are consistent across pairs, but these are also observations, not a passed 5% stability claim. Candidate helper calls were 3 / 9 / 3 in each pair, and completed output validation passed. The fixed-500 scratch upper bound is 2,082,000 bytes, below 8 MiB. Module-level timers do not cover aliases captured elsewhere.

The user requested closeout after the current work, with no further optimization. Retain this exact, bitwise-compatible source improvement based on focused correctness and the measured reduction of the targeted work; do not extend the G11 optimization loop. Final cumulative 500 and 5471 comparisons will adjudicate actual time and memory, with unmet 80% / decimal 1 GB targets reported explicitly. This round does not claim those targets passed.

Root evidence is in Backtrader `docs/_internal/opts/requirements/迭代49-因子分析性能优化/measurements/7col-500-g11-positions-{pairs.json,raw.zip,measured-source.zip}`. The raw ZIP preserves actual run logs, direct Root QA log/XML, static review and executed pair configs.

Summary SHA-256: `d97f54030fd18f5ac76e4318d30b42ede0f0dfa9fad79b2c95edcd301f9f18dd`. Raw ZIP SHA-256: `11e0996a53a6a982738693255891524d4caa385d7b311a7f41bdaf869eaf758a`. Measured-source ZIP SHA-256: `553575113f767c7838ffa903113b26b15fb519aa37d2584e45e8911a46e3f2dc`. Repository FinCore, branch dev, parent `650da05cefff8db22aed59cd24b25bef008062ce`; scoped commit recorded by Root receipt.
