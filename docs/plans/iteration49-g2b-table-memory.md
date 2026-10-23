# Iteration 49 G2b: evaluation long-table memory

**Status:** focused compatibility QA passed. Three paired 500-symbol runs passed semantic verification; full-universe performance acceptance remains pending.

**Base:** fincore `dev`, G1c commit `cdf8ea9e8ac8c86a243cb458cdb80b7eeec1dfe8`.

This round changes only `fincore/factor_analysis/research.py` and `tests/factor_analysis/test_factor_analysis_research.py`. Labels and signals are assembled into owned, final-sized typed arrays, then each DataFrame is constructed once instead of building per-period/per-factor frames and concatenating them. Fresh index blocks avoid aliasing input frames. Existing `row_values` storage is reused for stability checks and signal comparisons; processed matrices are released after their last correlation use. Factor validation and study-slice defensive copies remain unchanged.

The focused regression compares against the previous concat assembly and covers multi-asset/multi-period ordering, dtypes and time units, object groups, NaN/infinity/signed-zero behavior, combination and comparison paths, and input isolation. Root reports `ff376e` passed the research and reuse-contract test files: 34 passed, exit 0, 9.29 seconds. The earlier Ruff, syntax, and whitespace checks also passed.

## Paired 500-symbol screen

Root ran three A/B pairs per example on the fixed 500-symbol, 243-date, seven-column input (121,500 panel cells): 18 full runs and 9 verifier comparisons. A is the G1c baseline; B is this G2b candidate. All three verification receipts passed comparisons for 025, 026, and 027, including factor values, output tables, reports, evaluation IDs/fingerprints, and summary metadata.

For each run, the primary captured peak is `max(main_peak_working_set_bytes, resource_peak_tree_rss_bytes)`: this includes the OS-captured main-process high-water and the process-tree sampler. The sampled tree RSS is also reported separately; it is sampled at roughly 0.5-second intervals. Ratios are medians of the three paired B/A ratios. Spread is `(max(ratio) - min(ratio)) / median(ratio)`. Memory values use decimal MB.

| Example | Captured peak B/A (spread); median A -> B | Sampled tree RSS B/A (spread); median A -> B | Main wall B/A (spread) | Main CPU B/A (spread) |
| --- | ---: | ---: | ---: | ---: |
| 025 | 0.9515 (0.48%); 345.6 -> 329.1 MB | 0.9465 (2.01%); 333.6 -> 312.9 MB | 1.0251 (4.42%) | 1.0044 (5.32%) |
| 026 | 0.8924 (2.60%); 419.0 -> 374.0 MB | 0.8967 (1.72%); 403.6 -> 361.9 MB | 0.9898 (3.23%) | 0.9900 (2.64%) |
| 027 | 0.9906 (0.14%); 322.2 -> 319.3 MB | 0.9921 (1.12%); 307.2 -> 301.6 MB | 1.0009 (3.09%) | 0.9824 (1.07%) |

The results support a memory improvement, strongest for 026, modest for 025, and small for 027. Wall-time changes are small and mixed: 025 is about 2.51% slower, 026 about 1.02% faster, and 027 effectively flat. CPU changes are also small; no speedup claim follows from them.

This is the 500-symbol screen only. Its peaks do not establish the required <=1 GB decimal captured process-tree peak for the 5,471-symbol full run, and it provides no evidence for the separate 80% wall-time target. The earlier approximately 383 MB estimate of repeated float64 column slots is only a planning heuristic, not a strict lower bound; pandas block layout and sharing during concatenation were not measured independently.

The compact machine-readable summary records every result path and measured metric, both raw peak fields, result hashes, and the three verifier paths/hashes at `.codex-tmp/iteration49-g2b-pair-summary.json`.
