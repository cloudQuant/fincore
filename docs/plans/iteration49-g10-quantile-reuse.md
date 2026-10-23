# Iteration 49 G10: research period quantile reuse — rejected

Root's fixed-500 paired screen completed 18 measured runs, nine comparisons,
and three complete output verifiers. Output equivalence passed, but the cache
candidate is rejected because the wall-time change was small and inconsistent
with a stable evaluation speedup:

| Example | Total paired median reduction | Relative dispersion | `fincore_evaluate` paired median reduction | Relative dispersion |
| --- | ---: | ---: | ---: | ---: |
| 025 | 5.342% | 2.989% | -2.120% | 20.84% |
| 026 | -1.532% | 12.170% | 0.350% | 36.15% |
| 027 | -2.507% | 7.869% | Not used for this decision | Not used |

The 025 total reduction came mainly from lower collection time, a phase this
candidate does not change. The evaluation-phase paired result was slower and
noisy for 025; 026's small positive evaluation result also had high dispersion.
These measurements do not support a stable causal gain. They cover the fixed
500-asset screen only and establish no full-universe speed or memory result.

The candidate was source-QA'd as five unique tests across two runs: four
production-equivalence/cache-bound tests passed before the lifetime-only edit;
the added lifetime test passed separately and verified cache release before
result fingerprinting. There was no combined five-test run. Each successful
run reported one existing Hypothesis warning about skipping `.hypothesis`.
The preserved QA receipt is
`D:\source_code\fincore\.codex-tmp\iteration49-g10-source-qa-summary.json`.

The exact four candidate files are retained at
`D:\source_code\fincore\.codex-tmp\iteration49-g10-rejected\`. The G10
measurement archives remain under
`D:\source_code\backtrader\docs\_internal\opts\requirements\迭代49-因子分析性能优化\measurements\`:

- `7col-500-g10-quantile-reuse-pairs.json`
- `7col-500-g10-quantile-reuse-raw.zip`
- `7col-500-g10-quantile-reuse-source.zip`

The production `data.py` and `research.py` changes were restored from FinCore
`dev` HEAD `4a8018105cf331969bb88cd26c47e158704a32e5`; the candidate test was
moved out of `tests/`. This record is not a performance acceptance or a commit.
