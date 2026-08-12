# SDD ledger — plan: docs/plans/2026-08-12-fincore-empyrical-pyfolio-convergence.md

- Controller: /root
- Branch: `codex/fincore-convergence-alphalens`
- Plan baseline commit: `257b7fe`
- Task 1: complete (commits `60a1327`, `53af921`, `e858725`; final review CLEAN; focused `35 passed`; trusted baseline `2279 passed, 14 skipped`; branch coverage `94.0%`)
- Task 2: complete (commits `99a57df`, `d277ea6`, `a569315`; final review CLEAN; manifest suite `25 passed`; 54/49 empyrical and 11-workflow pyfolio targets frozen)
- Task 3: complete (commits `a5302f4`, `404f80b`; final review CLEAN; focused `203 passed`; broad `1349 passed`; manifest `25 passed`; C0 `54/54`, C1 `49/49`)
- Task 4: review-fix implementation complete, follow-up review pending (base commit `cc12cb7`; expanded focused `109 passed`; context impact `140 passed`; isolated broad gate `1133 passed`; manifest `26 passed`; Task 3 regression `203 passed`; Task 5 selector regression `117 passed`)
- Task 5: initial commit `575a040`; review CHANGES REQUIRED and follow-up in progress (Task 4 does not claim Task 5 complete)
- Task 7 ledger: replace direct enhanced binary uses of the legacy outer-join `metrics.basic.aligned_series` with explicit `strict`/`inner`/`outer_dropna` plus timezone policy at the `alpha_beta`, `ratios`, `risk`, `rolling`, `stats`, `timing`, and `yearly` public entry points enumerated in `task-4-report.md`; acceptance must retain strict façade and direct legacy-shim regressions.
- Task 12 ledger: update three stale instance-call assertions in `tests/integration/test_workflows.py` to the Task 3 stored-state positional-binding contract before the offline integration release gate.
- Status: Task 4 review-fix ready for independent review; Task 5 follow-up remains in progress
