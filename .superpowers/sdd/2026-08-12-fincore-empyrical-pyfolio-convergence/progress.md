# SDD ledger — plan: docs/plans/2026-08-12-fincore-empyrical-pyfolio-convergence.md

- Controller: /root
- Branch: `codex/fincore-convergence-alphalens`
- Plan baseline commit: `257b7fe`
- Task 1: complete (commits `60a1327`, `53af921`, `e858725`; final review CLEAN; focused `35 passed`; trusted baseline `2279 passed, 14 skipped`; branch coverage `94.0%`)
- Task 2: complete (commits `99a57df`, `d277ea6`, `a569315`; final review CLEAN; manifest suite `25 passed`; 54/49 empyrical and 11-workflow pyfolio targets frozen)
- Task 3: complete (commits `a5302f4`, `404f80b`; final review CLEAN; focused `203 passed`; broad `1349 passed`; manifest `25 passed`; C0 `54/54`, C1 `49/49`)
- Task 4: complete (commits `cc12cb7`, `4e0824d`, `5f7529a`, `816e128`; final review CLEAN; convenience/frozen-value RED `19 failed, 187 passed`; follow-up focused `206 passed`; enhanced-binary matrix `151 passed`; context impact `291 passed, 3 pinned warnings`; isolated broad gate `1297 passed, 3 pinned warnings`; manifest `26 passed`; Task 3 regression `203 passed`; Task 5 selector regression `159 passed`)
- Task 5: complete (commits `575a040`, `d686b11`, `fd03bf6`; final review CLEAN; first follow-up RED `17 failed, 10 passed` plus generated-fixture RED; first focused matrix `28 passed`; volume P1 RED `13 failed, 2 passed`; volume P1 focused `15 passed`; expanded positions `88 passed`; manifest + domain `159 passed`; Task 3 regression `203 passed`)
- Task 6: second independent review follow-up awaiting final re-review (base `816e128`; implementation commits `5863321`, `cd01847`; initial assertion RED `4 failed, 9 passed`; public/lazy/drawdown RED `26 failed, 2 passed`; perf/full/no-write RED `7 failed`; stored-state RED `2 failed`; common-utils import RED `1 failed`; first-review RED `3 failed, 1 passed`; first-review focused `36 passed`; second-review RED `1 failed, 1 passed`; second-review focused `37 passed`; Task 6 impact `889 passed, 9 expected business warnings`; Pyfolio legacy `90 passed`; Task 4 `291 passed, 3 pinned warnings`; Task 5 `159 passed`; Task 3 `203 passed`; manifest `26 passed`; moved-class audit `69/69` methods)
- Task 7 alignment ledger: closed by the Task 4 second review fix; all 33 enumerated enhanced binary entry points now use the explicit shared policy, dependent callers forward it, the strict module remains frozen, and the direct legacy shim remains unchanged.
- Task 12 ledger: update three stale instance-call assertions in `tests/integration/test_workflows.py` to the Task 3 stored-state positional-binding contract before the offline integration release gate.
- Status: Task 4 and Task 5 are complete and CLEAN; Task 6 second-review follow-up is awaiting final independent re-review
