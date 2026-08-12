# Task 3 Report: empyrical compatibility façade

## Outcome

Task 3 restores the structural empyrical 0.6.0 compatibility surface while preserving fincore 0.3.x enhanced flat and metrics APIs.

- `fincore.empyrical` exports all 54 frozen public symbols, including five literal period constants.
- All 49 callable exports expose the frozen canonical signature and perform real `Signature.bind()` validation before dispatch.
- The 9 factory-derived signatures retain the frozen manifest's `needs_dynamic_review=true` and `reviewed=false` flags; the manifest was not edited.
- `fincore.empyrical.beta` restores `(returns, factor_returns, risk_free=0.0, out=None)`, including fourth-positional `out` mutation.
- `fincore.empyrical.calmar_ratio` restores `(returns, period='daily', annualization=None)` without exposing the enhanced `risk_free` positional parameter.
- `Empyrical` descriptors bind stored `returns` and `factor_returns` only for registry entries declaring those binding modes; class calls still require explicit data.
- Stateful instance descriptors bind positional and keyword arguments against the public signature with state parameters removed, then inject stored state and validate the complete kernel call. Instances without stored state retain the historical explicit-data call path.
- Eager `AnalysisContext` construction was removed. `_ctx` remains a compatibility-only lazy property and is never stored on instances.
- Added `roll_alpha_aligned`, `roll_beta_aligned`, `roll_alpha_beta_aligned`, `roll_annual_volatility`, and `roll_sortino_ratio`, including supplied `out` buffer support.

## Registry design

`MetricSpec` has the exact eleven-field schema required by the brief. `METRIC_REGISTRY` is uniquely keyed by `(surface, public_name, variant)` and contains independent `empyrical_module`, `fincore_flat`, `empyrical_class`, `metrics`, and `context` entries. Kernel and adapter references are lazy `module:attribute` strings. Strict module adapters are separate from enhanced flat/class/metrics entries.

Strict wrapper construction resolves signatures through `signature_manifest_key`, verifies that the key exists and names the same public symbol, and checks that `out_policy` agrees with the frozen signature. `validation_profile`, `result_contract_key`, and `result_projection` remain explicit Task 4 policy hooks; Task 3 stores them but does not yet claim full numerical/result-contract enforcement.

The fincore flat façade resolves its enhanced 0.3.x entries from this registry. The literal `_FLAT_API` map remains as the Task 2 migration-audit input, with an import-time equality assertion against the registry view. Enhanced `fincore.calmar_ratio`, `fincore.beta`, and the underlying metrics signatures were snapshot-tested and did not change. `fincore/metrics/alpha_beta.py` and `fincore/metrics/ratios.py` therefore required no edits.

## TDD evidence

Initial RED, before production edits:

```text
tests/compat/empyrical (four requested modules)
192 collected: 190 failed, 2 passed
```

Failures covered missing exports, absent `MetricSpec`, all-callable signatures, actual missing/extra argument rejection, out mutation, missing rolling APIs, state binding, and eager `_ctx`.

Initial Task 3 GREEN:

```text
four requested compatibility modules: 193 passed in 0.88s
```

Initial review-fix RED:

```text
state binding + registry metadata tests
19 collected: 9 failed, 10 passed
manifest integrity: 24 passed, 1 failed
```

Review-fix GREEN:

```text
state binding + registry metadata tests: 19 passed in 0.46s
four requested compatibility modules: 203 passed in 0.80s
manifest integrity: 25 passed in 6.94s
```

Final required regression gate:

```text
four Task 3 compat modules + tests/test_empyrical + tests/test_metrics
1349 passed in 2.64s
```

Additional evidence:

```text
tests/test_smoke_import.py: 37 passed in 1.11s
C0=54/54 C1=49/49 identity=True beta_out=True flat_strict_separate=True
ruff check: All checks passed
ruff format --check: 10 files already formatted
git diff --check: clean
```

## Commit and scope

- Base commit: `a5302f4 fix: restore empyrical public and positional contracts`
- Review follow-up commit message: `fix: enforce empyrical instance binding contracts`
- Branch: `codex/fincore-convergence-alphalens`
- Only Task 3-owned source files, the four requested compatibility modules, the flat migration fixture, and this report are staged.
- The empyrical and pyfolio frozen API fixtures, Task 2 documents, plan, and progress files are unchanged.

## Risks and follow-up

- This task establishes C0/C1 and structural `out`/binding behavior. Numerical C2/C3 oracle convergence, including aggregate calendar grouping, CVaR semantics, and complete rolling shape/index parity, remains explicitly assigned to Task 4.
- Fresh controlled generation from pinned sibling roots confirmed the empyrical and pyfolio fixtures were byte-identical. Only `fincore-flat-api-migrations.json`'s repository source SHA changed, restoring the full 25/25 manifest-integrity gate without altering upstream provenance or review flags.
