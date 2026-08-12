# Task 3 Report: empyrical compatibility façade

## Outcome

Task 3 restores the structural empyrical 0.6.0 compatibility surface while preserving fincore 0.3.x enhanced flat and metrics APIs.

- `fincore.empyrical` exports all 54 frozen public symbols, including five literal period constants.
- All 49 callable exports expose the frozen canonical signature and perform real `Signature.bind()` validation before dispatch.
- The 9 factory-derived signatures retain the frozen manifest's `needs_dynamic_review=true` and `reviewed=false` flags; the manifest was not edited.
- `fincore.empyrical.beta` restores `(returns, factor_returns, risk_free=0.0, out=None)`, including fourth-positional `out` mutation.
- `fincore.empyrical.calmar_ratio` restores `(returns, period='daily', annualization=None)` without exposing the enhanced `risk_free` positional parameter.
- `Empyrical` descriptors bind stored `returns` and `factor_returns` only for registry entries declaring those binding modes; class calls still require explicit data.
- Eager `AnalysisContext` construction was removed. `_ctx` remains a compatibility-only lazy property and is never stored on instances.
- Added `roll_alpha_aligned`, `roll_beta_aligned`, `roll_alpha_beta_aligned`, `roll_annual_volatility`, and `roll_sortino_ratio`, including supplied `out` buffer support.

## Registry design

`MetricSpec` has the exact eleven-field schema required by the brief. `METRIC_REGISTRY` is uniquely keyed by `(surface, public_name, variant)` and contains independent `empyrical_module`, `fincore_flat`, `empyrical_class`, `metrics`, and `context` entries. Kernel and adapter references are lazy `module:attribute` strings. Strict module adapters are separate from enhanced flat/class/metrics entries; result projection and out policy are stored per entry.

The fincore flat façade resolves its enhanced 0.3.x entries from this registry. The literal `_FLAT_API` map remains as the Task 2 migration-audit input, with an import-time equality assertion against the registry view. Enhanced `fincore.calmar_ratio`, `fincore.beta`, and the underlying metrics signatures were snapshot-tested and did not change. `fincore/metrics/alpha_beta.py` and `fincore/metrics/ratios.py` therefore required no edits.

## TDD evidence

Initial RED, before production edits:

```text
tests/compat/empyrical (four requested modules)
192 collected: 190 failed, 2 passed
```

Failures covered missing exports, absent `MetricSpec`, all-callable signatures, actual missing/extra argument rejection, out mutation, missing rolling APIs, state binding, and eager `_ctx`.

Final focused GREEN (one enhanced-signature non-drift test was added after the initial RED run):

```text
193 passed in 0.88s
```

Final required regression gate:

```text
tests/compat/empyrical tests/test_empyrical tests/test_metrics
1339 passed in 2.49s
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

- Commit message: `fix: restore empyrical public and positional contracts`
- Branch: `codex/fincore-convergence-alphalens`
- Only Task 3-owned source files, the four requested compatibility modules, and this report are staged.
- Frozen manifests, Task 2 documents, plan, and progress files are unchanged.

## Risks and follow-up

- This task establishes C0/C1 and structural `out`/binding behavior. Numerical C2/C3 oracle convergence, including aggregate calendar grouping, CVaR semantics, and complete rolling shape/index parity, remains explicitly assigned to Task 4.
- A supplemental run of the whole Task 2 manifest-integrity module produced `24 passed, 1 failed`: byte regeneration updates only `fincore-flat-api-migrations.json`'s source SHA because this task is required to edit `fincore/__init__.py`. The frozen fixture is intentionally not edited under the Task 3 scope rule; the required Task 3 gates above are green.
