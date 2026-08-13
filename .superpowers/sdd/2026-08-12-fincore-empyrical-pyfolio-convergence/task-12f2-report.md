# Task 12F-2 Report: release-gate tooling, CI enforcement, fixture regeneration

## Status: DONE

## Script designs

### scripts/audit_test_markers.py
- `RELEASE_SELECTORS` mirrors ci.yml's exact selectors (fast, serial, non-serial, integration-offline, compat path); each must collect >= 1 test (pytest --collect-only exit-code parsing; exit 5 = valid empty).
- `slow` is an UNOWNED selector: currently 0 tests; the audit fails the day someone adds one, demanding a CI owner (documented in docstring).
- Integration subtype enforcement via an in-process pytest collection plugin: every test with `integration` must have exactly one of `integration_offline`/`integration_online`; subtype without parent also fails (covers the 5 test_workflows.py classes globally).
- `--compare-junit A B` aggregates testsuite counts and fails on collected/passed/skipped mismatch.

### scripts/check_coverage_baseline.py
- Overall branch coverage from pytest-cov JSON `totals.percent_covered` >= baseline `branch-coverage` run's `branch_coverage_percent`.
- Changed lines = `git diff --unified=0 <baseline source.commit> -- fincore` added lines; classified covered (executed/excluded), uncovered (missing), unmeasured (non-executable or coverage-omitted files, reported but excluded from the ratio). Requires >= 95% (default); lists uncovered changed lines on failure. --changed-base override for shallow clones; CI checkout uses fetch-depth: 0.

## Fixture-diff analysis (explicit regen-and-review)
Regenerated with scripts/generate_compat_manifest.py against pinned sibling roots. Only change: `tests/compat/fixtures/fincore-flat-api-migrations.json` -> `source.sha256` of `fincore/__init__.py` (bd6fc2b9... -> a2799fec...). No signature/default/public-symbol semantic change (verified by diff excluding the sha field). empyrical/pyfolio/portfolio-contracts fixtures byte-identical. Compat suite: 648 passed.

## Required extra fixes found by gate validation (beyond listed items)
The release gates could not pass without these; all are test/doc-only except perf_stats docstring:
1. 6 stale coverage-exact tests (test_coverage_exact, test_coverage_final_edges, test_coverage_gaps, test_exact_line_coverage, test_final_coverage_edges) still asserted old NaN-tolerant mar/calmar behavior; migrated to the converged fail-fast contract (NumericalError / DataAlignmentError), same pattern as 12F-1.
2. tests/contracts/test_metric_surface_profiles.py: parametrize over `METRIC_REGISTRY.values()` collected in nondeterministic order across worker processes, failing xdist collection; sorted deterministically.
3. tests/test_empyrical/test_empyrical_zipline_coverage.py reloaded fincore.empyrical in place without restoring the module dict; the leaked Empyrical class broke `call_explicit_metric` isinstance checks (stateful binding TypeError in a later pyfolio test). Fixed with full module __dict__ snapshot/restore.
4. fincore/metrics/perf_stats.py docstring listed kwargs-only n_samples/random_seed as parameters (griffe warnings); mkdocs_docs/guide/visualization.md had a duplicate VizBackend primary URL. mkdocs --strict now exits 0.

## Coverage gate results (real data)
- overall branch coverage 95.29% >= baseline 94.00%.
- changed lines: 2775 covered / 80 uncovered / 4615 unmeasured = 97.20% >= 95% (was 94.05% before the new tests/quality/test_release_gate_changed_lines.py, 38 tests).
- Non-serial single vs xdist JUnit: collected=3321 passed=3307 skipped=14 — identical.

## CI job graph changes
- ci.yml: test matrix (fast+serial, --maxfail=0), non-serial-single + non-serial-parallel + compare-nonserial (JUnit gate), compat, integration-offline, lint, typecheck (full-package mypy), security, marker-audit, coverage-branch (same selector as baseline + changed-lines gate, fetch-depth 0), docs (mkdocs --strict), perf kept intact; build needs ALL blocking jobs (not only test) and runs the full wheel-consumer profile matrix.
- publish.yml: new `verify` job runs every blocking gate inline; `publish` needs verify (trusted publishing unchanged).
- docs.yml: trigger fixed main -> master. ci-enhanced.yml deleted. test-priority.yml: --maxfail=0, stale timing/count comments removed.
- pyproject.toml: registered integration_offline/integration_online markers.

## Gate results
1. audit_test_markers.py -> clean (fast 3321, serial 6, non-serial 3321, integration-offline 15, compat 648; 0 integration violations).
2. tests/compat --maxfail=0 -> 648 passed.
3. -m integration_offline -> 15 passed.
4. Real-data validation: non-serial single 3307 passed EXIT 0; xdist 3307 passed EXIT 0; JUnit compare OK; coverage gate OK (95.29% / 97.20%).
5. ruff check + format full scope clean; git diff --check clean; mypy fincore = 0 errors (98 files); serial suite 6 passed; mkdocs --strict exit 0.

## Concerns
- The 80 still-uncovered changed lines are concentrated in _pyfolio_impl/empyrical legacy adapter branches and defensive paths; the 97.20% margin is comfortable but future edits can re-trip the gate.
- Unmeasured changed lines in coverage-omitted files (`fincore/__init__.py` etc.) never enter the ratio by design; documented in the script.
- publish.yml re-runs the full gate inline on releases (slow but self-contained); GitHub cannot cross-workflow `needs`, so this is the enforcement mechanism.
- 6 stale tests were migrated in this commit even though the 12F-2 brief listed only the manifest failure as outstanding; the fast suite had 7 deterministic failures on the branch.
