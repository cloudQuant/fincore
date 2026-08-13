# Release Candidate Checklist — fincore 0.3.0

This checklist itemizes the evidence required before a 0.3.0 release can be
declared. Each item links to the real evidence location. Items whose evidence
lives in CI artifacts are marked **[CI artifact]** and cannot be verified from
a checkout alone.

**Nothing in this project may claim Stable, Production, or 1.0.** The package
maturity is Beta (`Development Status :: 4 - Beta`). An empty or missing item
blocks the release; it must never be replaced by an assertion.

## 1. Compatibility evidence (C0–C4)

| # | Item | Evidence location | Status gate |
|---|------|-------------------|-------------|
| 1.1 | empyrical 0.6.0 frozen manifest | `tests/compat/fixtures/empyrical-0.6.0-api.json` (54 symbols / 49 callables, pinned commit `74655e9`) | `pytest -o addopts='' tests/compat -q` |
| 1.2 | C0: all 54 public symbols resolve in `fincore.empyrical` | `tests/compat/empyrical/test_public_api.py` | same |
| 1.3 | C1: all 49 callable signatures match the manifest | `tests/compat/empyrical/test_signatures.py` | same |
| 1.4 | C3: core callables numerically verified (CVaR ties, annual_volatility, cum_returns, rolling family, `out` buffers, alignment, perf-attrib) | `tests/compat/empyrical/test_numeric_contracts.py`, `test_out_contract.py`, `test_rolling_contracts.py`, `test_index_contracts.py`, `test_perf_attrib_alignment.py`, `test_enhanced_binary_alignment.py`, `test_state_binding.py` | same |
| 1.5 | pyfolio 0.9.6 profile frozen manifest (11 workflows, pinned commit `724bbd7`) | `tests/compat/fixtures/pyfolio-0.9.6-api.json` | same |
| 1.6 | C1: all 11 workflow paths/signatures | `tests/compat/pyfolio/test_public_api.py` | same |
| 1.7 | C4 main chains (risk/returns/perf-attrib/full-sheet) run compute-plot-sheet end-to-end | `tests/compat/pyfolio/test_risk_e2e.py`, `test_full_tear_sheet_e2e.py`, `test_perf_attrib.py`, `test_drawdown_e2e.py` | same |
| 1.8 | No-write side-effect safety (compat workflows never write into the package) | `tests/compat/pyfolio/test_no_source_writes.py` | same |
| 1.9 | Manifest integrity (fixture drift, oracle attestation invalidation) | `tests/compat/test_manifest_integrity.py` | same |
| 1.10 | CI job `compat` green on the release commit | `.github/workflows/ci.yml` | **[CI artifact]** |

## 2. Test evidence (serial / parallel / full)

| # | Item | Evidence location | Status gate |
|---|------|-------------------|-------------|
| 2.1 | Fast suite (parallel) + serial suite | CI job `test` | **[CI artifact]** |
| 2.2 | Non-serial suite single-process and xdist, JUnit-equal | CI jobs `non-serial-single`, `non-serial-parallel`, `compare-nonserial` | **[CI artifact]** |
| 2.3 | Offline integration suite | CI job `integration-offline` | **[CI artifact]** |
| 2.4 | Machine-generated baseline counts (runs, passed, skipped, warnings, branch coverage) | `docs/quality/current-baseline.{json,md}` (Task 1 baseline; **the final acceptance run regenerates these files** — do not trust numbers copied elsewhere) | `python scripts/collect_quality_baseline.py` |
| 2.5 | Marker/selector audit (no unregistered markers, subtype enforcement) | `scripts/audit_test_markers.py`; CI job `marker-audit` | **[CI artifact]** |

## 3. Static quality gates

| # | Item | Evidence location | Status gate |
|---|------|-------------------|-------------|
| 3.1 | Ruff lint + format | CI job `lint` | **[CI artifact]** |
| 3.2 | mypy on the full package, 0 errors (makes `py.typed` honest) | CI job `typecheck` | **[CI artifact]** |
| 3.3 | Bandit security scan | CI job `security` | **[CI artifact]** |
| 3.4 | Branch coverage >= baseline; changed lines >= 95% | CI job `coverage-branch`; `scripts/check_coverage_baseline.py` | **[CI artifact]** |
| 3.5 | Docs build (`mkdocs build --strict`) | CI job `docs`; local gate in this task | **[CI artifact]** |
| 3.6 | Doc examples execute | `tests/docs/test_examples.py` (`pytest -o addopts='' tests/docs -q`) | local |

## 4. Packaging / wheel evidence

| # | Item | Evidence location | Status gate |
|---|------|-------------------|-------------|
| 4.1 | Single metadata source: `pyproject.toml` (version 0.3.0, classifiers incl. Beta, Python >=3.11) | `pyproject.toml`; `scripts/check_release_consistency.py` | `python scripts/check_release_consistency.py --dist dist/` |
| 4.2 | sdist + wheel build; `twine check` clean | CI job `build` | **[CI artifact]** |
| 4.3 | Fresh-consumer wheel matrix (core pyfolio interactive bayesian report-pdf all) | `scripts/test_installed_wheel.py`; CI job `build` | **[CI artifact]** |
| 4.4 | Packaging contract tests | `tests/packaging/`; CI job `build` | **[CI artifact]** |
| 4.5 | No self-dependency / stale or unexpected assets in the wheel | `scripts/check_release_consistency.py` | same as 4.1 |
| 4.6 | Uploaded dist artifact for human inspection | CI artifact name `dist` | **[CI artifact]** |

## 5. Report rendering evidence

| # | Item | Evidence location | Status gate |
|---|------|-------------------|-------------|
| 5.1 | Offline HTML report (one computation, multiple renderings; charts complete) | `tests/test_report/test_offline_report.py` | `pytest tests/test_report/` |
| 5.2 | PDF rendering (two-stage, Playwright browser binaries) | `tests/test_report/test_render_pdf.py`, `test_pdf_cleanup.py` | same |
| 5.3 | XLSX export via `export=` (caller-owned destination, no implicit writes) | `fincore/utils/common_utils.py::print_table`; report tests | same |

## 6. Performance evidence

| # | Item | Evidence location | Status gate |
|---|------|-------------------|-------------|
| 6.1 | Rolling metrics: time/RSS regression <= 25% vs platform-labelled baseline | CI job `perf`; `scripts/run_rolling_benchmarks.py`, `scripts/compare_benchmarks.py` (provenance: commit/python/numpy/pandas per payload) | **[CI artifact]** |
| 6.2 | Round-trip benchmarks: same budget | `scripts/run_round_trip_benchmarks.py` | **[CI artifact]** |
| 6.3 | Benchmark runner schema gates | `tests/benchmarks/test_rolling_regression.py`, `test_round_trip_scaling.py` | **[CI artifact]** |
| 6.4 | Uploaded benchmark payloads for provenance review | CI artifact name `benchmark-comparison` | **[CI artifact]** |

## 7. Provenance / legal evidence

| # | Item | Evidence location | Status gate |
|---|------|-------------------|-------------|
| 7.1 | Upstream source register (files, headers, commits, transformations) | `docs/upstream-provenance.md` | human review |
| 7.2 | Third-party notice decision (pyfolio root LICENSE = MIT text vs Apache-2.0 headers in source; no conclusion made here) | `docs/upstream-provenance.md`; `docs/compatibility/pyfolio-0.9.6.md` | **human/license review — pending** |
| 7.3 | Historical 1.0-era claims quarantined as snapshots | `CHANGELOG.md` "Historical snapshots"; `docs/迭代计划/README.md`; header note in `docs/已实现函数索引.md` | local |

## 8. Version / claim consistency

| # | Item | Evidence location | Status gate |
|---|------|-------------------|-------------|
| 8.1 | README, MkDocs, CHANGELOG, tag, runtime version, and wheel metadata all say 0.3.0 | `pyproject.toml`, `fincore/__init__.py` (single source), `scripts/check_release_consistency.py` | same as 4.1 |
| 8.2 | No Stable/1.0/100%-coverage claims anywhere in release-facing docs | README, CHANGELOG, `mkdocs_docs/`, `docs/MIGRATION.md`, `docs/API_STABILITY.md` | human review |
| 8.3 | Docs deploy triggers on master push | `.github/workflows/docs.yml` (branches: master; paths: mkdocs_docs/**, mkdocs.yml, fincore/**, README.md, CHANGELOG.md) | local |

## Completion protocol

The final acceptance run (controller-owned) re-runs the wheel/consistency
gates and regenerates `docs/quality/current-baseline.*`. Until every item
above has evidence, the release candidate stays Beta and version 0.3.0.
