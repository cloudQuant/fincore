# Technical Acceptance Checklist — Fincore 0.5

This is the evidence checklist for a 0.5 technical candidate. It is not a release declaration.

## Capability and architecture

| Item | Evidence |
| --- | --- |
| Required analytical capabilities remain reachable | immutable capability ledger, canonical scenario results, and numerical-oracle records |
| Each leaf capability has one owner and operation ID | builtin catalog report plus architecture tests |
| Retired package-shaped modules and root aliases are absent | negative source and installed-wheel surface tests |
| No obsolete extras or compatibility profiles ship | `pyproject.toml`, metadata, and wheel-content checks |
| Domain import boundaries have no legacy support edge or cycle | architecture convergence check |

## Correctness and quality

| Item | Evidence |
| --- | --- |
| Complete isolated test suite | `pytest -o addopts='' tests -q --tb=short --maxfail=0` |
| Executable examples and strict documentation build | `tests/docs`, `mkdocs build --strict` |
| Static checks | Ruff, type checking, and security checks configured for the candidate |
| Report model/render semantics | canonical report, renderer, and artifact lifecycle tests |
| Package source isolation | staged-source wheel/sdist build and fresh-consumer smoke profiles |

## Performance and maintainability

| Item | Evidence |
| --- | --- |
| Fixed runtime overhead | `python scripts/check_performance.py` |
| Multi-scale workload regressions | benchmark payloads with input/output digests and platform identity |
| Production LOC and duplicate bodies | exact-SHA architecture report against frozen D0 baseline |
| Changed-line and branch coverage | fresh candidate quality report against the documented threshold |

## Candidate boundary

All evidence must name one exact candidate source tree or commit, environment,
dependency set, and generated artifact digest. Historical upstream tests,
fixtures, and pre-0.5 quality snapshots may serve as read-only oracle inputs,
but they are not public API or release evidence by themselves.
