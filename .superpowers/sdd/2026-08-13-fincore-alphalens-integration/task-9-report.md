# Task 9 report — extras, wheel consumer, and CI gates

## Requirements addressed

- Kept `pyproject.toml` as the runtime dependency source and labeled
  `requirements.txt` and `requirements-test.txt` as contributor-only files;
  both contributor files include `statsmodels>=0.14`.
- Guarded the functional-extra union, prohibited `fincore[...]` self-references,
  external `alphalens`/`empyrical`, and direct requirement URLs.
- Validated wheel inclusion of the Alphalens/factor-analysis runtime modules
  and `py.typed`, plus exclusion of tests, notebooks, PNGs, sibling paths,
  Versioneer, and compatibility-oracle requirements. The release check applies
  the same layout/Apache-license rule to both wheel and sdist.
- Added five installed-consumer profiles: `core`, `factor-analysis`,
  `alphalens`, `alphalens-pyfolio`, and `all`. Each rejects source-checkout
  imports; the factor profile runs prepare/IC/alpha-beta, Alphalens renders a
  plot and summary under Agg, and Alphalens-Pyfolio renders the real returns
  sheet.
- The `all` profile now uses an isolated temporary virtual environment. Its own
  `python -m pip check` validates the installed environment rather than relying
  on `PYTHONPATH` for a `--target` directory. A regression test creates broken
  installed metadata and proves venv `pip check` returns nonzero.
- Added bounded install/consumer timeouts per required profile and a regression
  assertion for those bounds.
- Added the blocking `compat-alphalens` job in CI, added it to `build.needs`,
  and added the matching Alphalens compatibility gate to the publish workflow.

## RED/GREEN evidence

RED command:

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' tests/packaging -q --tb=short --maxfail=0
```

Observed intended failure: the installed-wheel CLI rejected
`factor-analysis` as an invalid profile (`argparse` exit 2). A later isolated
consumer run exposed a missing bootstrap `json` import; a dedicated regression
test now asserts that the emitted `-S -E` consumer imports it. A second RED
test exposed absent explicit profile timeouts, and a third exposed that `all`
was not using a venv for `pip check`.

GREEN commands and results:

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' tests/packaging -q --tb=short --maxfail=0
# 25 passed

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m build \
  --outdir build/alphalens-dist
# built fincore-0.3.0.tar.gz and fincore-0.3.0-py3-none-any.whl

/Users/yunjinqi/opt/anaconda3/bin/conda run --no-capture-output -n base python -u \
  scripts/test_installed_wheel.py --dist build/alphalens-dist \
  --profiles core factor-analysis alphalens alphalens-pyfolio all
# 5/5 profiles passed

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python \
  scripts/check_release_consistency.py --dist build/alphalens-dist
# Release consistency: OK

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m ruff check \
  scripts/test_installed_wheel.py scripts/check_release_consistency.py tests/packaging
# All checks passed
```

## Files changed

- `pyproject.toml`
- `MANIFEST.in`
- `requirements.txt`, `requirements-test.txt`
- `tests/packaging/test_optional_extras.py`
- `tests/packaging/test_extras_union.py`
- `tests/packaging/test_wheel_contents.py`
- `scripts/test_installed_wheel.py`
- `scripts/check_release_consistency.py`
- `.github/workflows/ci.yml`, `.github/workflows/publish.yml`

## License/notice decision

`LICENSE` is packaged and asserted as Apache-2.0. `THIRD_PARTY_NOTICES.md` was
not created: the task brief reserves it for a separate human license decision.

## Commit

`d10931e build: package alphalens integration`

## Review follow-up (2026-08-15)

The Task 9 review found two Critical packaging gaps.  This follow-up resolves
both without changing runtime wheel behavior:

- Removed the bare external `empyrical` requirement from `requirements.txt`.
  The packaging regression now scans `requirements.txt`,
  `requirements-test.txt`, PEP 517 build requirements, project dependencies,
  and every optional dependency for external `alphalens`, `empyrical`, and
  direct URL requirements.  It also checks built wheel `METADATA` and sdist
  `PKG-INFO` for the same prohibited external requirements/URLs.
- Stopped including `requirements.txt` and `requirements-test.txt` in the
  sdist.  Wheel and sdist layout regressions now assert that contributor/test
  requirements are absent; the release-consistency script enforces the same
  rule for either artifact type.
- Corrected CI/publish wording: the release gate runs the five required Task 9
  consumer profiles, rather than a misleading “full profile matrix” label.

Review RED command and observed result:

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' \
  tests/packaging/test_extras_union.py::test_supported_dependency_inputs_do_not_install_external_compatibility_packages_or_urls \
  tests/packaging/test_wheel_contents.py::test_sdist_excludes_contributor_and_test_requirement_artifacts \
  -q --tb=short --maxfail=0
# 2 failures: requirements.txt contained external empyrical; sdist contained
# requirements.txt and requirements-test.txt
```

Review GREEN commands and results:

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' tests/packaging -q --tb=short --maxfail=0
# 28 passed in 8.45s

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m build \
  --outdir build/alphalens-dist-task9-followup
# built fincore-0.3.0.tar.gz and fincore-0.3.0-py3-none-any.whl

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python \
  scripts/check_release_consistency.py --dist build/alphalens-dist-task9-followup
# Release consistency: OK

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m ruff check \
  scripts/test_installed_wheel.py scripts/check_release_consistency.py tests/packaging
# All checks passed

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m ruff format --check \
  scripts/test_installed_wheel.py scripts/check_release_consistency.py tests/packaging
# 6 files already formatted
```

The original fresh-wheel consumer gate remains evidenced above as `5/5`
profiles passed (`core`, `factor-analysis`, `alphalens`, `alphalens-pyfolio`,
and `all`).  This follow-up is confined to contributor metadata, sdist layout,
artifact verification, and workflow labels; it does not alter the installed
consumer code or wheel runtime dependency metadata.

Follow-up fix commit: `4dcf019 fix: harden alphalens package artifacts`.

## Re-review follow-up (2026-08-15)

The remaining Critical bypass was case-sensitive comparison of
`Requirement.name`.  Source dependency inputs, wheel/sdist metadata, and the
release-consistency extra guard now compare `canonicalize_name(requirement.name)`
against the canonical PEP 503 prohibited names (`alphalens`, `empyrical`).

Two parameterized regressions inject the valid mixed-case requirements
`Empyrical>=1` and `AlphaLens>=1` into the actual source-input and wheel-metadata
guards.  Each must be rejected, so future raw-name comparisons cannot silently
permit an external compatibility package.

RED command and result:

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' \
  tests/packaging/test_extras_union.py::test_source_requirement_guard_rejects_mixed_case_external_names \
  tests/packaging/test_wheel_contents.py::test_wheel_metadata_guard_rejects_mixed_case_external_names \
  -q --tb=short --maxfail=0
# 4 failed: the unnormalized guards did not raise for either mixed-case name
```

GREEN commands and results:

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' \
  tests/packaging/test_extras_union.py::test_source_requirement_guard_rejects_mixed_case_external_names \
  tests/packaging/test_wheel_contents.py::test_wheel_metadata_guard_rejects_mixed_case_external_names \
  -q --tb=short --maxfail=0
# 4 passed in 0.32s

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' \
  tests/packaging/test_extras_union.py::test_supported_dependency_inputs_do_not_install_external_compatibility_packages_or_urls \
  tests/packaging/test_wheel_contents.py::test_sdist_excludes_contributor_and_test_requirement_artifacts \
  tests/packaging/test_wheel_contents.py::test_distribution_metadata_has_no_external_compatibility_requirements_or_urls \
  -q --tb=short --maxfail=0
# 4 passed in 4.90s

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' tests/packaging -q --tb=short --maxfail=0
# 32 passed in 8.25s

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m ruff check \
  scripts/check_release_consistency.py tests/packaging/test_extras_union.py tests/packaging/test_wheel_contents.py
# All checks passed

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m ruff format --check \
  scripts/check_release_consistency.py tests/packaging/test_extras_union.py tests/packaging/test_wheel_contents.py
# 3 files already formatted
```

Re-review fix commit: `a40b4ad fix: canonicalize package dependency guards`.

## Re-review 2 follow-up (2026-08-15)

The release-consistency script previously checked only wheel self-dependencies
and sdist version headers.  It now parses the actual wheel `METADATA` and
sdist `PKG-INFO` `Requires-Dist` fields through one shared helper.  The helper
canonicalizes names with PEP 503 normalization and rejects external
`alphalens`/`empyrical` distributions (including mixed case) and direct URLs.
The source-extra check shares the same predicate.

The new focused regression builds a real wheel/sdist pair, injects a valid
header before each metadata body, and runs the release script as a subprocess.
It proves rejection of wheel `Empyrical>=1`, sdist `AlphaLens>=1`, and a wheel
direct URL requirement.

RED command and result:

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' \
  tests/packaging/test_release_consistency.py::test_release_consistency_rejects_prohibited_artifact_requirements \
  -q --tb=short --maxfail=0
# 3 failed in 5.75s: release consistency accepted each injected artifact requirement
```

GREEN commands and results:

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' \
  tests/packaging/test_release_consistency.py::test_release_consistency_rejects_prohibited_artifact_requirements \
  -q --tb=short --maxfail=0
# 3 passed in 5.47s

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' tests/packaging -q --tb=short --maxfail=0
# 35 passed in 54.81s

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python \
  scripts/check_release_consistency.py --dist build/alphalens-dist-task9-followup
# Release consistency: OK (wheel and sdist Requires-Dist guards both clean)

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m ruff check \
  scripts/check_release_consistency.py tests/packaging/test_release_consistency.py
# All checks passed

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m ruff format --check \
  scripts/check_release_consistency.py tests/packaging/test_release_consistency.py
# 2 files already formatted
```

Re-review 2 fix commit: `e9d8561 fix: guard artifact dependency metadata`.

## Concern

The isolated Alphalens rendering profile emits intended legacy summary tables
to stdout. This is harmless but makes wheel-matrix logs verbose.

## Re-review 3 follow-up (2026-08-15)

Malformed `Requires-Dist` values in wheel `METADATA` or sdist `PKG-INFO` now
produce an ordinary release-consistency failure containing the raw value.
Artifact parsing catches only `InvalidRequirement`; valid mixed-case external
packages and direct URLs continue through the existing prohibited-dependency
guard.

RED command and result:

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' \
  tests/packaging/test_release_consistency.py::test_release_consistency_rejects_prohibited_artifact_requirements \
  -q --tb=short --maxfail=0
# 2 failed, 3 passed: malformed wheel/sdist metadata raised InvalidRequirement
```

GREEN commands and results:

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' \
  tests/packaging/test_release_consistency.py::test_release_consistency_rejects_prohibited_artifact_requirements \
  -q --tb=short --maxfail=0
# 5 passed in 6.35s

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' tests/packaging -q --tb=short --maxfail=0
# 37 passed in 14.15s

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m ruff check \
  scripts/check_release_consistency.py tests/packaging/test_release_consistency.py
# All checks passed

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m ruff format --check \
  scripts/check_release_consistency.py tests/packaging/test_release_consistency.py
# 2 files already formatted

git diff --check
# clean
```

Re-review 3 fix commit: `4ae8012 fix: reject malformed artifact requirements`.
