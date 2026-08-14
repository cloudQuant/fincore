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

`968feb5 build: package alphalens integration`

## Concern

The isolated Alphalens rendering profile emits intended legacy summary tables
to stdout. This is harmless but makes wheel-matrix logs verbose.
