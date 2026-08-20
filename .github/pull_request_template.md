## Description

What does this PR change and why?

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Numerical correctness (adds/updates an independent oracle)
- [ ] Documentation
- [ ] Refactor (no behavior change)

## Verification

- [ ] `python -m pytest -o addopts='' tests/contracts tests/numerical tests/oracles -q`
- [ ] `python -m ruff check fincore tests scripts`
- [ ] `python -m mypy fincore --ignore-missing-imports`
- [ ] `python scripts/snapshot_public_api.py --check tests/contracts/fixtures/public-api-0.3.x.json`

## Breaking changes

If this changes a public path, list the ADR and the deprecation window:

- [ ] No breaking changes
- [ ] Breaking change — ADR: `docs/architecture/adr/...`
