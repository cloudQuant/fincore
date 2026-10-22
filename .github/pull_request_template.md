<!--
Target-branch rule: daily contributions target `dev`; `master` only accepts
promotion PRs (from `dev`) or `hotfix/*` security/release-blocking PRs.
If GitHub pre-selected `master` as the base for a routine change, please
change it to `dev` before submitting. Details: docs/governance/branch-governance.md
-->

## Description

What does this PR change and why?

## Target branch

- [ ] `dev` — routine contribution
- [ ] `dev -> master` promotion PR (scope, exclusions, rollback point listed below)
- [ ] `hotfix/* -> master` — security / release-blocking fix (linked private report required)

Reason for this target (one line):

## Risk level

See [branch governance](docs/governance/branch-governance.md#风险级别). Pick the
highest level your diff touches; maintainers may upgrade, never downgrade.

- [ ] R0 — docs / metadata / non-behavioral tooling
- [ ] R1 — local implementation, no public behavior or numerical change
- [ ] R2 — behavior / compatibility / numerical kernel / frozen manifest
- [ ] R3 — security / governance / release / supply chain

## Compatibility surface impact

Which API surfaces are affected?

- [ ] None
- [ ] `fincore.empyrical` (strict-compat surface — frozen manifests apply)
- [ ] `fincore.pyfolio`
- [ ] `fincore.metrics` / flat API / AnalysisContext
- [ ] `fincore.alphalens` / `fincore.factor_analysis`

## Verification (commands you actually ran and their results)

- [ ] `python -m pytest -o addopts='' tests/contracts tests/numerical tests/oracles tests/property -q` (required for R2)
- [ ] `python scripts/snapshot_public_api.py --check tests/contracts/fixtures/public-api-0.4.0.dev0.json` (required when public paths changed)
- [ ] `python -m ruff check fincore tests scripts`
- [ ] `python -m mypy fincore --ignore-missing-imports`

Results summary (paste exit codes / key numbers; do not claim suites you did not run):

## Related issue / ADR

Fixes / refs:

ADR (required for semantic divergence or breaking changes): `docs/architecture/adr/...`

## Breaking changes

If this changes a public path, list the ADR and the deprecation window:

- [ ] No breaking changes
- [ ] Breaking change — ADR:
