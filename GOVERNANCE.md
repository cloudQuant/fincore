# Governance

fincore is a community-maintained project under the Apache License 2.0.  This
document records how decisions are made and how the release is governed.

## Roles

| Role | Responsibility |
| --- | --- |
| Maintainers | Merge PRs, cut releases, approve breaking changes (see `MAINTAINERS.md`) |
| Quant leads | Own numerical correctness: every stable algorithm has an independent oracle |
| Release owner | Owns CI, packaging, supply chain, and the readiness seal |
| Contributors | Open PRs; no PR is merged without passing all gates |

## Decision process

1. **Semantics** — a change to enhanced financial semantics requires an ADR in
   `docs/architecture/adr/`.
2. **Numerical correctness** — no stable kernel is changed without an
   independent oracle and a property test.
3. **Breaking changes** — require an ADR, a SemVer bump, and a deprecation
   window (see `docs/architecture/public-surface-policy.md`).
4. **Releases** — a release may only publish the single candidate artifact that
   passed every gate; the readiness seal must show no unresolved blockers.

## Release gates

The release is governed by the `docs/quality/1.0-readiness.md` seal, which is
`blocked` unless: all stable domains have oracles + property tests + adversarial
fixtures; strict compatibility (C0–C4) is green; catalog coverage is 100%;
quality snapshot is fresh on the current commit; typing checks pass on the
installed wheel; performance has no over-budget regression; packaging and
supply-chain digests match; and the license approval is recorded.
