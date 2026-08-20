# Public Surface Policy

## Profiles

| Profile | Meaning |
| --- | --- |
| `strict_empyrical_0_6_0` | Frozen empyrical 0.6.0 surface (54 symbols, C0–C3 verified) |
| `strict_pyfolio_0_9_6` | Frozen pyfolio 0.9.6 workflow profile (11 workflows) |
| `strict_alphalens_cloudquant_0_4_0` | Source-shaped alphalens cloudQuant 0.4.0 façade |
| `enhanced_v1` | fincore's own enhanced semantics (metrics, risk, simulation, attribution, factor-analysis, report, optimization) |
| `plugin_v1` | Extension points and entry-point discovery |

## Stability levels

| Level | Guarantee |
| --- | --- |
| `stable` | Signature, return shape, and semantics are frozen; a breaking change requires an ADR and a deprecation window |
| `experimental` | May change without a deprecation window; documented as such |
| `provider_required` | Requires an injected data provider; no default network access |
| `not_implemented` | A declared public path that raises `NotImplementedError` until a verified implementation ships (must be resolved or removed before 1.0) |

## Pre-1.0 vs 1.0

- **Pre-1.0 (0.4.x):** stability levels are advisory; the strict façades are
  frozen by compatibility fixtures, the enhanced layer is versioned but may
  evolve within a documented window.
- **1.0:** every `stable` export has non-`Any` public typing, an independent
  oracle, and a generated stub/docs entry.  No `not_implemented` path remains
  on the stable surface.

## Deprecation and breaking changes

- A public name is deprecated by an alias + `DeprecationWarning` + a documented
  window, never by immediate removal.
- A breaking change requires an ADR, a major/minor version bump per SemVer, and
  a deprecation period for the previous name.
- Strict façades are never shadowed or renamed; their behavior is pinned by
  `tests/compat/` fixtures.
