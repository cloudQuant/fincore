# ADR-0042: Unified Operation Model

- **Status:** Accepted
- **Date:** 2026-08-21
- **Iteration:** 0042
- **Supersedes:** implicit per-surface registries (see Context)

## Context

fincore exposes several public surfaces that have evolved independently:
`fincore.empyrical` (frozen empyrical 0.6.0), `fincore.pyfolio` (pyfolio 0.9.6
profile), `fincore.alphalens` (alphalens cloudQuant 0.4.0 source-shaped), the
enhanced `fincore.metrics` / flat API, and the risk / simulation / attribution /
factor-analysis / report / optimization domains.  Each surface maintains its
own notion of "which operations exist", "what their inputs mean", and "how
stable they are", which drifts.

The audit (iteration 0042) established that the numerical errors (Task 0) and
release evidence (Task 1) are the immediate priorities, but that a single
**semantic authority** is required before any stability claim can be made for
the enhanced layer.

## Decision

1. Introduce an **immutable `OperationCatalog`** as the single semantic
   authority.  An `OperationDefinition` records `operation_id`,
   `semantic_profile`, `domain`, `canonical_name`, aliases, stability, input
   and output contracts, kernel reference, optional extra, determinism, rng
   policy, provenance, and semantic version.
2. A `PublicBinding` maps an `operation_id` onto a `public_path` (module,
   surface), a profile, a signature, an adapter, and a result projection.  A
   logical operation is defined **once** and bound onto multiple surfaces
   (strict façade, enhanced module, class method, flat metric).
3. The existing `METRIC_REGISTRY`, workflow specs, and capability inventory
   remain as **read-only projections** generated from the catalog during the
   migration; they are not rewritten wholesale (Task 3).
4. Strict compatibility surfaces do **not** route through enhanced validation
   or stateful classes.  They share only the raw kernels and the orchestration
   protocol beneath both layers.
5. The high-level result state is the discriminant
   `Success | Unsupported | Failed` (Task 5).  Direct scalar return shapes are
   frozen; the discriminant envelope is exposed only through approved
   high-level APIs (`execute()`, context/report/risk/factor/optimization).

## Consequences

- One `operation_id + semantic_profile` has exactly one `OperationDefinition`;
  each public path has exactly one `PublicBinding`.
- Capability inventory, API map, docs tables, and deprecation map are all
  generated from the catalog — no hand-maintained duplicate sources.
- `import fincore` must not import heavy optional dependencies, and the strict
  façades' observable behavior (C0–C4) must not change.

## See also

- `docs/architecture/financial-semantics.md`
- `docs/architecture/public-surface-policy.md`
- `docs/architecture/public-api-map.md`
