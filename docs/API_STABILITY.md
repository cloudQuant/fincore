# API Stability Policy

This policy describes the Fincore **0.5.0** breaking API.

## Public contract

The supported public surface is the list of domain namespaces exported by
`fincore` and their documented leaf modules:

```python
import fincore

print(fincore.__all__)
```

Domain leaf functions and models own their contracts. For example, the Sharpe
ratio is `fincore.metrics.ratios.sharpe_ratio`, a portfolio report builder is
`fincore.report.portfolio.compute.build_portfolio_report`, and catalog-backed
execution is provided by `fincore.runtime.engine`.

The source and wheel tests verify that retired package-shaped modules, façade
classes, root callable aliases, registry/dispatcher modules, and old extras
are absent. Their successful import is a 0.5 defect, not a compatibility
feature.

## Compatibility boundary

0.5 makes no signature, type identity, exception-text, state-binding, class
MRO, or import-path compatibility promise for APIs retired before 0.5. It also
makes no promise to retain an alias when a capability's owner moves; the
canonical API map and migration guide are the source of truth for public
locations during this pre-1.0 series.

## Versioning

- `0.5.0` is a breaking release.
- Any public domain-path or semantic change before 1.0 is recorded in the
  changelog and migration guide.
- Capability-level semantic tests, immutable snapshots, and numerical oracle
  scenarios are evidence for a result; they are not an API-compatibility claim
  for retired surfaces.

## Maintainer requirements

A public change must update its owning module documentation, operation catalog
entry, capability scenarios, and packaging/documentation examples. New root
aliases, profile projections, dynamic compatibility dispatch, or old extra
names are not permitted.
