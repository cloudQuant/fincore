# API reference

The 0.5 API is organised by analytical responsibility. Package namespaces do
not duplicate leaf functions or install aliases; import the operation from its
owning module.

```text
fincore/
├── metrics/          quantitative metric kernels
├── performance/      cash-flow and return semantics
├── portfolio/        positions, transactions, capacity, round trips
├── report/           report models, builders, renderers
├── factor_analysis/  factor research workflow layers
├── attribution/      allocation and factor attribution
├── risk/             risk models and validation
├── optimization/     allocation optimisation
├── simulation/       simulation and scenarios
├── data/             provider and snapshot boundaries
├── extensions/       immutable extension discovery/snapshots
├── runtime/          catalog, session, errors, artifacts
└── viz/              explicit visualisation backends
```

Use the module pages in this section as the public API contract. Private
modules prefixed with `_` are implementation details.
