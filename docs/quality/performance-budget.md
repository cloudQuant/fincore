# Performance Budget

Iteration 0042 — the platform layer (catalog dispatch, invocation pipeline,
DAG executor) must not add order-of-magnitude overhead to a single scalar
metric.  The budgets below are the fixed overhead ceilings; they are enforced
by `scripts/check_performance.py` and the benchmarks in `benchmarks/`.

## Principles

1. **Semantics first.** Every benchmark asserts an output digest/tolerance
   against the reference kernel before comparing wall time or RSS.  A result
   that is not numerically equivalent is a failure, not a "performance pass".
2. **Multi-scale.** Workloads are measured at small/medium/large sizes; a
   single small-sample mean is not evidence.
3. **Clean, same-platform baseline.** Baselines are recorded on a clean,
   same-platform, approved commit with full repeats; candidate-only artifacts
   are never a release gate.

## Fixed overhead budgets

| Layer | Operation | Budget (p95) |
| --- | --- | --- |
| Catalog dispatch | resolve + invoke one scalar metric | ≤ 500 µs |
| DAG executor | plan + execute a 3-node chain | ≤ 1 ms |
| Snapshot | copy-on-ingest a medium series | ≤ 10 ms |

## Regression thresholds (vs approved baseline)

| Metric | Allowable regression |
| --- | --- |
| median wall time | ≤ 10% |
| p95 wall time | ≤ 15% |
| peak RSS | ≤ 10% |

## Backend policy

- pandas/NumPy is the reference backend (always available).
- Optional backends (Array API, compiled) cover only dense kernels that do not
  depend on labels, timezones, or calendars.
- Optional backends are opt-in and fall back to the reference; results carry
  the backend name/version.
- A backend is introduced only when a workload and profile prove a ≥ 1.5×
  wall-time or ≥ 30% RSS improvement at equal semantics.
