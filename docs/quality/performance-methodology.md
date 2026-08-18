# Performance methodology

Every proposed optimization must have a before profile, a fixed workload, a
numerical digest, and a success budget. This page documents the measurement
rules so results are comparable and reproducible.

## Workloads

`benchmarks/workloads.py` builds deterministic workloads in three sizes
(`small`, `medium`, `large`) with a fixed seed. Each workload records its
expected row count and a SHA256 input digest, so a profile run can prove it
measured the same input across commits and platforms. Digests depend only on
the seed and size.

| Workload | Domains |
| --- | --- |
| `factor_panel_workload` | factor preparation, quantiles, IC, weights |
| `single_series_workload` | scalar metrics (Sharpe, volatility, …) |
| `rolling_returns_workload` | rolling metrics |
| `transactions_workload` | FIFO transaction round trips |
| `report_workload` | report model computation |

## Measurement protocol

1. **Warmups**: at least 2 before the timed region, so JIT/allocator warm-up
   does not count as workload time.
2. **Median of N**: at least 5 repeats; report the median, not the mean.
3. **Fresh subprocess**: cold import time and peak RSS are measured in a fresh
   subprocess so the profiler's overhead cannot pollute them.
4. **Platform separation**: baselines are platform-labelled
   (`linux-x86_64`, `darwin-arm64`). Never compare across unmatched platforms.
5. **Noise rules**: absolute slack absorbs sub-second and sub-MiB noise; a
   regression must exceed both the relative budget and the absolute slack.

## Profiling

`scripts/profile_hotspots.py` runs a bounded workload under `cProfile`, writes
a JSON report with the top cumulative functions, and renders a human Markdown
summary beside it:

```sh
python scripts/profile_hotspots.py --scenario medium --output build/hotspots-before.json
```

The JSON includes wall seconds, cold-import seconds, peak RSS, and the workload
digest, so the before/after pair is comparable.

## Acceptance

An optimization is accepted only when it (a) preserves numeric and shape
semantics (unchanged output digest), (b) improves a profile-corpus case, and
(c) keeps the `tests/compat` suite green.
