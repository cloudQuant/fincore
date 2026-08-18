# Performance

Performance work in fincore follows a reproducible methodology (see
`docs/quality/performance-methodology.md` for the full protocol). The short
version:

1. Every workload is deterministic — a fixed seed and a SHA256 input digest so
   a run can prove it measured the same input across commits and platforms.
2. Baselines are platform-labelled (`linux-x86_64`, `darwin-arm64`) and never
   compared across unmatched platforms.
3. An optimization is accepted only when it preserves numeric and shape
   semantics (unchanged output digest) *and* improves a profiled case.

## Profiling a hotspot

```sh
python scripts/profile_hotspots.py --scenario medium --output build/hotspots-before.json
```

This writes a JSON report (top cumulative functions, wall seconds, cold-import
seconds, peak RSS) and a Markdown summary beside it. Run it again after an
optimization and diff the pair.

## Factor benchmarks

Factor-analysis performance is release-blocking only when a matching approved
platform baseline exists. Platform baselines live in
`benchmarks/factor-analysis-baselines/`; a pending (candidate-only) baseline is
never selected, so an absent approval is explicit rather than a silent pass.
See `docs/quality/factor-benchmark-approval.md` for the promotion protocol.
