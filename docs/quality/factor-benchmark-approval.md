# Factor benchmark approval

Factor-analysis performance is release-blocking **only** when a matching,
reviewed, approved platform baseline exists. Otherwise its absence is explicit
rather than a silent pass.

## Platform baselines

Platform-labelled baselines live in `benchmarks/factor-analysis-baselines/`:

| File | Platform | Status |
| --- | --- | --- |
| `linux-x86_64.json` | Linux x86_64 | `candidate-only-not-release-approved` |
| `darwin-arm64.json` | macOS arm64 | `candidate-only-not-release-approved` |

Both are currently candidates. They record deterministic output digests and
shapes (identical across platforms because workloads use a fixed seed), but
their wall-time/RSS measurements have not completed the review protocol.

## Approval protocol

Promotion is a reviewed commit, never an automated CI overwrite:

1. Generate a clean candidate on the reference host with at least 2 warmups and
   5 repeats:
   `python scripts/run_factor_benchmarks.py --scenarios small-ci --warmups 2 --repeats 5 --output build/factor-candidate.json`
2. The kernel owner verifies output digests and the C2/C3 fixtures.
3. Track E reviews provenance (dirty=false), repeat variance, and the digest.
4. Only then set `approval.status="approved"`, `approved_by`, `approved_at`,
   and `reviewed_candidate_sha256` in the platform file, and commit it.

`scripts/compare_benchmarks.py` exposes `select_baseline(directory, platform_label)`
(returns only an approved matching-platform baseline) and
`list_candidate_baselines()`; a pending baseline is never selected.

## Current blockers

- Neither platform baseline has a kernel-owner review.
- Neither platform baseline has a Track E review.
- The Darwin candidate was captured with one repeat, no warmup, and dirty=true.
