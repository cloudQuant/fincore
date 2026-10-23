# Iteration49 G1f: market-cardinality string fingerprint batches

The fixed-500 asset column passes the existing sampled-uniqueness routing
condition. A 5,471-asset column can exceed that condition and the 1,024-unique
chunk limit, despite being a repeated, highly interleaved builtin-string
column. It then misses the bounded batch path validated on the smaller sample.

This round removes only the sampled-uniqueness condition, raises the per-chunk
unique limit to the existing 8,192-row chunk bound, and raises the leaf cache
caps to 16,384 entries and 1 MiB of encoded bytes. The high-transition routing,
exact builtin-string checks, 256-character limit, 8,192-row chunks and 1 MiB
joined-output limit remain. Unsupported chunks retain their prior fallback.
The cache byte cap excludes Python container overhead; actual process memory
is measured separately. Fingerprint bytes and public identifiers must remain
unchanged.

Independent static review passed. Regression tests exercise real DataFrame
fingerprinting with 500, 5,471 and 8,192 distinct values, the legacy byte oracle,
cache hits, bounded writer calls and overlength fallback before factorization.
Source tests and the complete fixed-500 output screen are pending.

This is a dispatch-coverage fix for full-market cardinality. Its incremental
fixed-500 screen uses one A/B pair per example, because the 500-asset input
already reaches the old batched path; it makes no stable speed claim. The final
combined candidate still requires three complete AB/BA/AB fixed-500 pairs
against the same seven-column B0, then final 5,471-asset measurement. No
full-market time reduction or 1 GB process-tree acceptance is established here.

## Measured fixed-500 acceptance

Root fingerprint focus: 58 passing markers, exit0 (c6bb4a,7.77s); no numeric pytest footer. The initial test failure counted DataFrame column-label metadata as row-data fallback; the test now distinguishes them, retaining exact legacy bytes and chunk bounds. Six complete fixed500 runs and three unique output comparisons PASS; observed wall/CPU/peak ratios are within the nonregression screen. A single pair is not stable-speed evidence and does not establish full-market resource or 80% acceptance.

| 示例 | A/B墙钟中位数(s) | 墙钟配对变化 | CPU配对变化 | A/B峰值中位数(MB) |
|---|---:|---:|---:|---:|
| 025 | 24.101/23.816 | -1.181% | -1.678% | 318.616/310.890 |
| 026 | 51.301/50.612 | -1.342% | -2.046% | 349.049/351.633 |
| 027 | 34.961/32.731 | -6.378% | -7.395% | 308.171/305.689 |

Actual complete processes: 6; unique output comparisons: 3; full verifiers: 1. All output comparisons passed. Positive timing changes mean slower. Peak combines OS main-process high-water and 0.5-second sampled process-tree RSS.

Exact measured source, config, raw runs and verifier receipts are in Backtrader's `measurements/7col-500-g1f-market-pairs.json` and its paired raw/source archives. Final combined fixed-500 and full5471 targets remain pending.
