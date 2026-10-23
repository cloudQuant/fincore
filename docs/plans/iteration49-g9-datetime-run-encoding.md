# Iteration 49 G9: bounded adjacent datetime run encoding

## Scope

The streamed fingerprint writer for timezone-aware datetime arrays currently
vector-formats and scatters every timestamp tick in each bounded chunk. This
round adds a chunk-local path for inputs with long adjacent runs of the same
timestamp. A run emits the same v1 timestamp (or pandas-nat) leaf frame once,
then repeats its length-prefixed bytes in original order. The existing vector
writer remains the path for short and unique-heavy chunks.

The change is internal to fincore/factor_analysis/models.py. The fingerprint
version, public APIs, length calculation, index handling, and datetime source
dtype remain unchanged. No timestamp values are sorted or deduplicated across
separated runs or chunks.

## Semantic and memory bounds

The run path compares adjacent signed int64 ticks for equality; it does not
subtract or normalize them. Thus negative pre-epoch ticks, each pandas unit,
timezone payload, and the int64 NaT sentinel keep their existing encoding.
Only the existing exact DatetimeArray route reaches this writer. Regular
timestamp leaves are built with the same v1 timestamp envelope and unit,
decimal tick, and timezone byte parts as the legacy encoder. NaT reuses the
legacy pandas-nat leaf. Repeating a frame preserves exact row order and
length-prefix bytes.

Run detection and output batching are chunk-local. A chunk uses the repeated
path only when it has at least eight rows and no more than one run per four
rows; unique-heavy chunks retain the vectorized writer. Run output is flushed
at the existing one-MiB leaf-batch budget, below the established eight-MiB
writer limit. Temporary run boundaries, encoded leaves, and repeated blocks
are discarded after each chunk. There is no full-column object conversion,
cross-column cache, or new fingerprint identifier.

## Regression coverage

The focused tests compare streamed bytes and final fingerprints with the
independent legacy _fingerprint_bytes encoder. Cases cover repeated and
non-adjacent timestamp runs, all-NaT input, negative ticks at ns/us/ms
resolution, UTC and America/New_York payloads, a run crossing a forced small
chunk boundary, unique-heavy vector-path routing, DataFrame and Series
payloads, and maximum writer-call size.

## Verification status

Implementation and focused tests are prepared on FinCore dev; no Python
command, import, test, lint, or benchmark has run in this round. Root's serial
performance work is active, so source QA and the counterbalanced 026 screen
remain pending until Root opens a QA slot. The earlier fixed-500 profile
reported seven datetime writer calls at 1.296 s self / 2.489 s cumulative.
Those are profile observations, not a projected saving; the maximum possible
self-time saving is limited and actual repeated-run gain must be measured.
No full-universe or fivefold claim is made.

## Measured fixed-500 acceptance

65个源码测试通过（65个progress markers，exit 0，无数字footer；1条既有Hypothesis配置warning）；18次完整500股运行、9项唯一比较、3份完整verifier均PASS。025：正向收益超过本轮配对离散范围，且离散度不超过5%。 026：正向收益超过本轮配对离散范围，且离散度不超过5%。 027：配对离散度超过5%，不宣称稳定提速。

| 示例 | A/B main中位数(s) | 配对降时 | A/B比率离散度 | CPU配对降幅 | 候选峰值中位数(MB) |
|---|---:|---:|---:|---:|---:|
| 025 | 20.032/18.584 | +8.072% | 3.395% | +7.506% | 309.993 |
| 026 | 43.654/41.689 | +4.920% | 4.511% | +3.876% | 348.430 |
| 027 | 26.148/26.348 | +3.822% | 5.005% | +2.106% | 299.540 |

配对顺序AB/BA/AB；降时按逐对1-B/A中位数，离散度按(max(A/B)-min(A/B))/median(A/B)。峰值取OS主进程high-water与0.5秒进程树RSS采样值的较大值。原始结果、验证器、测量源码和QA见Backtrader measurements/7col-500-g9-datetime-pairs.json及关联ZIP。5471股80%降时和1GB峰值仍待组合版本验证。
