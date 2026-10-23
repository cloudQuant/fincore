# Iteration 49 G6: reuse native UTC timestamp formatting during CSV export

## Scope

The labels and signals result tables repeat each panel timestamp across many assets. FinCore requires a UTC-aware panel index (`fincore/factor_analysis/research.py::_require_utc_index`) and builds both long tables by repeating `row_times`. Their `time` columns therefore retain the panel's `datetime64[ns, UTC]` dtype while the same timestamps recur in each output block.

G6 targets only those two CSV exports. It factorizes one bounded row chunk at a time, formats that chunk's unique timestamps with pandas' own `DatetimeArray._format_native_types(na_rep="", date_format=None)`, then gathers the returned strings back into row order. The chunk is written by pandas `to_csv` to the same serial file stream; the header is emitted once. The original result DataFrame and its columns, values, dtypes, and identity remain unchanged.

## Exactness and bounds

The fast path requires a nonempty table with a unique `time` column whose dtype is exactly `DatetimeTZDtype`, unit `ns`, and timezone `UTC`. It also requires the export's existing `index=False, encoding="utf-8"` options and pandas' native formatter. This matches the current labels/signals schema and avoids introducing a second timestamp formatter.

Pandas 2.2.3 uses a default CSV chunk budget of 100,000 cells. G6 uses the same cell budget to bound each temporary factorization and formatted object array: rows per chunk are `max(1, 100000 // column_count)`. No full-column codes, full-table object conversion, persistent cache, or custom CSV writer is created. For timezone-aware datetime arrays, pandas does not use the date-only inference path; formatting uses the array's timezone and resolution. Missing timestamps use the same empty representation as the existing CSV export.

If the table, dtype, formatter, or options do not match the gate, the helper returns before opening the destination and the existing full-table `DataFrame.to_csv` call handles the export. A chunk with all-distinct timestamps skips factorized string gathering and remains a native datetime chunk. Empty tables, naive or non-UTC time columns, other units, duplicate columns, and unsupported options use the existing export path.

## Verification evidence

The focused oracle checks optimized output byte-for-byte against pandas output, including nanosecond timestamps, date boundaries, `NaT`, NaN, Unicode, quoting, and embedded newlines. Separate cases cover all-`NaT` and all-unique chunks. The export-entry test asserts that labels and signals actually take the gated path and compares both emitted files with direct pandas `to_csv` bytes. Fallback tests cover unsupported time dtypes and CSV options.

The corrected R2 proposal is pinned by SHA-256 `E20EC78B90EA07A5E434A46F65348A5D5BEA5E102AE292175B763DD77273BE05` at `D:/source_code/backtrader/.codex-tmp/iteration49_g6_datetime_export_r2.patch`. Independent static review returned `GO_STATIC_ONLY`. The earlier R1 static draft contained an unsupported `unique.array` access because `pandas.factorize` returns an ExtensionArray for this input; R1 was not applied or run. R2 uses the returned unique ExtensionArray directly. Root's focused source QA passed 33 tests with one known Hypothesis warning.

A preliminary one-pair A/B screen and full-output verifier are diagnostic only; they do not establish a stable performance gain. Root's serial three-pair, 18-run screen is pending. Final paired metrics and acceptance disposition: **pending Root completion**.

## Measured fixed-500 acceptance

Root focused source QA: 33 passed, one existing hypothesis collection warning, exit0 (41843a); independent static GO. Eighteen complete fixed500 runs, nine unique output comparisons, three complete verifiers PASS (completion32b1c4). 026 paired wall reduction5.796%, dispersion4.927%; its CPU reduction6.275%, dispersion5.366% is an unstable observation. 025 wall reduction9.202%, dispersion5.711% is also not a stable-speed claim. 027 wall reduction0.319%, dispersion1.427% is a nonregression control. Median captured memory does not regress. Separate initial 026 screen passed all13 CSV byte comparisons and the full verifier; reused025/027 controls were not new runs. Full5471 and combined80%/1GB acceptance remain pending.

| 示例 | A/B墙钟中位数(s) | 墙钟配对变化 | CPU配对变化 | A/B峰值中位数(MB) |
|---|---:|---:|---:|---:|
| 025 | 22.825/20.962 | -9.202% | -9.110% | 314.192/313.455 |
| 026 | 47.869/45.202 | -5.796% | -6.275% | 351.830/351.359 |
| 027 | 32.505/32.813 | -0.319% | -0.350% | 307.360/305.529 |

Actual complete processes: 18; unique output comparisons: 9; full verifiers: 3. All output comparisons passed. Positive timing changes mean slower. Peak combines OS main-process high-water and 0.5-second sampled process-tree RSS.

Exact measured source, config, raw runs and verifier receipts are in Backtrader's `measurements/7col-500-g6-utc-pairs.json` and its paired raw/source archives. Final combined fixed-500 and full5471 targets remain pending.
