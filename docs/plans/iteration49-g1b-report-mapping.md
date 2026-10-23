# Iteration49 G1b: stream report mappings with integer keys

This small follow-up to G1 admits exact built-in `int` keys alongside exact
`str` keys in ordinary `dict` and `MappingProxyType` values. The legacy encoder
and v1 bytes are unchanged. Mapping entries remain ordered as the legacy
encoder orders their complete encoded-entry bytes. A repeated encoded-key
prefix still selects the compatibility fallback, because legacy ordering then
depends on the encoded value. Custom mappings, key subclasses, and other
unsupported keys retain the legacy fallback. The repository-owned
`_FrozenMapping` path is unchanged.

The production payload that motivated this change is
`analysis._analyze_factor`'s `result_payload["quantile_turnover"]`:
`_quantile_turnover_tables` creates an ordinary `dict[int, DataFrame]` keyed by
the validated turnover lags. Other result dictionaries use string keys, and
rank-autocorrelation lags are DataFrame columns. The G1 stream encoder rejected
the turnover keys, so the three report result fingerprints fell back to the
legacy whole-object byte encoder.

Only these two implementation/test files are in scope:

- `fincore/factor_analysis/models.py` — exact `str`/`int` key eligibility and
  unchanged collision/custom-key fallback.
- `tests/factor_analysis/test_fingerprint_streaming.py` — byte-oracle and
  production-payload regressions.

The implementation pins reported for this round are 126,060 bytes,
SHA-256 `8f4aa2e676612beb10bf9c9e42aa6780c0f11101fd1169fa9ddb0a9f426676ea`
for `models.py`, and 22,206 bytes,
SHA-256 `36d8b7920dc187fe07900098a8f18096a8024003923ab159ad9ca6b8c508c449`
for the focused test file.

Coverage includes mixed and nested string/integer maps, negative and very large
integer keys, full encoded-entry ordering against `_fingerprint_bytes`, custom
integer-subclass and custom-object keys, duplicate encoded-key fallback, and a
real `prepare_factor_data` plus `analyze_factor` result payload. The production
test checks that turnover keys are exact integers and that streamed bytes and
SHA-256 match the legacy encoder. Root reports the focused suite passed with
exit zero and 41 test dots; the independent static review returned GO. Root's
later paired measurement and pipeline diagnostic are recorded below.

## Local verification and paired measurement

Root's focused fingerprint suite passed 41 tests with exit zero; Ruff passed,
and independent static review returned GO. Root then ran the formal G1b paired
measurement against the same Backtrader snapshot (`02ee098a972164f3d49d087713f3efc52729992a`)
and fincore parent (`4c8e375be6b5889686882dd753b3180182e58f78`), comparing the
accepted fincore snapshot with the candidate source. It comprised three
AB/BA/AB pairs for each of 025, 026, and 027 (18 measured processes total).
Every one of the nine complete-output comparisons passed, with no extension
pair required.

The table reports median absolute measurements for A and B, followed by the
median paired B/A ratios from Root's summary. Peak is reported in bytes. The
paired ratio medians are computed per pair, so they can differ slightly from
the ratio of the two displayed absolute medians.

| Example | Wall A → B (s) | CPU A → B (s) | Peak A → B (bytes) | Median B/A wall, CPU, peak |
| --- | ---: | ---: | ---: | ---: |
| 025 | 31.775 → 27.691 | 30.594 → 26.531 | 367,292,416 → 344,612,864 | 0.871, 0.871, 0.939 |
| 026 | 76.586 → 65.001 | 73.703 → 62.313 | 418,574,336 → 419,434,496 | 0.850, 0.845, 0.996 |
| 027 | 45.112 → 40.494 | 43.125 → 38.984 | 387,670,016 → 318,509,056 | 0.900, 0.901, 0.822 |

This is `PASS_G1B_3PAIR_MEASUREMENT_ONLY`: a paired 500-symbol measurement, not
a full-market or final Iteration 49 acceptance, and not proof of the separate
1 GB whole-5,471-symbol process-tree limit. The preserved summary is
`D:/temp/iteration49-7col-g1b-paired-summary.json`.

## Production-shaped pipeline diagnostic

The separate 026 pipeline diagnostic recorded all thirteen fingerprint calls
as streamed, including the actual research result payload; none fell back to
the legacy encoder. Its report timers measured 11.318 seconds of inclusive
`report_analysis` wall time and 12.759 seconds for `report_total`. The earlier
R3 diagnostic measured 22.935 and 24.422 seconds respectively, so these
diagnostic observations show lower report times on that run. The diagnostic
also measured `research_total` at 18.330 seconds. These are diagnostic-only
observations, not the paired acceptance result or a fingerprint-only timing.
The file's before/after source hash was
`82f6539062d8d7121a4dff65ed2d412775cf2682cd935814c680096796325a81` in both
fields, and it measured models source hash
`8f4aa2e676612beb10bf9c9e42aa6780c0f11101fd1169fa9ddb0a9f426676ea`.
The diagnostic is preserved at
`D:/temp/iteration49-7col-g1b-026-pipeline-diagnostic/pipeline-diagnostic.json`
(SHA-256 `7a0d874223404180dba51d610be30a78d13084f32006c14b5d3a24e9b8d43202`).
