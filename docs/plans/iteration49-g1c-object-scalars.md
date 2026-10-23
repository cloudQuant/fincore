# Iteration49 G1c: batch repeated object-column string and integer leaves

**Status: fixed-500 paired measurement complete.** This record describes a
small follow-up to G1/G1b. The implementation and focused tests are frozen;
Root completed the 025-only fourth and fifth pairs, and all five
complete-output verifiers passed.

G1c targets the production profiling cost left by G1b: for a pandas object
column of ordinary scalar values, generic per-cell stream-length dispatch and
framed-part writes still incur millions of Python calls. The fast path is
limited to an exact NumPy object array whose cells pass the existing standard
scalar allowlist. Adjacent equal values are coalesced only for exact built-in
`str` and exact built-in `int`, whose v1 encodings are determined by their
values and types. Length calculation multiplies the encoded leaf length by the
run count; emission repeats the already-framed leaf in bounded batches. The
path does not build a column-sized factorization or cache. Other allowed
scalars keep the ordinary scalar encoding path, and unsupported/custom cells
retain the existing compatibility fallback. Legacy v1 bytes and the old
encoder oracle are unchanged.

The implementation stays within the existing fingerprint stream limits: it
keeps no per-cell encoded-byte array, and repeated framed leaves are appended
to the bounded output batch before one write. Exact type checks prevent string
or integer subclasses from taking the optimized path. Byte-oracle coverage
compares streamed bytes and SHA-256 with the legacy encoder for object arrays,
nested values, repeated and alternating strings/integers, Unicode, integer
boundaries, mixed standard scalars, and unsupported custom objects. The tests
also exercise an actual research/report-shaped payload and batch boundaries.

Only the following Fincore files are in scope for G1c:

- `fincore/factor_analysis/models.py` — bounded exact-string/integer object
  scalar run length and emission.
- `tests/factor_analysis/test_fingerprint_streaming.py` — byte-oracle,
  fallback, batching and production-shaped regressions.
- `docs/plans/iteration49-g1c-object-scalars.md` — this round record.

The candidate pins at the time of the paired screen are 129,150 bytes,
SHA-256 `78c9cc243cc5ee163c4231b365bad11f2d835427a49c5c4ded65ec69f9420404`
for `models.py`, and 25,740 bytes,
SHA-256 `94549d3cfc4231e3fea57fc90f7ba11a64abba3f48e8a4cba685aafc210c670b`
for the focused test file.

Root's G1c focused test command was
`C:/anaconda3/python.exe -m pytest -p no:asyncio -q tests/factor_analysis/test_fingerprint_streaming.py`;
it exited zero after 47 test dots reached 100% in 6.74 seconds; the captured
output had no numeric pytest summary footer. Ruff on the implementation and
focused test files exited zero (`All checks passed`). Independent static review
returned GO. A separate incorrect-path pytest launch collected no tests and is
not included in these results.

## Fixed-500 paired screen

The screen uses the same seven-column, 500-symbol by 243-date input and full
025/026/027 example outputs. Backtrader is fixed at G2a snapshot commit
`02ee098a972164f3d49d087713f3efc52729992a`; Fincore A is the accepted G1b
snapshot at commit `2ec366c033bb6b4c47fe1880d5a3f4ec5e407d64`, while Fincore B
is the same existing `dev` checkout at that parent commit with the pinned G1c
candidate files above. Every run is a fresh process and every pair uses the
complete-output verifier. Pair order is AB, BA, AB, BA, AB. Pairs 1–3 cover all
three examples; pairs 4–5 extend only 025 after its initial three-pair relative
dispersion exceeded the 5% extension threshold. For those last two verifiers,
the already verified pair-1 026/027 outputs are reused as controls; they are
not new measured comparisons.

The initial 025 measurements were:

| Pair | A wall (s) | B wall (s) | Paired B/A wall |
| --- | ---: | ---: | ---: |
| 1 | 27.7476489 | 25.2721991 | 0.910787 |
| 2 | 27.8312342 | 25.0551433 | 0.900253 |
| 3 | 26.9646762 | 25.5859646 | 0.948870 |

The three-pair median B/A wall ratio was `0.91078704`. Relative dispersion,
defined as `(max(pair ratios) - min(pair ratios)) / median(pair ratios)`, was
`5.3379%`, triggering the two additional 025 pairs. The final five-pair
wall/CPU/peak medians and dispersion, along with the 026/027 three-pair
statistics, are:

| Example | Pairs | Wall A → B median (s) | Paired B/A wall (median; dispersion) | CPU A → B median (s) | Paired B/A CPU (median; dispersion) | Peak A → B median (bytes) | Paired B/A peak (median; dispersion) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 025 | 5 | 27.649 → 25.586 | 0.927595; 5.241% | 26.391 → 24.516 | 0.921315; 1.961% | 345,042,944 → 342,761,472 | 0.994439; 0.867% |
| 026 | 3 | 63.946 → 60.427 | 0.941266; 0.674% | 61.813 → 57.375 | 0.928210; 2.531% | 420,425,728 → 420,642,816 | 1.000516; 0.949% |
| 027 | 3 | 40.255 → 40.246 | 0.996031; 1.607% | 38.625 → 38.766 | 1.001615; 1.616% | 321,417,216 → 318,328,832 | 0.991169; 0.803% |

025 showed a median wall reduction of `7.24%`, with CPU down `7.87%` and
similar low peak memory. Its five-pair wall-ratio dispersion remains just
above the `5%` stability threshold (`5.241%`), so treat that wall estimate as
unstable rather than as a stable speedup. The CPU direction agrees and its
relative dispersion is `1.961%`. 026 showed a repeatable median wall reduction
of `5.87%` and CPU reduction of `7.18%`; wall-ratio dispersion was `0.674%`,
and median peak was effectively unchanged (`+0.052%`). 027 has no material
speed gain: its median paired wall ratio was `0.996031` (wall time `0.397%`
lower) and CPU ratio was `1.001615` (CPU `0.161%` higher), so no speed gain is
claimed for that example.

All 22 measured processes completed and all five complete-output verifier
reports passed. Those reports contain 15 comparison checks in total, but p4
and p5 reuse the already verified p1 026/027 controls. The distinct measured
set is therefore 11 comparisons: five for 025 and three each for 026/027.
The pinned machine summary is
`D:/temp/iteration49-7col-g1c-paired-summary.json` (23,103 bytes,
SHA-256 `032527c523acecfe836682608753721a40aa07254b91d6fff8065bb3bac5fe10`);
it includes each run's result JSON pin and all five verifier pins.

The G1c screen is a 500-symbol paired measurement only. It does not establish the
5,471-symbol whole-process-tree memory ceiling of 1 GB, installed-wheel or
release acceptance, or the overall 80% runtime-reduction target.

The paired summary records exact run/result/verifier pins; the table above
reports the paired measurement medians and dispersion.
