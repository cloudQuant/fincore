# Iteration49 G1d: bounded interleaved-string fingerprint chunks

G1d extends the exact-v1 object-column path only for long, highly interleaved
arrays of exact built-in strings. It keeps the legacy encoder as the byte
oracle and fallback. Sampling is a routing hint only: before each chunk reaches
`pandas.factorize`, every cell in that chunk must pass `type(value) is str` and
the bounded string-length check. Mixed values and subclasses use the existing
adjacent-run path; custom hash/equality methods are never called by this path.

The fast path handles at most 8,192 cells at a time and emits no more than
1 MiB of framed output per write. It retains only the current chunk's factorize
codes, uniques, counts and gathered byte references, then releases them before
the next chunk. Strings longer than the bounded leaf threshold, chunks with
too many uniques, and chunks that exceed the output budget use the existing
stepwise/run writer. Low-adjacent-run string columns continue to use the
existing multiplication path. The v1 tuple header, leaf envelopes, dtype,
index, attributes, flags and all other fingerprint inputs remain unchanged.

The pre-implementation public payload diagnostic used the saved 500-symbol
G1c example026 `labels.csv` and `signals.csv` from
`D:/temp/iteration49-7col-g1c-p1-b-026/example/evaluation/1a376bfb5e31e2bcf9689cef4d51467be5fff24ad084604753d24d3d523448da/`.
Each contained 364,500 rows. Their interleaved asset columns had 500 distinct
labels and 364,500 adjacent runs; the other examined object columns had only
one or three adjacent runs and therefore already benefit from G1c. On those two
columns, the bounded factorize prototype reduced isolated legacy-run length
and emission CPU from 1.4219 s to 0.2031 s (about 1.22 s saved), with exact
22,599,018-byte output and matching body hashes. Its largest output chunk was
253,952 bytes and explicit array scratch was about 404,524 bytes, excluding
pandas hashtable internals. This is a one-run proxy diagnostic, not paired
pipeline evidence or an end-to-end speedup claim. CSV `time` columns are
textualized and were not counted as actual object-array hotspots.

## Scope

- `fincore/factor_analysis/models.py` — sampled routing and bounded,
  byte-identical exact-string chunk handling.
- `tests/factor_analysis/test_fingerprint_streaming.py` — v1 oracle,
  chunk/write bounds, low-run routing and mixed/custom fallback checks.
- This plan record.

## Accepted development round (2026-10-23)

Root ran `python -m pytest -p no:asyncio -q
tests/factor_analysis/test_fingerprint_streaming.py`: 55 passing progress
markers, 100% completion and exit zero, without warning lines. The captured
output did not include a pytest summary footer. Independent static review
also checked the four additional chunk-budget and fallback cases.

Three serial AB/BA/AB pairs ran all three complete 500-symbol, 243-date
examples (18 processes, nine output comparisons). Every full-output verifier
passed, including exact values, ordering, semantic IDs and report models.
Both arms used Backtrader G3 `b4c965bbd52438c43d941629742457702215c0e6`;
the FinCore baseline was `41cf7c918a8249dce06c1e7c1f6d50137284ace8`,
and the candidate changed only `models.py` production code.

| Example | Paired median wall reduction | Ratio dispersion | CPU reduction | Peak change |
| --- | ---: | ---: | ---: | ---: |
| 025 | 3.129% | 1.511% | 2.581% | -0.195% |
| 026 | 2.407% | 3.264% | 3.484% | +0.050% |
| 027 | -0.003% | 1.549% | 0.087% | -0.863% |

Dispersion is `(max(B/A)-min(B/A))/median(B/A)` across pairs. This accepts
the bounded encoding improvement for 025/026 and compatibility for 027;
it makes no material 027 speedup claim. The candidate source SHA-256 is
`86fc25db4da460281bfa861dd0f2e202de82dafc5d1ece3b646856aa6c8e9d64`.
Backtrader's Iteration49 measurements retain `7col-500-g1d-pairs.json`,
the raw run/verifier archive and the measured-source archive.

The 5x end-to-end target and 1 GB full-universe process-tree ceiling remain
unproven until the required complete full-universe runs pass. The scoped
dev commit and postcommit receipt are recorded separately after this file
is accepted.
