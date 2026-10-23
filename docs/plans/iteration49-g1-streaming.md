# Iteration49 G1: byte-identical bounded v1 fingerprint encoding

Development is on existing `dev`, based on `f2fa33a91fbaceca41fea6894368dcfd9e7c367b`.
This round changes only `fincore/factor_analysis/models.py`, the new
`tests/factor_analysis/test_fingerprint_streaming.py`, and this record.

The encoder retains the exact legacy v1 frames and SHA-256 identities. It uses
bounded numeric, integer, datetime and scalar batches plus bounded leaf caches,
without constructing complete parent byte strings or objectifying whole frames.
Standard timestamp frames and digest updates are capped at 8 MiB. Unsupported
custom/extension inputs and an oversized individual timezone envelope fall back
to the legacy encoder; that fallback is not allocation bounded. Public model
snapshot/getter semantics are unchanged. The legacy encoder remains the oracle.

R1 failed actual 500-stock testing on speed and nested-list evaluation IDs; it
was not committed. R2 fixed those failures and all actual outputs matched, but
static review found a 10.7 MB datetime frame. R3 only tightens that frame bound
and adds a writer-level regression. Final source SHA-256 is
`61a154d92b5b6c3f509039f24f55c92c1c584af8a7420bb0244c6a44246c0025`; test SHA-256 is
`8701ade46a25c33a0c0735e691674314c16fbfd1f60ca6e38a0dd6892d8b2bec`.

Validation on 2026-10-23:

- R3 focused streaming tests: 36 passed in 2.38s, no warnings, using CPython 3.11
  and `python -m pytest -p no:asyncio tests/factor_analysis/test_fingerprint_streaming.py -v --tb=short`.
  R2's six-file research/report/consumer command exited zero; quiet progress had
  79 dots but no numeric footer. Ruff and `git diff --check` pass.
- Frozen Backtrader G0 on both arms, fincore G0 versus R3, same seven-column
  500-stock by 243-date input and complete example output. Three fresh-process
  pairs for 025/026; five for noisy 027. All 11 new exact output comparisons
  passed, including factor cells, evaluation tables/IDs/fingerprints and all
  actual report models. There were 22 serial benchmark processes.
- Paired median wall reductions: 025 35.32%, 026 25.63%; 027 -0.23%, with
  10.12% ratio dispersion, so no 027 speed gain is claimed. Candidate median
  peaks: 431,382,528 / 522,977,280 / 497,127,424 bytes. Peak paired reductions
  for 025/026: 53.21% / 57.94%; 027 peak +1.06%, CPU +0.86%.
- Actual 026 diagnostic: 10/13 fingerprints streamed, 3 fallback,
  22.938s CPU versus the G0 single diagnostic 48.344s. This is about
  2.11x, below the 8x stretch target. On its same actual result payload,
  tracemalloc extra peak fell from 729,090,852 to 21,606,737 bytes
  (97.04%), with both digests equal to the exported manifest.
  This is traced allocation, not process RSS; diagnostic timings are excluded
  from the formal pairs.

The complete input, source, paired measurements and comparisons are recorded in
Backtrader `docs/_internal/opts/requirements/迭代49-因子分析性能优化/measurements/7col-500-g1-r3-*.json`.
Final commit identities are recorded in that repository's development record.
This is a local source/500-stock acceptance, not installed-wheel, full-universe,
release or deployment acceptance. The whole-flow 80% reduction and 1 GB full
process-tree goals remain unmet. Continue with the next accepted small round.
