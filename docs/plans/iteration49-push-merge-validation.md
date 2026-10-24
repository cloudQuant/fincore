# Iteration 49 push integration (2026-10-24)

Repository: `D:/source_code/fincore`, branch `dev`.

This merge integrates local Iteration 49 commit
`09f8dd9ecc5edafc38fa37ddaa6dcfbac492de2b` with remote `origin/dev`
`4b8f31b1b59707480d5db9603704d83bbf7e6946`, preserving both histories.

The local bounded fingerprint implementation in
`fincore/factor_analysis/models.py` is retained unchanged. It supersedes the
remote thresholded streaming plan from `fc512cd`; the latter's removed private
interfaces are not reintroduced. Existing tests cover byte compatibility,
metadata, mapping order/collisions and bounded writes. The incoming test module
retains the additional dataclass-plus-Mapping dispatch regression against the
legacy byte oracle.

The remote empty-quantile and empty-IC report fix from `4b8f31b` is integrated
with its tests. Missing statistics remain unavailable values while the complete
report can still render.

## Validation

Using `C:/anaconda3/python.exe` with `PYTHONUTF8=1` and
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, the following focus passed **80 tests in
10.22 seconds**, exit zero, with no warnings reported:

```text
python -m pytest tests/factor_analysis/test_fingerprint_streaming.py tests/factor_analysis/test_factor_analysis_fingerprint_streaming.py tests/parity/test_factor_report_copy_reuse.py tests/parity/test_factor_report_html.py --junitxml=.codex-tmp/iteration49-push-merge-qa.xml
```

The first collection attempt with automatic third-party plugin loading failed
inside the installed `pytest_asyncio` plugin (`Package.obj` AttributeError),
before tests ran. Disabling unrelated plugin autoload resolved that environment
conflict; no application change was made for it.

This is merge regression validation, not another performance run. The final
500/5471-stock comparison remains pinned to FinCore `09f8dd9` and is documented
in Backtrader's `迭代49-因子分析性能优化/7列最终组合验收.md` at
`2c74e9f116468308b709eac10bef798252678504`. No new speed or memory result is
claimed for this merge. The user requested publishing the completed work and
no further optimization.
