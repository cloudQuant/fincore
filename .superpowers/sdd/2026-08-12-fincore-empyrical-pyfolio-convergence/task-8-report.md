# Task 8 Report: compute-once report models and side-effect-free renderers

## Outcome

Task 8 separates the report computation stage from the rendering stage and
eliminates every runtime side effect owned by the report pipeline.

- `fincore/report/model.py` introduces `ReportModel` (a dict subclass, so all
  existing mapping-style consumers keep working) and the frozen-dataclass
  `SectionModel` view. `classify_sections()` groups raw entries by shape:
  scalar mappings become metric blocks, `pd.Series` become series,
  `pd.DataFrame` become tables, and scalars/tuples/short text become meta.
  `compute_sections` now returns a `ReportModel`, so `isinstance(sections, dict)`
  stays true and no existing renderer/test changed shape.
- `generate_html(..., model=None)` and `generate_pdf(..., model=None)` accept a
  precomputed model and never compute statistics when one is supplied
  (compute-once, render-many). Rendering shallow-copies the model, so the
  caller's model is never mutated (`_title` is no longer injected into it).
- `create_strategy_report(..., return_result=False)` is the enhanced-report
  entry point: when True it computes the model once, passes it to the
  renderer, and returns `ReportArtifacts(backend=..., files=[...], html=...,
  model=...)`; the default returns the output path exactly as before.
- `ReportArtifacts` (same object as Task 7) gains an optional `model` field;
  `close()` lifecycle is unchanged. The enhanced-only
  `Pyfolio.create_full_tear_sheet(..., return_result=False)` (threaded through
  `fincore/tearsheets/sheets.py`) returns a `ReportArtifacts` owning every
  matplotlib figure created during the run; the default return stays `None`
  and the strict `legacy_pyfolio` wrappers are untouched (no signature or
  warning changes — their pinned oracle never sees the new kwarg).
- `fincore/utils/common_utils.py` gains the frozen-dataclass `ExportConfig`
  (`output_dir`, optional `filename`, `resolve_path()` → 
  `strategy_performance_{name}.xlsx`). `print_table` writes a file only when
  the keyword-only `export=` is supplied and then returns `ReportArtifacts`
  with the owned file; default and `run_flask_app=True` calls still display
  only and write nothing, anywhere. `ExportConfig` is exported from
  `fincore.utils`.
- No `matplotlib.use()` (or any import-time backend call) existed in the
  current source; the RED tests now pin that contract for
  `fincore.pyfolio`, the lazy `Pyfolio` class trigger
  (`fincore/_pyfolio_impl`), and `fincore.tearsheets.sheets`. These were
  already green before implementation and are the regression guard.
- `fincore/report/render_pdf.py` moves all temp files (intermediate HTML and
  the pre-bookmark PDF) into a `tempfile.TemporaryDirectory(prefix=
  "fincore-report-")`; the browser is closed in a `finally`, and the context
  manager removes everything on success and every failure path (playwright
  startup crash, `page.pdf` crash, HTML-generation validation error).
- Offline ECharts: `fincore/report/assets/echarts.min.js` (official 5.5.0
  dist, 1,029,203 bytes, contains `version:"5.5.0"`, zero `</script`
  sequences so it is safe to inline) is vendored and embedded directly into
  generated HTML via `load_echarts_js()` (lru-cached). The three CDN
  fallback script tags are removed; rendering works with `socket.socket`
  monkeypatched to raise. Package data updated in `pyproject.toml`
  (`fincore = ["py.typed", "report/assets/*.js"]`) and `MANIFEST.in`
  (`recursive-include fincore *.js`); a wheel build confirms
  `fincore/report/assets/echarts.min.js` lands in the wheel.

## TDD evidence

Four RED test files were written before implementation:

```text
tests/test_report/test_offline_report.py     3 failed (CDN URLs), 1 passed (asset control)
tests/test_report/test_pdf_cleanup.py        3 failed (temp-file leaks), 1 passed (success control)
tests/test_pyfolio/test_backend_side_effect.py  3 passed (already clean -> pinned guard)
tests/test_utils/test_export_destination.py  1 collection error (ExportConfig missing)
```

The pdf render-failure leak was visible as a stray
`tmpXXXXXX.html` next to the output. After implementation the same selector
reports `15 passed`; adding `tests/test_report/test_model.py` and the two
enhanced tear-sheet lifecycle tests brings the Task 8 test files to
`22 passed`.

Gates after implementation:

```text
tests/test_report tests/test_pyfolio tests/test_tearsheets tests/test_utils:  326 passed, 11 warnings (0 failures)
tests/compat/pyfolio tests/compat/empyrical tests/test_core tests/test_viz tests/contracts:  1003 passed, 4 warnings
scoped mypy --follow-imports=skip (9 changed source files):  Success
ruff check / ruff format --check on all changed files:  clean
git diff --check:  clean
git status --short -- fincore/utils/static/:  empty (no source-tree writes)
```

Coverage of changed modules (report/utils selectors): `render_pdf.py` 100%,
`render_html.py` 95%, `compute.py` 90%, `common_utils.py` 99%, `model.py` 92%
(after adding the nested-mapping/tuple classification tests), `artifacts.py`
83% (pre-existing Task 7 paths). The 11 warnings are the documented
interesting-times business warnings; the 4 warnings in the compat run are the
pinned outer-concat warnings from Task 4.

## Files changed

Production: `fincore/report/model.py` (new), `fincore/report/assets/echarts.min.js`
(new asset), `fincore/report/__init__.py`, `fincore/report/artifacts.py`,
`fincore/report/compute.py`, `fincore/report/render_html.py`,
`fincore/report/render_pdf.py`, `fincore/_pyfolio_impl.py`,
`fincore/tearsheets/sheets.py`, `fincore/utils/common_utils.py`,
`fincore/utils/__init__.py`, `pyproject.toml`, `MANIFEST.in`.
Tests: `tests/test_report/test_offline_report.py` (new),
`tests/test_report/test_pdf_cleanup.py` (new),
`tests/test_report/test_model.py` (new),
`tests/test_pyfolio/test_backend_side_effect.py` (new),
`tests/test_utils/test_export_destination.py` (new).
`fincore/pyfolio.py` needed no change: it is already a lazy manifest with no
import-time backend call, and Task 6's lazy-import guard keeps passing.

## Design decisions

- `ReportModel` subclasses `dict` so `compute_sections`' return type change is
  invisible to every existing consumer (`tests/test_report/test_compute.py`
  asserts `isinstance(sections, dict)`).
- ECharts is inlined into each HTML report rather than referenced as a sibling
  file, keeping reports single-file self-contained (and Playwright's
  `networkidle` wait deterministic); the ~1 MB read is lru-cached per process.
- `export=` and `return_result` are keyword-only, so positional callers and
  the pinned strict wrappers are unaffected; `print_table` returns `None`
  unless `export` is supplied.
- Import of `ReportArtifacts` inside `print_table` stays function-local to
  keep `import fincore.utils` light and side-effect free.

## Concerns

- `Pyfolio(returns=...)` (stateful enhanced instance) with synthetic returns
  hits a pre-existing `cum_returns` dispatch TypeError
  (`'float' object does not support item assignment`, pandas `_assign_where`),
  unrelated to Task 8; the enhanced `return_result` tests therefore use the
  stateless `Pyfolio()` form used across `tests/test_pyfolio/test_tears.py`.
  Worth a follow-up outside this task.
- Inlining ECharts makes each HTML report ~1 MB larger; acceptable for a
  self-contained offline report, but a future task could add an opt-out to
  emit a sibling asset reference.
- `ReportArtifacts.model` is typed via TYPE_CHECKING-only import; consumers
  doing runtime `isinstance(artifacts.model, ReportModel)` must import the
  model module themselves (it is exported from `fincore.report.compute` and
  `fincore.report.model`).

## Fix round 1: ownership of figures in return_result

Review finding (Important): `create_full_tear_sheet` collected
`plt.get_fignums()` at the end of the run, claiming ALL open process figures —
including caller-owned ones that existed before the run — so
`result.close()` silently closed them.

Fix: `create_full_tear_sheet` snapshots `frozenset(plt.get_fignums())` at the
start of the run (only when `return_result=True`, so the default path pays
nothing) and collects only figure numbers that appear after the run:

```python
figures_before = frozenset(plt.get_fignums()) if return_result else frozenset()
...
figures = [plt.figure(num) for num in plt.get_fignums() if num not in figures_before]
return ReportArtifacts(backend="matplotlib", figures=figures)
```

New regression test
`tests/test_pyfolio/test_backend_side_effect.py::test_return_result_does_not_close_caller_owned_figures`
creates a caller-owned figure before the run, asserts the result does not
claim it, closes the result, and asserts the caller's figure is still open
while every returned (run-created) figure is closed. It was RED against the
old implementation (the result's `figures` included the caller's figure
number) and is GREEN after the fix.

Gates after the fix:

```text
tests/test_report tests/test_pyfolio tests/test_tearsheets tests/test_utils:  327 passed, 0 failures
focused RED -> GREEN:  1 failed -> 1 passed
ruff check / ruff format --check on sheets.py + the new test:  clean
git diff --check:  clean
```
