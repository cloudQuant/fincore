# Task 12B Report — close metric kernel boundaries

Commit: `1ba421c` `type: close metric kernel boundaries` (staged only `fincore/metrics/*` (15 files) + the one-line `pyproject.toml` override removal).

## Error inventory (before → after)

Full package:

- Before this batch: **158 errors / 29 files**.
- After: **53 errors / 14 files** (net −105). All remaining errors live in files owned by later batches:
  - 12C: `fincore/pyfolio.py` — 1
  - 12D: `fincore/viz/` — 11 (plotly_backend 3, bokeh_backend 3, html_backend 3, matplotlib_backend 2)
  - 12E: `fincore/risk/evt.py` — 20, `fincore/attribution/` — 13 (fama_french 7, brinson 4, style 2), `fincore/optimization/` — 5 (frontier 3, objectives 2), `fincore/tearsheets/` — 2 (risk 1, capacity 1), `fincore/simulation/bootstrap.py` — 1

Scoped run (`mypy fincore/metrics --ignore-missing-imports`):

- Baseline with the lax override removed: **156 owned errors / 15 files** (ratios 25, alpha_beta 21, drawdown 16, consecutive 16, yearly 15, risk 12, perf_attrib 12, stats 10, returns 7, rolling 6, timing 4, bayesian 4, positions 3, basic 3, transactions 2). Removing the override surfaced ~51 `no-any-return` errors on top of the ~105 lax-mode errors because global `warn_return_any = true` now applies to metrics.
- After fixes: **0 owned errors**. The literal command still prints 17 errors in 7 files (optimization 5, viz 11, pyfolio 1) because mypy follows imports; those belong to 12C/12D/12E.

## Changes made (all runtime-neutral)

1. **Align-boundary narrowing** — `align_binary_metric_inputs` (12A-owned) returns `Series | DataFrame | ndarray`; every metrics call site now unpacks into fresh names and `cast`s back to the declared input kinds, so pandas containers flow through to `*_aligned` kernels and `treynor_mazuy_timing` unchanged (a compat test pins that Series containers and normalized labels reach the timing kernel — an initial `np.asanyarray` conversion broke it and was reverted in favor of casts).
2. **`out`-buffer tail restructures** — `sortino_ratio`, `downside_risk`, `max_drawdown`, `alpha_aligned`, `cal_treynor_ratio`, `cum_returns` tails split into early returns (`out.item()` cast to the declared union; `pd.Series(out)` / `pd.DataFrame(out)` branches return directly). Mutually exclusive branches, identical execution order.
3. **Scalar pinning** — `float()`/`cast()` around `np.mean/std/nansum/percentile`, scipy calls (`skew`, `kurtosis`, `linregress`), `Series.max/min/idxmax/idxmin`, `ndarray[...]` indexing, `Index[0]` access, and `float ** float` (mypy types it as `Any`). Values are numerically identical (np.float64 → float).
4. **Pandas-stub blockers, precise only** — three per-line `# type: ignore[misc]` with comments for `.loc[label_slice:]` (stubs restrict slices to integers; all in drawdown.py); `DatetimeIndex` casts where the container contract guarantees it (`ensure_datetime_index_series`, transaction frames); `axis="rows"` → identical `axis="index"` alias (positions, perf_attrib); `Series.values` → `to_numpy()`/`np.asarray()` where stubs widen the union (same underlying values for the float64 series produced here); `Resampler.apply` gets a `_resample_apply_final` Series-callable adapter; `rolling_beta`'s `DataFrame.apply(partial(...))` becomes an explicit `column_beta` wrapper.
5. **Private helpers annotated** instead of global ignores: `_safe_correlation`, `_market_correlation`, `_compute_annualized_return`, `_capture_aligned`, `_get_annual_return`, plus widened params of private `_conditional_alpha_beta` (public signatures untouched).
6. `pyproject.toml` — removed `"fincore.metrics.*"` from the permissive override (`fincore.plugin.*`, `fincore.data.*` left for later batches).

No new global ignores. The three new per-line ignores are `[misc]` on pandas-stub label slices (all in drawdown.py).

## Gate results

1. **Scoped mypy**: 0 owned errors (literal command still non-zero via import-following, see concerns).
2. **Full-package mypy**: 53 ≤ 158 baseline. OK.
3. **Regression**: `pytest -o addopts='' tests/test_metrics tests/compat -q --tb=short --maxfail=0` → **1123 passed, 1 failed**. The single failure (`test_full_generator_is_byte_idempotent_when_pinned_roots_are_available`) was verified pre-existing by running it on the pristine tree (`git stash`); the 2nd failure seen mid-batch (timing strict-policy test) was introduced by my np.asanyarray conversion and fixed by reverting to a cast — the selector now matches its pre-batch profile.
4. **Lint**: `ruff check` clean (TC006 auto-fixed to quoted cast types), `ruff format --check` clean, `git diff --check` clean, `compileall` clean.

## Concerns

- The scoped acceptance command remains literally non-zero until 12C–12E merge (import-following reports their files' errors). The owned-file count is 0; the per-file remainder inventory above is the handoff.
- `py.typed` retained; still valid only when full-package mypy reaches 0 (post-12E).

## Fix round 1 (review)

Review finding (Important): the ~20 `float(...)` wraps added to pin scalar returns changed the runtime scalar container from `np.float64` to Python `float` — numerically identical but a runtime change, which the batch rules forbid in type commits.

- Mechanically replaced every added `float(...)` wrap with `cast("float", ...)` (identity at runtime, same mypy effect): ratios.py (conditional_sharpe mean/std/sqrt, mar_ratio mean, omega numer/denom, kappa lpm3/mu/cuberoot), risk.py (cvar, tail_ratio, residual_risk, trading_value_at_risk), rolling.py (2 `sqrt_ann` + rolling_sharpe), stats.py (stutzer ip/return, r_cubed_turtle avg_max_dd), yearly.py (annual_return scalar path). Two comments referencing "float() is a no-op" reworded to cast().
- Minor 1: corrected the report's per-line-ignore count — drawdown.py has **three** `# type: ignore[misc]` (get_max_drawdown_period:437, max_drawdown_days:469, max_drawdown_recovery_days:552), not two.
- Minor 2: softened the stats.py `normalize` cast comment (no "historical public API" overstatement).
- Gates re-run after the fix: scoped mypy owned files **0**; full-package **53**; regression selector `tests/test_metrics tests/compat` → **1123 passed, 1 failed** (the pre-existing manifest byte-idempotence failure; identical to the pre-batch profile); `ruff check` / `ruff format --check` / `git diff --check` clean.

Follow-up commit: `fix: restore exact runtime containers at metric boundaries`.
