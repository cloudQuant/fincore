# Adversarial Review Report — fincore

*Cynical review of core modules (empyrical, context, engine, drawdown, ratios)*

---

## Findings (minimum 10)

1. **Silent failure in Empyrical.__init__** — AnalysisContext creation catches only `(TypeError, ValueError, KeyError)`; ImportError, AttributeError, or OSError would either propagate or be masked by a broader handler. Users get no feedback when context creation fails for unexpected reasons.

2. **None propagation in @_dual_method wrappers** — When `_get_returns(returns)` or `_get_factor_returns(factor_returns)` returns None, it is passed directly to downstream metric functions (e.g. drawdown, ratios). Many metrics do not document or handle None, leading to cryptic TypeErrors.

3. **Inconsistent inf handling in regression_annual_return** — The function checks `np.isnan(alpha_val)` and `np.isnan(beta_val)` but not `np.isinf`. If upstream alpha/beta computation yields inf, the arithmetic propagates inf rather than returning np.nan.

4. **AnalysisContext.__repr__ IndexError on empty returns** — When `len(self._returns) == 0`, `self._returns.index[0]` and `self._returns.index[-1]` raise IndexError. The repr path lacks a guard for empty Series.

5. **RollingEngine.compute accepts empty metrics list** — Passing `metrics=[]` produces an empty dict and no error. Callers may assume non-empty output; API should reject or document this explicitly.

6. **Division-by-zero risk in RollingEngine._compute_sortino** — When all returns are non-negative, `downside` is all zeros; `rolling_downside_std` can be zero, yielding inf from division. `np.errstate` suppresses the warning but does not replace inf with nan.

7. **drawdown.max_drawdown potential zero-division** — When cumulative returns hit exactly zero (e.g. total loss), `(cumulative - max_return) / max_return` can divide by zero. The nanmin call does not guard against invalid intermediates.

8. **get_max_drawdown_underwater fragile peak detection** — Reliance on `underwater[:valley][underwater[:valley]==0].index[-1]` with a bare IndexError fallback is brittle; edge cases (e.g. all-negative underwater, duplicate indices) may produce surprising peaks.

9. **AnalysisContext.to_html path handling** — `open(path, "w")` is called without checking path validity. Empty string, path in non-existent directory, or permission issues cause ungraceful failures.

10. **Missing MODULE_PATHS alias validation** — `_resolve_module` assumes `MODULE_PATHS[alias]` exists. A registry typo (e.g. `"_drawdwn"`) raises KeyError with no clear message, making debugging harder.

11. **Lazy method resolution has no fallback for missing functions** — `getattr(_resolve_module(...), func_name)` raises AttributeError if the function was removed or renamed. No user-friendly error or validation at startup.

12. **futures_market_correlation and similar pass futures_returns without null check** — Empyrical.futures_market_correlation passes `futures_returns` directly to the stats module. If None is passed at instance level with no fallback, downstream behavior is undefined.

---

*Review conducted per bmad-review-adversarial-general. Recommendations should be triaged by severity and effort.*
