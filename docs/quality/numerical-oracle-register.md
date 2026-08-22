# Numerical Oracle Register

Iteration 0042 (Task 0) — P0 numerical-correctness fixes and their independent
oracles. Each entry records the formula, source, units, boundary behavior,
oracle generation method, and the tolerance asserted by
`tests/numerical/test_*_reference_oracles.py`.

## Principles

1. Every P0 domain function has at least one **independent oracle** that never
   imports `fincore` (plain NumPy/SciPy/statsmodels references).
2. Fixtures are reproducible (fixed seed) and documented; no mystery numbers.
3. A "wrong-model" counter-example is present where the error would flip a
   financial conclusion (e.g. a negative LR statistic accepting a broken VaR).

## Risk

### 1. Kupiec LR-POF

- **File:** `fincore/risk/backtesting.py::kupiec_lr`
- **Formula:** `LR = 2 * [ x·ln(x/(n·p)) + (n−x)·ln((n−x)/(n·(1−p))) ]`
  where `n` = observations, `x` = exceptions, `p = 1 − confidence_level`.
- **Source:** Kupiec (1995), *Journal of Derivatives* 3(2), 73–84.
- **Boundary:** `x = 0` → `−2n·ln(1−p)`; `x = n` → `2n·ln(1/p)`. Implemented via
  `scipy.special.xlogy` (continuous limit `0·log 0 = 0`), so LR ≥ 0 always.
- **Oracle:** `tests/oracles/risk/kupiec_oracle.py` — xlogy form plus a
  brute-force `math.log` form with explicit boundary limits (two independent
  code paths).
- **Tolerance:** `rtol = atol = 1e-12`; `kupiec_lr(100, 5, 0.99) ≈ 8.258217002871657`.
- **Wrong-model example:** a 99% VaR with 5 exceptions in 100 obs gives
  `LR > 3.84` (χ²(1) 95% critical value), rejecting the null.

### 2. VaR / Expected Shortfall forecast pair

- **File:** `fincore/risk/models.py::forecast_var` / `forecast_es`
- **Normal ES (losses-negative):** `ES = −σ·φ(z_α)/α`, `VaR = σ·z_α`, with
  `z_α = Φ⁻¹(α)`, `α = 1 − confidence_level`.
- **Student-t ES:** `ES = −σ·f(t)/α · (ν + t²)/(ν − 1)`, `t = F_ν⁻¹(α)`.
- **Source:** McNeil, Frey & Embrechts (2015), *Quantitative Risk Management*.
- **Boundary:** ES ≤ VaR ≤ 0 under losses-negative; GARCH ES must not equal GARCH VaR.
- **Oracle:** `tests/oracles/risk/normal_es_oracle.py` (closed-form normal/t).
- **Tolerance:** analytic `rtol ≤ 1e-8`; monotonicity `ES ≤ VaR`.
- **Horizon:** multi-horizon VaR/ES aggregates forecast variances
  (square-root-of-sum), so horizon 1/5/10/20 differ measurably.

### 3. GARCH-family model identity and forecast

- **File:** `fincore/risk/garch.py`
- **Recursions:** GARCH(1,1) `s²_t = ω + α·ε²_{t−1} + β·s²_{t−1}`;
  GJR(1,1) `s²_t = ω + (α + γ·I(ε<0))·ε²_{t−1} + β·s²_{t−1}`;
  EGARCH(1,1) `log s²_t = ω + α·|z| + γ·z + β·log s²_{t−1}`, where
  `z = ε_{t−1}/sqrt(s²_{t−1})` is the preceding conditional standardized
  innovation used by both fitting and forecasting.
- **Forecast:** one-step then persistence recursion; GJR persistence
  `α + 0.5γ + β`; EGARCH decays toward `ω/(1−β)` with drift `α·√(2/π)`.
- **Source:** Engle (1982); Bollerslev (1986); Nelson (1991); Glosten,
  Jagannathan & Runkle (1993).
- **Boundary:** only `(p, q) = (1, 1)` is exposed; other orders raise
  `ValueError`. A fit is marked `converged=False` unless the optimizer succeeds
  and GARCH/GJR persistence is strictly below one (`α+β` or `α+0.5γ+β`), or
  EGARCH has `|β|<1`; the enhanced adapter then emits `status="failed"` rather
  than a seemingly valid forecast. Overflowing EGARCH optimizer probes receive
  a finite likelihood penalty. EGARCH rejects non-finite, zero-variance and
  overflow-variance return inputs before optimization.
- **Oracle:** `tests/oracles/risk/garch_oracle.py` (independent recursions).
- **Tolerance:** recursion `rtol ≤ 1e-12`; `arch` package is not importable in
  this environment, so convergence is validated by seeded fixtures rather than
  a second optimizer.

### 4. EVT tail selection, GPD PWM / threshold domain, GEV ES and Hill tail index

- **File:** `fincore/risk/evt.py`
- **Fix:** `gpd_fit`/`gev_fit`/`evt_var`/`evt_cvar` honor the `tail` argument
  (lower → negated losses, upper → gains); `gev_fit` converts SciPy's
  `genextreme` shape `c` to the standard GEV shape `xi = −c` (SciPy `c > 0` is
  bounded Weibull, standard `xi > 0` is heavy Fréchet). `hill_estimator` uses
  the threshold Hill form `xi_hat = (1/k) Σ log(x_i / u)` for positive tail
  magnitudes `x_i > u`; it does not take logarithms of excesses `x_i − u`.
  GPD PWM now derives the first two sample L-moments (`l1=b0`,
  `l2=2*b1-b0`) and uses `xi=2-l1/l2`, `beta=l1*(1-xi)`, rejecting an
  invalid non-positive scale. GPD VaR/ES never applies a conditional-excess
  model below its fitted threshold: an explicit threshold requires
  `alpha <= n_exceed/n_total`; an automatic VaR/ES threshold retains the
  90th tail percentile when it covers `alpha` and otherwise lowers only far
  enough to retain the requested empirical tail mass. GEV ES is the
  conditional tail mean `alpha^-1 integral_(1-alpha)^1 Q(p) dp`, evaluated by
  the lower-incomplete-gamma expression (or its Gumbel/E1 limit), rather than
  a fixed additive offset from VaR. The GEV quantile uses `log1p(-alpha)` and
  the Gumbel small-tail branch divides its E1-series terms before summing, so
  representable subnormal `alpha` values do not round away or cancel to zero.
- **Source:** Embrechts, Klüppelberg & Mikosch (1997); McNeil, Frey &
  Embrechts (2015).
- **Boundary:** lower-tail VaR/ES negative, upper-tail positive; standard
  `xi > 0` for heavy-tailed (Student-t) data. `alpha` is finite and in
  `(0,1)`; GPD requires the target tail probability to be covered by the POT
  threshold; GEV ES is finite only for `xi < 1`. Hill thresholds must be
  finite and positive; lower-tail reflection preserves the tail index.
- **Oracle:** `scipy.stats.genextreme.fit` sign convention verified against
  Student-t(3) block maxima (standard `xi ≈ +0.33`), a standalone NumPy
  threshold-Hill/L-moment PWM reference, and independent SciPy-PDF quadrature
  for GEV ES in `tests/oracles/risk/evt_oracle.py`.
- **Tolerance:** tail identity `upper ≠ lower` on skewed data; `xi > 0` for
  heavy tails; threshold-Hill/PWM formula `rtol = atol = 1e-12`; GEV ES PDF
  quadrature `rtol = atol = 1e-10`.

### 5. Deflated Sharpe Ratio

- **File:** `fincore/metrics/ratios.py::deflated_sharpe_ratio`
- **Fix:** use **ordinary** kurtosis (`excess + 3`, normal = 3) in
  `V[SR] = 1 − γ₃·SR + (γ₄ − 1)/4·SR²`, and the exact expected-max-Sharpe
  hurdle `SR* = √V[SR]·[(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]` with the
  Euler–Mascheroni constant `γ ≈ 0.5772` (previously an `√(2 ln N)`
  approximation).
- **Source:** Bailey & López de Prado (2014), *Journal of Portfolio
  Management* 40(5), 94–107.
- **Oracle:** independent reference using `scipy.stats.skew/kurtosis`.
- **Tolerance:** `rtol = atol = 1e-10`.

## Simulation

### 6. Geometric Brownian Motion

- **File:** `fincore/simulation/paths.py`, `fincore/simulation/monte_carlo.py`
- **Fix:** remove double annualization/de-annualization — `gbm_from_returns`,
  `MonteCarlo.simulate`, `price_paths` and `from_parameters` now pass annualized
  `mu`/`sigma` directly to `geometric_brownian_motion`, which applies the single
  `dt` scaling. `MonteCarlo.simulate` honors `drift`/`volatility` args, and
  `antithetic=True` pairs `Z` with `−Z` from the same stream.
- **Formula:** `log(S_T/S0) ~ N((μ − ½σ²)T, σ²T)`.
- **Source:** standard GBM; Black–Scholes/Merton.
- **Boundary:** 20% annual vol over one year → ~20% terminal log-return std.
- **Oracle:** `tests/oracles/simulation/gbm_oracle.py` (analytic moments + 99%
  Monte Carlo confidence intervals).
- **Tolerance:** terminal log-vol relative error ≤ 1%; mean/std inside analytic CI.

## Attribution

### 7. Fama-French OLS/WLS + Newey-West HAC

- **File:** `fincore/attribution/fama_french.py::FamaFrenchModel.fit`
- **Fix:** true WLS (weighted least squares via `√w` scaling, `weights=` arg)
  and the Newey-West sandwich covariance `(X'X)⁻¹ S (X'X)⁻¹` with Bartlett
  kernel weights `w_j = 1 − j/(L+1)`, replacing a scalar residual-autocorrelation
  "adjustment factor" that produced identical standard errors for all coefficients.
- **Source:** Newey & West (1987), *Econometrica* 55(3), 703–708.
- **Oracle:** `statsmodels` `OLS(cov_type="HAC")` and `WLS` (0.14.6).
- **Tolerance:** HAC `rtol = atol = 1e-10`; WLS coefficients `rtol = atol = 1e-8`.

### 8. Multi-period Brinson linking

- **File:** `fincore/attribution/brinson.py::brinson_cumulative`
- **Fix:** Carino geometric linking replaces arithmetic summation. Carino
  constant `k_t = [ln(1+rᵖ_t) − ln(1+rᵇ_t)] / (rᵖ_t − rᵇ_t)` (or `1/(1+rᵖ_t)`
  when *exactly* equal) is evaluated as
  `log1p((rᵖ_t−rᵇ_t)/(1+rᵇ_t)) / (rᵖ_t−rᵇ_t)` to avoid subtracting nearly
  equal logs.  Component returns and weights must be finite; the Carino domain
  guard applies to the aggregate portfolio and benchmark period/cumulative
  returns, which must be finite and greater than `−1`. For compounded
  portfolio and benchmark returns `R_p` and
  `R_b`, `K = [ln(1+R_p) − ln(1+R_b)] / (R_p−R_b)` (with its equal-return
  limit); each linked effect is `E_cum = Σ (k_t/K)·E_t`. The returned
  allocation, selection, and interaction therefore sum directly to the
  standard BHB cumulative active return `R_p−R_b`.
- **Source:** Carino (1999), *Journal of Performance Measurement* 3(4), 5–14.
- **Oracle:** standalone NumPy/math BHB + Carino reference in
  `tests/oracles/attribution/brinson_oracle.py`, plus an 80-digit Decimal
  reference for the two-period near-total-loss fixture; it does not import
  `fincore` or derive effects through production code.
- **Tolerance:** reconciliation residual ≤ 1e-12.

### 9. Style beta and momentum

- **File:** `fincore/attribution/style.py`
- **Fix:** regression-attribution beta uses the slope `cov/var` instead of the
  correlation coefficient; `_calculate_momentum` uses geometric cumulative
  returns `(1+r).cumprod()` and a trailing-window relative change instead of
  the constant-zero `shift/shift − 1` expression.
- **Source:** standard OLS beta; momentum definition.
- **Oracle:** `np.cov/np.var` direct computation.
- **Tolerance:** beta `rtol = atol = 1e-6`.

## Factor research

### 10. Fama-MacBeth cross-sectional label alignment

- **File:** `fincore/factor_analysis/inference.py::fama_macbeth`
- **Fix:** a one-row exposure panel is a static cross-section and is broadcast
  over each return date; a time-varying panel is aligned by asset label before
  every cross-sectional fit. Missing exposure dates are skipped rather than
  matched by row position. Duplicate labels are rejected, preventing a
  silently inverted coefficient when callers reorder assets.
- **Method:** each usable date fits `R_i = alpha + beta X_i` by least squares;
  the reported coefficient is the time-series mean. The default `"iid"`
  standard error is `ddof=1 / sqrt(n)`. The explicit
  `covariance="newey-west"` profile applies the uncorrected Bartlett HAC
  covariance to each chronological sequence of fitted intercepts/slopes:
  `Var(mean) = [S_0 + Σ_(j=1)^L (1-j/(L+1)) (S_j+S'_j)] / n²`.
- **Boundary:** Newey-West requires a chronological returns index and
  `0 <= L < n_fitted_cross_sections`; retained fitted cross-sections preserve
  chronological order, while skipped dates are not silently re-dated. The result attrs record
  the covariance profile, lag count and fitted count. It is not a clustered or
  multi-factor cross-sectional covariance claim.
- **Oracle:** `statsmodels.api.OLS` independently fits each cross-section in
  `tests/numerical/test_factor_inference.py`; its coefficient means and
  i.i.d. standard errors are compared without importing the production
  routine. `tests/oracles/factor/newey_west_oracle.py` separately compares the
  deterministic serial-coefficient fixture to
  `OLS(coefficients, ones).fit(cov_type="HAC")`.
- **Tolerance:** `rtol = atol = 1e-12` for the deterministic fixtures. Static
  exposure, shuffled asset-label, non-chronological and invalid-lag inputs are
  separate adversarial fixtures.

### 11. Benjamini-Hochberg false-discovery-rate correction

- **File:** `fincore/factor_analysis/inference.py::benjamini_hochberg`
- **Method:** sort the `m` p-values increasingly, reject through the largest
  rank `k` satisfying `p_(k) <= alpha*k/m`, and compute monotone adjusted
  values with the reverse cumulative minimum of `m*p_(k)/k`.
- **Source:** Benjamini & Hochberg (1995), *Journal of the Royal Statistical
  Society, Series B* 57(1), 289–300.
- **Boundary:** `alpha` is finite and in `(0, 1]`; p-values are finite and in
  `[0, 1]`. A labelled Series must have unique hypothesis labels, and an empty
  input returns an explicit empty audit result rather than an implicit
  no-discovery claim.
- **Oracle:** `statsmodels.stats.multitest.multipletests(method="fdr_bh")`
  independently supplies the rejection decisions and adjusted p-values for
  shuffled, tied factor hypotheses.
- **Tolerance:** exact boolean decisions; adjusted values `rtol = atol =
  1e-12`.

### 12. IC t-statistic and confidence-interval boundaries

- **File:** `fincore/factor_analysis/inference.py::ic_mean`,
  `ic_t_stat`, and `ic_confidence_interval`
- **Method:** missing (`NaN`) IC observations are omitted; the t-statistic uses
  the sample standard error `s / sqrt(n)` with `ddof=1`, and the interval is
  `mean ± z*SE` under the explicitly documented i.i.d. assumption.
- **Boundary:** infinite observations are rejected instead of producing a
  warning/undefined statistic. Fewer than two usable observations return
  `NaN`; a zero-mean, zero-variance sample has t-statistic zero, while a
  nonzero constant sample has the corresponding signed infinity. The interval
  multiplier must be finite and strictly positive.
- **Oracle:** `scipy.stats.ttest_1samp` supplies the non-degenerate t-statistic
  reference; zero-variance limits and invalid-input behavior use explicit
  adversarial fixtures.
- **Tolerance:** non-degenerate t-statistic `rtol = atol = 1e-12`; boundary
  outputs exact.

### 13. Causal PIT factor materialization

- **File:** `fincore/factor_analysis/pit.py::materialize_pit_factor` and
  `fincore/factor_analysis/data.py::prepare_pit_factor_data`
- **Method:** an enhanced factor ledger contains `asset`, `as_of`, `known_at`,
  `effective_from`, `value`, and `in_universe`. For each requested evaluation
  date, select only revisions satisfying
  `as_of <= known_at <= effective_from <= evaluation_date`; choose the latest
  eligible revision per asset by `(effective_from, known_at, as_of)`. A latest
  `in_universe=False` record removes the asset. The PIT preparation wrapper
  rejects the legacy full-sample `filter_zscore` option before computing
  forward returns.
- **Boundary:** timestamps must use one timezone (or all be naive), evaluation
  dates are sorted and unique, values are finite, revision tuples are unique,
  and causal ordering is enforced. Invalid data fails closed instead of being
  re-timestamped or silently carried forward.
- **Oracle:** hand-specified event-time timeline plus an adversarial future
  perturbation fixture in `tests/numerical/test_factor_pit_materialization.py`.
  This is a deterministic selection contract, not a fitted numerical model:
  the expected `(date, asset)` series is written independently of production
  helper calls, and an observation known after the tested horizon must leave
  that horizon unchanged.
- **Tolerance:** exact labels and values; no tolerance-based causal exception.
- **Scope:** this is an additive enhanced input path. It does not yet prove
  versioned corporate actions/calendars, research-trial tracking, integration
  with every factor report, or liquidity/borrow provenance and execution
  calibration for the separate cost ledger.

### 14. Enhanced factor-model IC/FDR post-analysis

- **File:** `fincore/factor_analysis/inference.py::factor_model_inference` and
  `information_coefficient_inference`
- **Method:** consume the immutable model's aggregate date-by-period IC table.
  For every forward period with at least two finite observations, compute the
  i.i.d. two-sided Student-t p-value for mean IC equal to zero, then apply BH
  jointly to those testable periods. Periods with fewer than two observations
  retain their count and mean but have `testable=False`, `NaN` p/q values, and
  no rejection.
- **Oracle:** `scipy.stats.ttest_1samp` provides independent p-values and
  `statsmodels.stats.multitest.multipletests(method="fdr_bh")` provides BH
  decisions/q-values in `tests/numerical/test_factor_model_inference.py`.
  The model-wrapper test verifies that the enhanced workflow consumes the
  stored aggregate IC snapshot rather than recomputing an alternate series.
- **Tolerance:** Student-t p-values and adjusted values `rtol = atol =
  1e-12`; exact testability and rejection flags.
- **Scope:** intentionally i.i.d. only. It is not a HAC/cluster claim and does
  not provide a pre-registered factor family, trial registry, integration with
  the separate cost/capacity ledger, or report-level disclosure workflow.

### 15. Per-horizon enhanced factor-data availability

- **File:** `fincore/factor_analysis/data.py::prepare_factor_data_by_horizon`
- **Method:** calculate factor bins from the finite factor/universe panel once,
  then create a separate table for every unique computed forward-return label.
  Each table admits only finite returns for its own horizon and carries a
  standalone `FactorLossReport`; it must not inherit availability loss from a
  different horizon.
- **Boundary:** duplicate computed horizon labels fail closed; full-sample
  `filter_zscore` is rejected; every horizon enforces `max_loss` independently.
  The strict Alphalens-compatible all-column cleaner is deliberately not
  modified.
- **Oracle:** a hand-specified three-date/four-asset price panel in
  `tests/numerical/test_factor_multihorizon_preparation.py` has twelve valid
  `1D` rows and eight valid `3D` rows. Mutating only the terminal price changes
  the eligible `3D` table but must leave the complete `1D` table byte-for-byte
  unchanged.
- **Tolerance:** row counts, labels, loss ratios, and short-horizon frame are
  exact; no floating tolerance is used for availability decisions.
- **Scope:** this establishes data-availability isolation, not corporate-action
  or calendar provenance, trial tracking, or integration with the separate
  cost/borrow/slippage/capacity ledger.

### 16. Enhanced factor cost, borrow and capacity accounting

- **File:** `fincore/factor_analysis/costs.py::apply_factor_costs` and
  `estimate_factor_capacity`
- **Method:** for gross-normalized factor holdings `w[t,i]`, begin with zero
  holdings and calculate one-way trade weight `q[t,i] = |w[t,i]-w[t-1,i]|`.
  The ledger reports turnover `0.5 * sum(q)`, spread
  `sum(q) * half_spread_bps / 10_000`, temporary impact
  `sum(q * coefficient * participation**exponent)`, borrow
  `sum(max(-w,0) * per_period_borrow_rate)`, and
  `net = gross - spread - impact - borrow`. Participation is
  `q * portfolio_value / dollar_volume`; the hard capacity is the minimum
  nonzero-trade inequality `max_participation * dollar_volume / q`.
- **Boundary:** sparse `(date, asset)` positions are explicit zero holdings so
  entry/exit trades cannot become `NaN`. Gross normalization, complete labels,
  positive dollar volume, finite numeric values, and same-currency portfolio
  value/volume are mandatory. Any short requires both finite per-period borrow
  rates and boolean availability; unavailable/missing borrow and an
  over-capacity portfolio raise instead of emitting a net return.
- **Oracle:** `tests/oracles/factor/costs_oracle.py` is a NumPy-only ledger;
  it never imports `fincore`. The labelled fixture in
  `tests/numerical/test_factor_costs.py` compares every ledger component and
  the binding capacity to that oracle, then separately adversarially tests
  label reordering, unavailable borrow, capacity rejection, sparse exits, and
  defensive output snapshots.
- **Tolerance:** arithmetic ledger and capacity use `rtol = atol = 1e-12` in
  the reference comparison; availability/capacity rejection is exact.
- **Scope:** an explicit research accounting convention, not calibrated market
  impact, FX conversion, an order-level execution simulator, or a replacement
  for retained liquidity/borrow source provenance.

## Regeneration

Re-run the numerical gate with:

```bash
python -m pytest -o addopts='' tests/numerical tests/oracles tests/test_risk \
  tests/test_simulation tests/test_attribution tests/property/test_risk_invariants.py \
  -q --tb=short --maxfail=0
```
