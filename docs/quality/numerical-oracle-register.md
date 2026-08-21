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
  EGARCH(1,1) `log s²_t = ω + α·|z| + γ·z + β·log s²_{t−1}`.
- **Forecast:** one-step then persistence recursion; GJR persistence
  `α + 0.5γ + β`; EGARCH decays toward `ω/(1−β)` with drift `α·√(2/π)`.
- **Source:** Engle (1982); Bollerslev (1986); Nelson (1991); Glosten,
  Jagannathan & Runkle (1993).
- **Boundary:** only `(p, q) = (1, 1)` is exposed; other orders raise
  `ValueError`. Unconverged optimizers never report `status="ok"`.
- **Oracle:** `tests/oracles/risk/garch_oracle.py` (independent recursions).
- **Tolerance:** recursion `rtol ≤ 1e-12`; `arch` package is not importable in
  this environment, so convergence is validated by seeded fixtures rather than
  a second optimizer.

### 4. EVT tail selection and GEV shape sign

- **File:** `fincore/risk/evt.py`
- **Fix:** `gpd_fit`/`gev_fit`/`evt_var`/`evt_cvar` honor the `tail` argument
  (lower → negated losses, upper → gains); `gev_fit` converts SciPy's
  `genextreme` shape `c` to the standard GEV shape `xi = −c` (SciPy `c > 0` is
  bounded Weibull, standard `xi > 0` is heavy Fréchet).
- **Source:** Embrechts, Klüppelberg & Mikosch (1997); McNeil, Frey &
  Embrechts (2015).
- **Boundary:** lower-tail VaR/ES negative, upper-tail positive; standard
  `xi > 0` for heavy-tailed (Student-t) data.
- **Oracle:** `scipy.stats.genextreme.fit` sign convention verified against
  Student-t(3) block maxima (standard `xi ≈ +0.33`).
- **Tolerance:** tail identity `upper ≠ lower` on skewed data; `xi > 0` for
  heavy tails.

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

## Regeneration

Re-run the numerical gate with:

```bash
python -m pytest -o addopts='' tests/numerical tests/oracles tests/test_risk \
  tests/test_simulation tests/test_attribution tests/property/test_risk_invariants.py \
  -q --tb=short --maxfail=0
```
