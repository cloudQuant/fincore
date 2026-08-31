# Risk validation

fincore separates risk *estimation* (EVT, GARCH) from risk *validation*
(out-of-sample backtesting). The enhanced layer in `fincore.risk.models` and
`fincore.risk.backtesting` records what was forecast, under which convention
and horizon, and whether realized outcomes pass, fail, or are statistically
inconclusive.

## Forecasting VaR and ES

`forecast_var` and `forecast_es` return an immutable `RiskEstimate`:

```python
import numpy as np
import pandas as pd

from fincore.risk.models import forecast_es, forecast_var

rng = np.random.default_rng(7)
returns = pd.Series(rng.normal(0.0, 0.02, 500))

var = forecast_var(returns, method="historical", confidence_level=0.99)
es = forecast_es(returns, method="historical", confidence_level=0.99)

print(var.estimate)   # negative under the losses_negative convention
print(var.sign_convention)
```

`method` can be `historical` (empirical quantile), `evt` (extreme-value
theory) or `garch` (conditional volatility). The underlying legacy EVT/GARCH
kernels live in `fincore.risk.evt` and `fincore.risk.garch`; their enhanced
adapters preserve the `losses_negative` sign convention.

## EVT tail-index convention

`hill_estimator` estimates the extreme-value tail index from positive tail
magnitudes. With a positive threshold `u`, it returns the threshold Hill
estimate `mean(log(x / u))` over observations `x > u`, together with those
selected magnitudes. Lower return tails are reflected into positive loss
magnitudes first. It is a legacy estimator rather than an out-of-sample
validated risk model, so use its threshold and tail choice as explicit model
assumptions.

## EVT threshold and Expected Shortfall semantics

For a GPD peaks-over-threshold (POT) estimate, `alpha` is an unconditional
return-tail probability. An explicit `threshold` is accepted only when the
fitted exceedance fraction covers it: `alpha <= n_exceed / n_total`. Otherwise
the body of the return distribution is not modelled by the conditional GPD and
the function raises `ValueError` rather than silently extrapolating below the
threshold. When `evt_var` or `evt_cvar` receives no threshold, it keeps the
usual 90th percentile of the selected tail if that covers `alpha`; otherwise
it selects the highest empirical threshold that still does. `gpd_fit` on its own
continues to use the 90th tail percentile because it fits parameters rather
than answering a particular VaR/ES query.

GEV estimates are for the selected **block-extreme** distribution, so their
`alpha` is a block-tail probability, not automatically a daily probability.
GEV Expected Shortfall is the conditional tail mean beyond GEV VaR and is
defined only for `xi < 1`; it is not an arbitrary constant increment from
VaR. These legacy estimators are still not out-of-sample validated models.

## Backtesting VaR

```python
# -- minimal-backtest
from fincore.risk.backtesting import backtest_var

forecast = pd.Series([-0.02, -0.02, -0.02], index=pd.date_range("2024-01-01", periods=3, tz="UTC"))
realized = pd.Series([-0.01, -0.03, -0.02], index=forecast.index)

result = backtest_var(forecast, realized, confidence_level=0.99)

assert result.observations == 3
assert result.exceptions == 1
# -- minimal-backtest
```

`backtest_var` reports the exception count plus two standard statistics:

- **Unconditional coverage** (Kupiec): does the exception rate match the
  chosen confidence level?
- **Independence** (Christoffersen): are exceptions clustered?

Both are likelihood-ratio tests with an explicit null hypothesis. When the
sample is too small to be meaningful (fewer than 3 observations, or fewer than
5 expected exceptions), the result is `inconclusive` rather than a silent pass.

## Auditable walk-forward VaR (experimental)

For the enhanced walk-forward boundary, use `RiskModelSpec` and
`walk_forward_var`. Each forecast uses only data strictly before its timestamp.
`walk_forward_var` returns a `WalkForwardVaRResult`; pass that result to
`build_risk_validation_report` to write every forecast, realised return,
exception, refit parameters, timestamp index name/timezone, and both
input/backtest digests to a deterministic JSON artifact. When a VaR backtest
is available, the artifact also contains a traffic-light zone together with
the observations and confidence level used to derive that reference field.
Timezone metadata is emitted only as a portable IANA name or fixed UTC-offset
token; a timezone that cannot be represented and replayed that way is rejected
when the report is built. Timestamp index names must likewise be native JSON
scalars so the backtest digest can be replayed exactly.

```python
import numpy as np
import pandas as pd

from fincore.risk.diagnostics import walk_forward_var
from fincore.risk.report import build_risk_validation_report
from fincore.risk.specs import RiskModelSpec

returns = pd.Series(
    np.linspace(-0.02, 0.02, 60),
    index=pd.date_range("2024-01-02", periods=60, freq="B", tz="UTC"),
)
spec = RiskModelSpec(confidence_level=0.95, distribution="normal", window=40, refit_cadence=5)
walk_forward = walk_forward_var(returns, spec)
audit_report = build_risk_validation_report(walk_forward)
audit_report.write_json("risk-validation.json")
```

This surface is **experimental**. It currently validates one-step lower-tail
VaR with Normal or finite-sample calibrated historical forecasts; it does not
turn legacy EVT/GARCH estimates into an out-of-sample validated model. Its
Basel traffic-light and backtest fields are reference aids, not regulatory
approval or a compliance certification.

## Backtesting ES (experimental)

Expected Shortfall backtesting is an open problem. The first fincore
implementation uses a bootstrap calibration score (mean realised shortfall in
the exception tail versus the forecast ES) and reports status `experimental`;
it is not a compliance statement.

```python
from fincore.risk.backtesting import backtest_es

result = backtest_es(forecast, realized, confidence_level=0.975)
print(result.status)   # "experimental"
```

## Sign convention

All enhanced risk results use the `losses_negative` convention: a VaR/ES
estimate is a negative number, and an exception occurs when the realized return
falls strictly below the forecast threshold.
