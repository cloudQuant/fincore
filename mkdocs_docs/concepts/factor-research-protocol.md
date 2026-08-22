# Factor Research Protocol

## Scope and API boundary

`fincore.factor_analysis` offers an enhanced point-in-time (PIT) input route
for new research. It is additive: `fincore.alphalens` remains the strict,
source-shaped compatibility facade and is not silently reinterpreted as a PIT
workflow.

The current PIT route establishes causal factor materialization and causal
factor-data preparation. It does not by itself certify a research result,
replace a versioned corporate-action source, or record every research trial.
The separate cost/capacity ledger below provides explicit arithmetic, but its
liquidity and borrow provenance and any execution calibration remain caller
responsibilities.

## Event-time ledger

Call `materialize_pit_factor(observations, evaluation_dates)` with one or more
revisions per asset. The ledger has these required columns:

| Column | Meaning | Contract |
| --- | --- | --- |
| `asset` | Security identifier | Non-missing and hashable. |
| `as_of` | Time the underlying fact describes | Must not be later than `known_at`. |
| `known_at` | Time the research system learned the fact | Must not be later than `effective_from`. |
| `effective_from` | First time the factor is permitted for use | Must be on or before the evaluation timestamp to be selected. |
| `value` | Numeric factor observation | Finite only. |
| `in_universe` | Membership revision | Boolean; a selected `False` removes the asset. |

All timestamp columns must share the evaluation-date timezone (or all be
naive), and evaluation dates must be sorted and duplicate-free. The causal
ordering is:

```text
as_of <= known_at <= effective_from <= evaluation_date
```

For each `(evaluation_date, asset)`, fincore selects the latest eligible
revision ordered by `effective_from`, `known_at`, then `as_of`. It rejects
duplicate revisions with the same asset and event-time tuple, non-finite
values, time-order violations, and timezone mismatches rather than guessing.

## Prepare data without full-sample filtering

```python
from fincore.factor_analysis import prepare_pit_factor_data

prepared = prepare_pit_factor_data(
    observations,
    prices,
    evaluation_dates,
    periods=(1, 5),
    quantiles=5,
    max_loss=0.35,
)
```

`prepare_pit_factor_data` first materializes eligible values and then uses the
enhanced factor preparation kernel. It rejects `filter_zscore`; using a
full-sample forward-return distribution as a filter would permit future data
to affect historical eligibility. The returned `PreparedFactorData` still has
ordinary forward-return availability loss, so callers must inspect
`loss_report` and document the selection protocol.

## Preserve availability separately for each forward horizon

For a multi-horizon enhanced study, use
`prepare_factor_data_by_horizon(factor, prices, periods=(1, 5, 20))` rather
than treating the legacy all-column cleanup result as a single research panel.
It returns `MultiHorizonPreparedFactorData.by_horizon`, an immutable mapping
from each computed forward-return label (for example, `"1D"`) to its own
`PreparedFactorData`.

Each period's `loss_report` counts forward-return availability and final
binning only for that period. A missing 20-day outcome must not remove an
otherwise usable 1-day observation, and a price change that affects only a
later long horizon cannot rebucket an already available short-horizon row.
The API therefore computes factor bins from the finite factor/universe panel
before applying each horizon's return-availability mask.

This is an enhanced-only API. The strict `fincore.alphalens` route deliberately
retains its source-shaped all-horizon cleaning semantics. Horizon labels must
be unique, every horizon separately enforces `max_loss`, and full-sample
`filter_zscore` is rejected to keep this route causal. The API does not yet
provide corporate-action/calendar provenance or a complete research-trial
workflow.

## Explicit factor cost, borrow, slippage and capacity ledger

Use `apply_factor_costs` after constructing the enhanced factor portfolio.
It is deliberately a separate, labelled accounting step: the strict
`fincore.alphalens` facade is not changed, and the API does not silently choose
an execution, liquidity, FX, or borrow policy for the caller.

```python
import pandas as pd

from fincore.factor_analysis import FactorCostModel, apply_factor_costs

dates = pd.date_range("2024-01-02", periods=2, freq="B", tz="UTC", name="date")
weights = pd.Series(
    [0.60, -0.40, 0.20, -0.80],
    index=pd.MultiIndex.from_product((dates, ("A", "B")), names=("date", "asset")),
)
gross_returns = pd.Series([0.010, -0.005], index=dates)
dollar_volume = pd.DataFrame({"A": [1_000.0, 1_500.0], "B": [2_000.0, 1_000.0]}, index=dates)
borrow_rates = pd.DataFrame({"A": [0.0, 0.0], "B": [0.002, 0.003]}, index=dates)
borrow_available = pd.DataFrame(True, index=dates, columns=("A", "B"))

ledger = apply_factor_costs(
    gross_returns,
    weights,
    dollar_volume,
    portfolio_value=250.0,
    model=FactorCostModel(
        half_spread_bps=10.0,
        impact_coefficient=0.01,
        impact_exponent=0.5,
        max_participation=0.50,
    ),
    borrow_rates=borrow_rates,
    borrow_available=borrow_available,
)

assert (ledger.participation <= ledger.model.max_participation).all().all()
assert (ledger.net_returns == ledger.gross_returns - ledger.total_cost).all()
```

The `weights` input is a two-level `(date, asset)` Series whose absolute
weights sum to one on every date. It is normally produced by
`factor_weights`; `group_adjust=True` is the existing enhanced route for
group-neutral weights. Missing `(date, asset)` entries in this sparse ledger
mean a zero position, so an entry or exit creates a real trade rather than an
unknown value. `gross_returns`, dollar volume and weights must cover exactly
the same rebalance dates; dollar volume must cover every asset and be strictly
positive. Dollar volume and `portfolio_value` must use the same reporting
currency—this API performs no FX conversion.

For weight `w[t, i]`, initial `w[-1, i] = 0`, and portfolio value `V`, the
ledger uses:

```text
q[t, i]          = abs(w[t, i] - w[t-1, i])
turnover[t]       = 0.5 * sum_i q[t, i]
participation[t,i]= q[t, i] * V / dollar_volume[t, i]
spread[t]         = sum_i q[t, i] * half_spread_bps / 10_000
impact[t]         = sum_i q[t, i] * impact_coefficient * participation[t,i] ** impact_exponent
borrow[t]         = sum_i max(-w[t, i], 0) * borrow_rate[t, i]
net[t]            = gross[t] - spread[t] - impact[t] - borrow[t]
capacity          = min_(t,i:q[t,i]>0) max_participation * dollar_volume[t,i] / q[t,i]
```

`max_participation` is a hard inequality, not a warning: a supplied portfolio
value above `capacity` fails closed. Any short exposure requires both a finite
per-period `borrow_rates` panel and a boolean `borrow_available` panel; an
unavailable borrow, missing asset/date, non-finite value, or invalid capacity
input also fails closed. Returned ledgers use defensive snapshots so changing a
returned pandas object cannot modify the stored result.

This is an arithmetic research ledger, not an execution simulator or a claim
that `impact_coefficient` is calibrated for a venue. Calibrate its assumptions
against the market, order type, and frequency being studied; retain that
calibration and the source/liquidity snapshot with the research record. The
temporary-impact form is compatible with the modelling family introduced by
[Almgren and Chriss](https://doi.org/10.21314/JOR.2001.041), but this API does
not implement their optimal execution model.

## Post-analysis IC inference and FDR

After enhanced analysis, run the explicit post-analysis step rather than
reading a raw IC average as a discovery claim:

```python
from fincore.factor_analysis import analyze_factor, factor_model_inference

model = analyze_factor(prepared.data, periods=("1D", "5D"), include_pyfolio=False)
inference = factor_model_inference(model, alpha=0.05)
audit_table = inference.hypotheses
```

`factor_model_inference` consumes the model's stored aggregate date-by-period
IC snapshot; it does not recompute returns or weights. For each forward
period, the audit table records finite sample count, mean IC, a two-sided
Student-t statistic and p-value, Benjamini-Hochberg adjusted p-value, and the
rejection decision. The BH family includes only rows with at least two finite
IC observations. Untestable rows remain visible with `testable=False`, `NaN`
p/q values, and `rejected=False`; they must not be reported as non-findings.

This inference path assumes independent IC observations. It is not HAC or
clustered inference, does not pre-register a hypothesis family, and does not
replace a research-trial register. Callers must define the tested horizons and
factor family before viewing results, retain the returned audit table, and
state any dependence correction or trial policy that is not yet supplied by
the platform.

## Required research evidence

For each strategy or factor study, retain the source snapshot identity,
corporate-action and calendar versions, universe construction rule, evaluation
timestamps, and the exact factor ledger used. Add a future-perturbation test:
changing an observation that is not yet known/effective must not change any
earlier materialized factor value. The repository keeps an executable
adversarial fixture for this property in
`tests/numerical/test_factor_pit_materialization.py`.

PIT materialization does not remove the need for out-of-sample validation,
multiple-testing control, calibrated cost/slippage/borrow assumptions,
capacity interpretation, or an explicit trial register. Treat an omitted
control as undisclosed, not as passing by default.
