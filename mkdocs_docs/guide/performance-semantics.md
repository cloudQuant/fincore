# Performance return semantics

`fincore.performance` is the enhanced performance layer. It does not change
the frozen Empyrical or Pyfolio compatibility surfaces. Its APIs make the
return convention, cashflow timing, fee treatment, and currency conversion
explicit so a report never silently selects a financial interpretation.

## Cashflow-adjusted time-weighted returns

Provide valuations in the reporting currency on a unique, increasing
timezone-aware `DatetimeIndex`, with strictly positive capital at every period
opening. A terminal zero valuation is allowed to represent a total loss. A
positive external cashflow is a contribution into the portfolio; a negative
one is a withdrawal. Every flow must have the same date as a valuation because
an unvalued intra-period flow has no defensible timing.

```python
import pandas as pd

from fincore.performance import cashflow_adjusted_returns, cashflow_adjusted_twr

dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"], utc=True)
valuations = pd.Series([100.0, 110.0, 121.0], index=dates)
cashflows = pd.Series([10.0], index=[dates[1]])  # contribution at 2024-02-29

period_returns = cashflow_adjusted_returns(valuations, cashflows, timing="end")
total_return = cashflow_adjusted_twr(valuations, cashflows, timing="end")

assert period_returns.round(12).tolist() == [0.0, 0.1]
assert round(total_return, 12) == 0.1
```

For an end-of-period flow, the period return is
`(V_end + fee_if_gross - flow) / V_start - 1`. For a start-of-period flow it
is `(V_end + fee_if_gross) / (V_start + flow) - 1`. Choose `timing="start"`
only when the operational record supports that convention; fincore never
infers it from a timestamp alone. If a valuation record has mixed start- and
end-of-period flows, pass a `cashflow_timings` Series with one `"start"` or
`"end"` value for every nonzero cashflow date; a partial timing ledger is
rejected rather than silently falling back to the scalar policy.

When multiple transactions share one valuation timestamp, do not net them
before calculation: use an event ledger with one row per transaction. Its
timezone-aware index must match the valuations and its only columns are
`amount` and `timing`; every row declares `"start"` or `"end"`. A single
`cashflow_currency` still applies to the whole request, so normalize a
mixed-currency ledger before calling this API.

```python
ledger = pd.DataFrame(
    {"amount": [10.0, -5.0], "timing": ["start", "end"]},
    index=[dates[1], dates[1]],
)
one_period = cashflow_adjusted_twr(
    pd.Series([100.0, 116.0], index=dates[:2]),
    ledger,
)
assert round(one_period, 12) == 0.1
```

## Fees and currencies

Returns are net-of-fees by default: the fee is already reflected in the ending
valuation. Use `fee_treatment="gross"` only with an explicit fee series in the
reporting currency to add those fees back for a gross result.

Cashflows in a different currency require a full FX series whose index is
exactly the valuation index. FX values mean reporting-currency units per one
cashflow-currency unit. Missing FX, unvalued flow dates, nonpositive capital,
and ambiguous timing are errors rather than silently adjusted values.

## Scope and disclosure

TWR measures the compound return after external-flow neutralization;
`mwr`/`xirr` measure money-weighted return with a separately documented,
conservative conventional-cashflow policy. Every enhanced strategy report now
renders a calculation disclosure. With a plain periodic return series, its
default is deliberately conservative: it says that cashflow and fee treatment
were not supplied and no cashflow adjustment was performed. It does **not**
silently label that series as TWR.

Pass a `DisclosureContext` only when the calculation record supports the
declarations. Its established defaults are themselves declarations: TWR,
gross-of-fees, no cashflows and annualized metrics. Therefore, treat every
context instance as a complete caller assertion, including when only one field
is overridden. Omit `disclosure_context` entirely to receive conservative
values derived from the validated report input. A legacy precomputed report
model without a disclosure is rendered only with its immutable model metadata
and explicit ``legacy/unknown`` provenance; later raw inputs are not consulted.
The resolved, structured disclosure is rendered in HTML/PDF. An optional audit
manifest records its sanitized form, redacting credentials and omitting local
paths rather than copying sensitive free-form text verbatim.

```python
import pandas as pd

from fincore.performance import DisclosureContext
from fincore.report import create_strategy_report

returns = pd.Series(
    [0.001 if day % 2 else -0.0005 for day in range(60)],
    index=pd.date_range("2024-01-02", periods=60, freq="B", tz="UTC"),
)
context = DisclosureContext(
    convention="TWR after external-flow neutralization",
    return_type="simple",
    units="decimal return per period",
    frequency="daily",
    fees="net-of-fees",
    cashflows="timed transaction ledger",
    benchmark="S&P 500 total return",
    risk_free="USD 3M Treasury",
)

# ``returns`` is the already-calculated, validated periodic return series.
artifacts = create_strategy_report(
    returns,
    output="report.html",
    disclosure_context=context,
    return_result=True,
    audit_manifest=True,
)
assert artifacts.model["performance_disclosure"]["convention"].startswith("TWR")
```

These helpers provide GIPS-aware calculation and disclosure support; they do
not certify GIPS compliance.
