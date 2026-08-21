"""Explicit cashflow, fee, timing, and currency semantics for enhanced TWR.

Positive external cashflows are contributions into the portfolio and negative
cashflows are withdrawals.  A flow must occur on a valuation date, where its
``timing`` determines whether it adjusts the opening or closing capital of
that period.  Fees are denominated in the reporting currency and may either
remain in the net return or be added back for an explicitly gross-of-fees
return.  This module deliberately rejects ambiguous dates, capital bases, and
currency conversions instead of silently applying a convention.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd

CashflowTiming = Literal["end", "start"]
FeeTreatment = Literal["net", "gross"]

__all__ = [
    "CashflowTiming",
    "FeeTreatment",
    "cashflow_adjusted_returns",
    "cashflow_adjusted_twr",
]


def _normalized_currency(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty currency code")
    return value.strip().upper()


def _as_finite_series(values: pd.Series, name: str) -> pd.Series:
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain finite values")
    return numeric


def _validated_valuations(valuations: pd.Series) -> pd.Series:
    if not isinstance(valuations, pd.Series):
        raise TypeError("valuations must be a pandas Series")
    if not isinstance(valuations.index, pd.DatetimeIndex):
        raise ValueError("valuations must use a DatetimeIndex")
    if valuations.index.tz is None:
        raise ValueError("valuations must use a timezone-aware DatetimeIndex")
    if len(valuations) < 2:
        raise ValueError("at least two valuations are required")
    if not valuations.index.is_unique or not valuations.index.is_monotonic_increasing:
        raise ValueError("valuation index must be unique and monotonically increasing")
    numeric = _as_finite_series(valuations, "valuations")
    values = numeric.to_numpy(dtype=float)
    if np.any(values < 0.0) or np.any(values[:-1] <= 0.0):
        raise ValueError("valuations must be finite with a strictly positive opening capital")
    return numeric


def _require_matching_timezone(index: pd.DatetimeIndex, valuation_index: pd.DatetimeIndex, name: str) -> None:
    if index.tz is None:
        raise ValueError(f"{name} must use a timezone-aware DatetimeIndex")
    if index.tz != valuation_index.tz:
        raise ValueError(f"{name} timezone must match the valuation index")


def _aligned_optional_series(
    values: pd.Series | None,
    valuation_index: pd.DatetimeIndex,
    name: str,
) -> pd.Series:
    if values is None:
        return pd.Series(0.0, index=valuation_index, dtype=float)
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must use a DatetimeIndex")
    _require_matching_timezone(values.index, valuation_index, name)
    if not values.index.is_unique:
        raise ValueError(f"{name} index must be unique")
    if not values.index.isin(valuation_index).all():
        raise ValueError(f"{name} dates must be present in the valuation index")
    return _as_finite_series(values, name).reindex(valuation_index, fill_value=0.0)


def _aligned_cashflow_timings(
    cashflow_timings: pd.Series | None,
    cashflows: pd.Series,
    valuation_index: pd.DatetimeIndex,
    default: CashflowTiming,
) -> pd.Series:
    """Return one explicit timing policy for each valuation date.

    Omitting the optional ledger means the scalar ``timing`` is the declared
    policy for every flow.  Supplying a ledger makes timing per-flow metadata:
    it must cover every and only nonzero cashflow date, so a partial ledger
    cannot silently fall back to a different convention.
    """

    result = pd.Series(default, index=valuation_index, dtype=object)
    if cashflow_timings is None:
        return result
    if not isinstance(cashflow_timings, pd.Series):
        raise TypeError("cashflow_timings must be a pandas Series")
    if not isinstance(cashflow_timings.index, pd.DatetimeIndex):
        raise ValueError("cashflow_timings must use a DatetimeIndex")
    _require_matching_timezone(cashflow_timings.index, valuation_index, "cashflow_timings")
    if not cashflow_timings.index.is_unique:
        raise ValueError("cashflow_timings index must be unique")
    if not cashflow_timings.index.isin(valuation_index).all():
        raise ValueError("cashflow_timings dates must be present in the valuation index")

    flow_dates = pd.DatetimeIndex(cashflows[cashflows != 0.0].index)
    missing = flow_dates.difference(cashflow_timings.index)
    extra = cashflow_timings.index.difference(flow_dates)
    if len(missing) or len(extra):
        raise ValueError("cashflow_timings must cover every and only nonzero cashflow date")
    if not cashflow_timings.map(lambda value: isinstance(value, str) and value in {"start", "end"}).all():
        raise ValueError("cashflow_timings values must be either 'end' or 'start'")
    result.loc[cashflow_timings.index] = cashflow_timings
    return result


def _reporting_cashflows(
    cashflows: pd.Series,
    valuation_index: pd.DatetimeIndex,
    *,
    cashflow_currency: str | None,
    reporting_currency: str,
    fx_rates: pd.Series | None,
) -> pd.Series:
    reporting = _normalized_currency(reporting_currency, "reporting_currency")
    source = _normalized_currency(
        reporting if cashflow_currency is None else cashflow_currency,
        "cashflow_currency",
    )
    if source == reporting:
        if fx_rates is not None:
            raise ValueError("FX rates must be omitted when cashflow and reporting currencies match")
        return cashflows
    if fx_rates is None:
        raise ValueError("FX rates are required when cashflow and reporting currencies differ")
    if not isinstance(fx_rates, pd.Series) or not isinstance(fx_rates.index, pd.DatetimeIndex):
        raise ValueError("FX rates must use the same valuation index as valuations")
    _require_matching_timezone(fx_rates.index, valuation_index, "FX rates")
    if not fx_rates.index.equals(valuation_index):
        raise ValueError("FX rates must use the same valuation index as valuations")
    rates = _as_finite_series(fx_rates, "FX rates")
    if np.any(rates.to_numpy(dtype=float) <= 0.0):
        raise ValueError("FX rates must be strictly positive")
    aligned_rates = rates.reindex(cashflows.index)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = pd.Series(
            cashflows.to_numpy(dtype=float) * aligned_rates.to_numpy(dtype=float),
            index=cashflows.index,
            dtype=float,
        )
    converted_values = converted.to_numpy(dtype=float)
    if not np.isfinite(converted_values).all():
        raise ValueError("FX conversion produced non-finite cashflows")
    cashflow_values = cashflows.to_numpy(dtype=float)
    if np.any((cashflow_values != 0.0) & (converted_values == 0.0)):
        raise ValueError("FX conversion is not representable in the reporting currency")
    return converted


def _aggregate_cashflow_components(
    amounts: pd.Series,
    timings: pd.Series,
    valuation_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    """Aggregate transaction amounts only after their timing is explicit."""

    start_flows = (
        amounts[timings == "start"].groupby(level=0, sort=False).sum().reindex(valuation_index, fill_value=0.0)
    )
    end_flows = amounts[timings == "end"].groupby(level=0, sort=False).sum().reindex(valuation_index, fill_value=0.0)
    if (
        not np.isfinite(start_flows.to_numpy(dtype=float)).all()
        or not np.isfinite(end_flows.to_numpy(dtype=float)).all()
    ):
        raise ValueError("cashflow aggregation produced non-finite values")
    return start_flows.astype(float), end_flows.astype(float)


def _transaction_ledger_components(
    ledger: pd.DataFrame,
    valuation_index: pd.DatetimeIndex,
    *,
    cashflow_currency: str | None,
    reporting_currency: str,
    fx_rates: pd.Series | None,
) -> tuple[pd.Series, pd.Series]:
    """Validate a duplicated-timestamp event ledger and retain timing detail."""

    required_columns = {"amount", "timing"}
    if not ledger.columns.is_unique or set(ledger.columns) != required_columns:
        raise ValueError("cashflow ledger columns must be exactly {'amount', 'timing'}")
    if not isinstance(ledger.index, pd.DatetimeIndex):
        raise ValueError("cashflow ledger must use a DatetimeIndex")
    _require_matching_timezone(ledger.index, valuation_index, "cashflow ledger")
    if not ledger.index.isin(valuation_index).all():
        raise ValueError("cashflow ledger dates must be present in the valuation index")

    amounts = _as_finite_series(ledger["amount"], "cashflow ledger amounts")
    timings = ledger["timing"]
    if not timings.map(lambda value: isinstance(value, str) and value in {"start", "end"}).all():
        raise ValueError("cashflow ledger timing values must be either 'end' or 'start'")
    first_date_amounts = amounts.loc[amounts.index == valuation_index[0]]
    if np.any(first_date_amounts.to_numpy(dtype=float) != 0.0):
        raise ValueError("cashflows on the first valuation date are not assignable to a period")

    reporting_amounts = _reporting_cashflows(
        amounts,
        valuation_index,
        cashflow_currency=cashflow_currency,
        reporting_currency=reporting_currency,
        fx_rates=fx_rates,
    )
    return _aggregate_cashflow_components(reporting_amounts, timings, valuation_index)


def _cashflow_components(
    cashflows: pd.Series | pd.DataFrame | None,
    cashflow_timings: pd.Series | None,
    valuation_index: pd.DatetimeIndex,
    *,
    timing: CashflowTiming,
    cashflow_currency: str | None,
    reporting_currency: str,
    fx_rates: pd.Series | None,
) -> tuple[pd.Series, pd.Series]:
    """Return independently aggregated start- and end-of-period cashflows.

    A Series keeps the original one-net-flow-per-valuation-date API. A
    DataFrame ledger supports multiple transactions at one valuation time and
    therefore requires a timing column on every row.
    """

    if cashflows is None:
        if cashflow_timings is not None:
            raise ValueError("cashflow_timings requires a pandas Series of cashflows")
        zeros = pd.Series(0.0, index=valuation_index, dtype=float)
        return zeros, zeros.copy()
    if isinstance(cashflows, pd.DataFrame):
        if cashflow_timings is not None:
            raise ValueError("cashflow_timings is not valid with a transaction ledger")
        return _transaction_ledger_components(
            cashflows,
            valuation_index,
            cashflow_currency=cashflow_currency,
            reporting_currency=reporting_currency,
            fx_rates=fx_rates,
        )
    if not isinstance(cashflows, pd.Series):
        raise TypeError("cashflows must be a pandas Series or DataFrame ledger")

    raw_cashflows = _aligned_optional_series(cashflows, valuation_index, "cashflows")
    reporting_cashflows = _reporting_cashflows(
        raw_cashflows,
        valuation_index,
        cashflow_currency=cashflow_currency,
        reporting_currency=reporting_currency,
        fx_rates=fx_rates,
    )
    timing_series = _aligned_cashflow_timings(cashflow_timings, reporting_cashflows, valuation_index, timing)
    start_flows = reporting_cashflows.where(timing_series == "start", 0.0)
    end_flows = reporting_cashflows.where(timing_series == "end", 0.0)
    return start_flows.astype(float), end_flows.astype(float)


def cashflow_adjusted_returns(
    valuations: pd.Series,
    cashflows: pd.Series | pd.DataFrame | None = None,
    *,
    fees: pd.Series | None = None,
    timing: CashflowTiming = "end",
    cashflow_timings: pd.Series | None = None,
    fee_treatment: FeeTreatment = "net",
    cashflow_currency: str | None = None,
    reporting_currency: str = "USD",
    fx_rates: pd.Series | None = None,
) -> pd.Series:
    """Return per-period TWR returns after explicit cashflow adjustments.

    ``valuations`` must be a unique, increasing ``DatetimeIndex`` series in
    ``reporting_currency`` with strictly positive opening capital. A terminal
    zero valuation is allowed to represent a total loss. Positive values in
    ``cashflows`` are portfolio contributions. A Series represents one net
    flow per valuation date: the scalar ``timing`` is its declared policy, or
    ``cashflow_timings`` provides an explicit ``start``/``end`` value for every
    nonzero date. A DataFrame ledger supports multiple same-date transactions,
    has exactly ``amount`` and ``timing`` columns, and declares timing on every
    row. For an ``end`` flow the period return is
    ``(V_end + fee_if_gross - flow) / V_start - 1``; for ``start`` it is
    ``(V_end + fee_if_gross) / (V_start + flow) - 1``.  ``fees`` are already
    in reporting currency.  A non-reporting-currency cashflow requires a full,
    same-index FX series expressed as reporting currency per cashflow currency.

    The first valuation is an opening observation, so nonzero fees or flows on
    it are rejected rather than silently assigned to an unknown prior period.
    """

    if timing not in ("end", "start"):
        raise ValueError("timing must be either 'end' or 'start'")
    if fee_treatment not in ("net", "gross"):
        raise ValueError("fee_treatment must be either 'net' or 'gross'")

    valuation_series = _validated_valuations(valuations)
    valuation_index = cast("pd.DatetimeIndex", valuation_series.index)
    start_flow_series, end_flow_series = _cashflow_components(
        cashflows,
        cashflow_timings,
        valuation_index,
        timing=timing,
        cashflow_currency=cashflow_currency,
        reporting_currency=reporting_currency,
        fx_rates=fx_rates,
    )
    fee_series = _aligned_optional_series(fees, valuation_index, "fees")
    if start_flow_series.iloc[0] != 0.0 or end_flow_series.iloc[0] != 0.0 or fee_series.iloc[0] != 0.0:
        raise ValueError("cashflows and fees on the first valuation date are not assignable to a period")

    opening = valuation_series.iloc[:-1].to_numpy(dtype=float)
    closing = valuation_series.iloc[1:].to_numpy(dtype=float)
    start_flows = start_flow_series.iloc[1:].to_numpy(dtype=float)
    end_flows = end_flow_series.iloc[1:].to_numpy(dtype=float)
    period_fees = fee_series.iloc[1:].to_numpy(dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        closing_for_return = closing + period_fees if fee_treatment == "gross" else closing
        capital_base = opening + start_flows
        adjusted_closing = closing_for_return - end_flows
    if not np.isfinite(closing_for_return).all():
        raise ValueError("fee adjustment produced non-finite closing capital")
    if not np.isfinite(capital_base).all():
        raise ValueError("cashflow-adjusted capital base must be finite")
    if not np.isfinite(adjusted_closing).all():
        raise ValueError("cashflow-adjusted closing capital must be finite")
    if np.any(capital_base <= 0.0):
        raise ValueError("cashflow-adjusted capital base must be strictly positive")
    if np.any(adjusted_closing < 0.0):
        raise ValueError("cashflow-adjusted closing capital must be non-negative")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        gross_factors = adjusted_closing / capital_base
        returns = gross_factors - 1.0
    if not np.isfinite(gross_factors).all() or not np.isfinite(returns).all():
        raise ValueError("cashflow-adjusted returns must be finite")
    if np.any((adjusted_closing > 0.0) & (returns == -1.0)):
        raise ValueError("cashflow-adjusted return factor is not representable")

    return pd.Series(returns, index=valuation_series.index[1:], dtype=float)


def cashflow_adjusted_twr(
    valuations: pd.Series,
    cashflows: pd.Series | pd.DataFrame | None = None,
    *,
    fees: pd.Series | None = None,
    timing: CashflowTiming = "end",
    cashflow_timings: pd.Series | None = None,
    fee_treatment: FeeTreatment = "net",
    cashflow_currency: str | None = None,
    reporting_currency: str = "USD",
    fx_rates: pd.Series | None = None,
) -> float:
    """Compound :func:`cashflow_adjusted_returns` into a total TWR result."""

    returns = cashflow_adjusted_returns(
        valuations,
        cashflows,
        fees=fees,
        timing=timing,
        cashflow_timings=cashflow_timings,
        fee_treatment=fee_treatment,
        cashflow_currency=cashflow_currency,
        reporting_currency=reporting_currency,
        fx_rates=fx_rates,
    )
    factors = 1.0 + returns.to_numpy(dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compounded = float(np.prod(factors))
        total_return = compounded - 1.0
    if not np.isfinite(compounded) or not np.isfinite(total_return):
        raise ValueError("cashflow-adjusted TWR must be finite")
    if (compounded == 0.0 and np.all(factors > 0.0)) or (compounded > 0.0 and total_return == -1.0):
        raise ValueError("cashflow-adjusted TWR factor is not representable")
    return total_return
