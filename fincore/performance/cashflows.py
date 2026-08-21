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
    if len(valuations) < 2:
        raise ValueError("at least two valuations are required")
    if not valuations.index.is_unique or not valuations.index.is_monotonic_increasing:
        raise ValueError("valuation index must be unique and monotonically increasing")
    numeric = _as_finite_series(valuations, "valuations")
    if np.any(numeric.to_numpy(dtype=float) <= 0.0):
        raise ValueError("valuations must be finite and strictly positive")
    return numeric


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
    if not values.index.is_unique:
        raise ValueError(f"{name} index must be unique")
    if not values.index.isin(valuation_index).all():
        raise ValueError(f"{name} dates must be present in the valuation index")
    return _as_finite_series(values, name).reindex(valuation_index, fill_value=0.0)


def _reporting_cashflows(
    cashflows: pd.Series,
    valuation_index: pd.DatetimeIndex,
    *,
    cashflow_currency: str | None,
    reporting_currency: str,
    fx_rates: pd.Series | None,
) -> pd.Series:
    reporting = _normalized_currency(reporting_currency, "reporting_currency")
    source = _normalized_currency(cashflow_currency or reporting, "cashflow_currency")
    if source == reporting:
        return cashflows
    if fx_rates is None:
        raise ValueError("FX rates are required when cashflow and reporting currencies differ")
    if not isinstance(fx_rates, pd.Series) or not fx_rates.index.equals(valuation_index):
        raise ValueError("FX rates must use the same valuation index as valuations")
    rates = _as_finite_series(fx_rates, "FX rates")
    if np.any(rates.to_numpy(dtype=float) <= 0.0):
        raise ValueError("FX rates must be strictly positive")
    return cashflows * rates


def cashflow_adjusted_returns(
    valuations: pd.Series,
    cashflows: pd.Series | None = None,
    *,
    fees: pd.Series | None = None,
    timing: CashflowTiming = "end",
    fee_treatment: FeeTreatment = "net",
    cashflow_currency: str | None = None,
    reporting_currency: str = "USD",
    fx_rates: pd.Series | None = None,
) -> pd.Series:
    """Return per-period TWR returns after explicit cashflow adjustments.

    ``valuations`` must be a positive, unique, increasing ``DatetimeIndex``
    series in ``reporting_currency``.  Positive values in ``cashflows`` are
    portfolio contributions.  For an ``end`` flow the period return is
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
    raw_cashflows = _aligned_optional_series(cashflows, valuation_index, "cashflows")
    reporting_cashflows = _reporting_cashflows(
        raw_cashflows,
        valuation_index,
        cashflow_currency=cashflow_currency,
        reporting_currency=reporting_currency,
        fx_rates=fx_rates,
    )
    fee_series = _aligned_optional_series(fees, valuation_index, "fees")
    if reporting_cashflows.iloc[0] != 0.0 or fee_series.iloc[0] != 0.0:
        raise ValueError("cashflows and fees on the first valuation date are not assignable to a period")

    opening = valuation_series.iloc[:-1].to_numpy(dtype=float)
    closing = valuation_series.iloc[1:].to_numpy(dtype=float)
    flows = reporting_cashflows.iloc[1:].to_numpy(dtype=float)
    period_fees = fee_series.iloc[1:].to_numpy(dtype=float)
    closing_for_return = closing + period_fees if fee_treatment == "gross" else closing

    if timing == "end":
        capital_base = opening
        adjusted_closing = closing_for_return - flows
    else:
        capital_base = opening + flows
        adjusted_closing = closing_for_return
    if np.any(capital_base <= 0.0):
        raise ValueError("cashflow-adjusted capital base must be strictly positive")
    if np.any(adjusted_closing <= 0.0):
        raise ValueError("cashflow-adjusted closing capital must be strictly positive")

    return pd.Series(adjusted_closing / capital_base - 1.0, index=valuation_series.index[1:], dtype=float)


def cashflow_adjusted_twr(
    valuations: pd.Series,
    cashflows: pd.Series | None = None,
    *,
    fees: pd.Series | None = None,
    timing: CashflowTiming = "end",
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
        fee_treatment=fee_treatment,
        cashflow_currency=cashflow_currency,
        reporting_currency=reporting_currency,
        fx_rates=fx_rates,
    )
    return float(np.prod(1.0 + returns.to_numpy(dtype=float)) - 1.0)
