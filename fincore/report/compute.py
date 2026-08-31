"""Statistics computation engine for strategy reports.

Computes all metrics, time-series data, and summary text needed by the
HTML / PDF renderers.  This module has **no** rendering logic.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import replace

import numpy as np
import pandas as pd

from fincore.exceptions import InputContractError
from fincore.metrics.alpha_beta import alpha_beta
from fincore.metrics.consecutive import (
    max_consecutive_down_days,
    max_consecutive_down_months,
    max_consecutive_down_weeks,
    max_consecutive_up_days,
    max_consecutive_up_months,
    max_consecutive_up_weeks,
)
from fincore.metrics.drawdown import (
    gen_drawdown_table,
    max_drawdown,
    max_drawdown_days,
    max_drawdown_months,
    max_drawdown_recovery_days,
    max_drawdown_recovery_months,
    max_drawdown_recovery_weeks,
    max_drawdown_weeks,
    second_max_drawdown,
    third_max_drawdown,
)
from fincore.metrics.ratios import (
    burke_ratio,
    calmar_ratio,
    common_sense_ratio,
    down_capture,
    information_ratio,
    kappa_three_ratio,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    stability_of_timeseries,
    sterling_ratio,
    up_capture,
)
from fincore.metrics.returns import aggregate_returns, cum_returns, cum_returns_final
from fincore.metrics.risk import annual_volatility, downside_risk, tail_ratio, tracking_error, value_at_risk
from fincore.metrics.rolling import rolling_beta, rolling_sharpe, rolling_volatility
from fincore.metrics.stats import hurst_exponent, kurtosis, loss_rate, serial_correlation, skewness, win_rate
from fincore.metrics.yearly import annual_return, annual_return_by_year, max_drawdown_by_year, sharpe_ratio_by_year
from fincore.portfolio.positions import gross_lev
from fincore.portfolio.transactions import get_turnover
from fincore.report.model import ReportModel

__all__ = ["ReportModel", "compute_sections"]


# Module-level logger
logger = logging.getLogger(__name__)


def _period_title(period):
    return {
        "daily": "Daily",
        "weekly": "Weekly",
        "monthly": "Monthly",
    }.get(period, "Daily")


def _period_unit(period):
    return {
        "daily": "Day",
        "weekly": "Week",
        "monthly": "Month",
    }.get(period, "Day")


def _period_unit_plural(period):
    return {
        "daily": "trading days",
        "weekly": "weeks",
        "monthly": "months",
    }.get(period, "trading days")


def _approx_months(period_count, period):
    if period == "weekly":
        return int(period_count * 12 / 52)
    if period == "monthly":
        return int(period_count)
    return int(period_count / 21)


def _period_defs(period):
    if period == "weekly":
        return [("1W", 1), ("1M", 4), ("3M", 13), ("6M", 26), ("1Y", 52), ("3Y", 156), ("5Y", 260)]
    if period == "monthly":
        return [("1M", 1), ("3M", 3), ("6M", 6), ("1Y", 12), ("3Y", 36), ("5Y", 60)]
    return _PERIOD_DEFS


_DEFAULT_DISCLOSURE_NOTE = "GIPS-aware disclosure support; not GIPS compliance certification."
_LEGACY_DISCLOSURE_NOTE = "Legacy precomputed model: raw-input validation and calculation provenance unavailable."


def _disclosure_text(value, default):
    """Keep report disclosures complete when a caller leaves a field blank."""

    return value if isinstance(value, str) and value.strip() else default


def _merge_disclosure_context(default, disclosure_context):
    """Merge non-empty caller declarations into a provenance-backed default."""

    from fincore.performance.disclosures import DisclosureContext

    if disclosure_context is None:
        return default
    if not isinstance(disclosure_context, DisclosureContext):
        raise TypeError("disclosure_context must be a DisclosureContext")

    return replace(
        default,
        convention=_disclosure_text(disclosure_context.convention, default.convention),
        sample_period=_disclosure_text(disclosure_context.sample_period, default.sample_period),
        data_quality=_disclosure_text(disclosure_context.data_quality, default.data_quality),
        fees=_disclosure_text(disclosure_context.fees, default.fees),
        cashflows=_disclosure_text(disclosure_context.cashflows, default.cashflows),
        benchmark=_disclosure_text(disclosure_context.benchmark, default.benchmark),
        risk_free=_disclosure_text(disclosure_context.risk_free, default.risk_free),
        annualized=bool(disclosure_context.annualized),
        notes=tuple(disclosure_context.notes) or default.notes,
        return_type=_disclosure_text(disclosure_context.return_type, default.return_type),
        units=_disclosure_text(disclosure_context.units, default.units),
        frequency=_disclosure_text(disclosure_context.frequency, default.frequency),
    )


def _resolved_performance_disclosure(returns, benchmark_rets, period, disclosure_context):
    """Resolve complete, honest report disclosure from validated inputs.

    Generic strategy reports receive a periodic returns series, not an
    independently auditable valuation/cashflow ledger.  The default therefore
    declares that limitation instead of claiming a TWR or fee convention.
    Callers that computed an enhanced TWR/MWR series can provide an explicit
    :class:`DisclosureContext`; missing sample/data-quality fields remain
    derived from the validated report input.
    """

    from fincore.performance.disclosures import DisclosureContext

    sample_period = (
        f"{returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')} "
        f"({len(returns)} {period} observations)"
    )
    default = DisclosureContext(
        convention="Simple periodic returns; geometrically compounded",
        sample_period=sample_period,
        data_quality=f"{len(returns)} finite observations; unique, increasing DatetimeIndex validated",
        fees="not supplied; caller-defined return series",
        cashflows="not supplied; no cashflow adjustment applied",
        benchmark="benchmark return series supplied" if benchmark_rets is not None else "none supplied",
        risk_free="not supplied; ratios use documented defaults",
        annualized=True,
        notes=(_DEFAULT_DISCLOSURE_NOTE,),
        return_type="simple",
        units="decimal return per period",
        frequency=period,
    )
    return _merge_disclosure_context(default, disclosure_context)


def _legacy_performance_disclosure(model, disclosure_context):
    """Resolve a fail-closed disclosure for a pre-disclosure report model.

    Legacy models are already computed, so their raw inputs must never affect
    a later rendering.  The durable model metadata supplies only the sample
    range/count/frequency; every unavailable financial semantic is labelled as
    such and a caller may add explicit declarations through ``DisclosureContext``.
    """

    from fincore.performance.disclosures import DisclosureContext

    raw_period = model.get("period")
    period = raw_period if isinstance(raw_period, str) and raw_period.strip() else "unknown"
    raw_date_range = model.get("date_range")
    raw_count = model.get("n_periods")
    if (
        isinstance(raw_date_range, (tuple, list))
        and len(raw_date_range) == 2
        and all(isinstance(value, str) and value for value in raw_date_range)
        and isinstance(raw_count, int)
        and raw_count >= 0
    ):
        sample_period = f"{raw_date_range[0]} to {raw_date_range[1]} ({raw_count} {period} observations)"
    else:
        sample_period = "not recorded in legacy precomputed model"

    default = DisclosureContext(
        convention="legacy precomputed model; calculation convention unavailable",
        sample_period=sample_period,
        data_quality="legacy precomputed model; raw-input validation provenance unavailable",
        fees="not recorded in legacy precomputed model",
        cashflows="not recorded in legacy precomputed model",
        benchmark="not recorded in legacy precomputed model",
        risk_free="not recorded in legacy precomputed model",
        annualized=True,
        notes=(_LEGACY_DISCLOSURE_NOTE, _DEFAULT_DISCLOSURE_NOTE),
        return_type="unknown",
        units="unknown",
        frequency=period,
    )
    return _merge_disclosure_context(default, disclosure_context)


def _disclosure_payload(context):
    """Return the JSON-safe disclosure shape shared by renderers and provenance."""

    return {
        "convention": context.convention,
        "return_type": context.return_type,
        "units": context.units,
        "frequency": context.frequency,
        "sample_period": context.sample_period,
        "data_quality": context.data_quality,
        "fees": context.fees,
        "cashflows": context.cashflows,
        "benchmark": context.benchmark,
        "risk_free": context.risk_free,
        "annualized": bool(context.annualized),
        "notes": [str(note) for note in context.notes],
    }


def _compute_core_perf(
    returns,
    benchmark_rets,
    period,
    *,
    positions=None,
    transactions=None,
    gross_leverage=None,
    turnover=None,
):
    """Compute core performance statistics."""
    period_title = _period_title(period)
    period_unit = _period_unit(period)
    perf = OrderedDict()
    perf["Annual Return"] = annual_return(returns, period=period)
    perf["Cumulative Returns"] = cum_returns_final(returns)
    perf["Annual Volatility"] = annual_volatility(returns, period=period)
    perf["Sharpe Ratio"] = sharpe_ratio(returns, period=period)
    perf["Calmar Ratio"] = calmar_ratio(returns, period=period)
    perf["Stability"] = stability_of_timeseries(returns)
    perf["Max Drawdown"] = max_drawdown(returns)
    perf["Omega Ratio"] = omega_ratio(returns)
    perf["Sortino Ratio"] = sortino_ratio(returns, period=period)
    perf["Skew"] = skewness(returns)
    perf["Kurtosis"] = kurtosis(returns)
    perf["Tail Ratio"] = tail_ratio(returns)
    perf[f"{period_title} Value at Risk"] = value_at_risk(returns)
    perf["Downside Risk"] = downside_risk(returns, period=period)

    perf[f"{period_title} Mean Return"] = float(np.nanmean(returns))
    perf[f"{period_title} Std Return"] = float(np.nanstd(returns, ddof=1))
    perf[f"Best {period_unit}"] = float(returns.max())
    perf[f"Worst {period_unit}"] = float(returns.min())

    if benchmark_rets is not None:
        alpha_value, beta_value = alpha_beta(returns, benchmark_rets, period=period)
        perf["Alpha"] = alpha_value
        perf["Beta"] = beta_value

    if positions is not None:
        leverage = gross_leverage if gross_leverage is not None else gross_lev(positions)
        perf["Avg Gross Leverage"] = float(leverage.mean())
        perf["Max Gross Leverage"] = float(leverage.max())
    if positions is not None and transactions is not None and turnover is not None:
        perf[f"Avg {period_title} Turnover"] = float(turnover.mean())

    return perf


def _compute_extended_stats(returns, period):
    """Compute extended strategy statistics."""
    period_unit_plural = _period_unit_plural(period)
    period_unit = _period_unit(period)
    ext = OrderedDict()
    ext[f"Win Rate ({period})"] = win_rate(returns)
    ext[f"Loss Rate ({period})"] = loss_rate(returns)
    ext["Serial Correlation"] = serial_correlation(returns)
    ext["Common Sense Ratio"] = common_sense_ratio(returns)
    ext["Sterling Ratio"] = sterling_ratio(returns, period=period)
    ext["Burke Ratio"] = burke_ratio(returns, period=period)
    ext["Kappa Three Ratio"] = kappa_three_ratio(returns, period=period)
    ext["2nd Max Drawdown"] = second_max_drawdown(returns)
    ext["3rd Max Drawdown"] = third_max_drawdown(returns)
    if period == "weekly":
        ext[f"Max Drawdown {period_unit_plural.title()}"] = max_drawdown_weeks(returns)
        ext[f"Max Drawdown Recovery {period_unit_plural.title()}"] = max_drawdown_recovery_weeks(returns)
        ext[f"Max Consecutive Up {period_unit_plural.title()}"] = max_consecutive_up_weeks(returns)
        ext[f"Max Consecutive Down {period_unit_plural.title()}"] = max_consecutive_down_weeks(returns)
    elif period == "monthly":
        ext[f"Max Drawdown {period_unit_plural.title()}"] = max_drawdown_months(returns)
        ext[f"Max Drawdown Recovery {period_unit_plural.title()}"] = max_drawdown_recovery_months(returns)
        ext[f"Max Consecutive Up {period_unit_plural.title()}"] = max_consecutive_up_months(returns)
        ext[f"Max Consecutive Down {period_unit_plural.title()}"] = max_consecutive_down_months(returns)
    else:
        ext[f"Max Drawdown {period_unit_plural.title()}"] = max_drawdown_days(returns)
        ext[f"Max Drawdown Recovery {period_unit_plural.title()}"] = max_drawdown_recovery_days(returns)
        ext[f"Max Consecutive Up {period_unit_plural.title()}"] = max_consecutive_up_days(returns)
        ext[f"Max Consecutive Down {period_unit_plural.title()}"] = max_consecutive_down_days(returns)
    ext[f"Max Single {period_unit} Gain"] = float(returns.max())
    ext[f"Max Single {period_unit} Loss"] = float(returns.min())
    ext["Hurst Exponent"] = hurst_exponent(returns)
    return ext


def _compute_time_series(returns, rolling_window, period):
    """Compute time-series data for charts."""
    ts = {}
    ts["returns"] = returns
    ts["cum_returns"] = cum_returns(returns, starting_value=1.0)
    cum_ret_0 = cum_returns(returns, starting_value=0)
    running_max = (1 + cum_ret_0).cummax()
    ts["drawdown"] = (1 + cum_ret_0) / running_max - 1
    ts["rolling_sharpe"] = rolling_sharpe(returns, rolling_sharpe_window=rolling_window, period=period)
    ts["rolling_volatility"] = rolling_volatility(returns, rolling_vol_window=rolling_window, period=period)
    ts["dd_table"] = gen_drawdown_table(returns, top=5)

    ts["yearly_stats"] = pd.DataFrame(
        {
            "Annual Return": annual_return_by_year(returns, period=period),
            "Sharpe Ratio": sharpe_ratio_by_year(returns, period=period),
            "Max Drawdown": max_drawdown_by_year(returns),
        }
    )
    ts["monthly_returns"] = aggregate_returns(returns, "monthly")
    monthly_rets = ts["monthly_returns"]
    ts["best_month"] = float(monthly_rets.max())
    ts["worst_month"] = float(monthly_rets.min())
    ts["avg_month"] = float(monthly_rets.mean())
    yearly_rets = aggregate_returns(returns, "yearly")
    ts["best_year"] = float(yearly_rets.max())
    ts["worst_year"] = float(yearly_rets.min())
    ts["return_quantiles"] = returns.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return ts


def _compute_benchmark(returns, benchmark_rets, perf, rolling_window, period):
    """Compute benchmark comparison stats."""
    bm = OrderedDict()
    bm["Alpha"] = perf["Alpha"]
    bm["Beta"] = perf["Beta"]
    bm["Information Ratio"] = information_ratio(returns, benchmark_rets, period=period)
    bm["Tracking Error"] = tracking_error(returns, benchmark_rets, period=period)
    bm["Up Capture"] = up_capture(returns, benchmark_rets, period=period)
    bm["Down Capture"] = down_capture(returns, benchmark_rets, period=period)
    bm["Capture Ratio"] = bm["Up Capture"] / bm["Down Capture"] if bm["Down Capture"] != 0 else np.nan
    bm["Correlation"] = float(returns.corr(benchmark_rets))
    return {
        "benchmark_stats": bm,
        "benchmark_cum": cum_returns(benchmark_rets, starting_value=1.0),
        "rolling_beta": rolling_beta(returns, benchmark_rets, rolling_window=rolling_window),
    }


def _compute_positions(positions, gross_leverage=None):
    """Compute position analysis."""
    s = {"has_positions": True}
    pos_no_cash = positions.drop("cash", axis=1, errors="ignore")
    s["positions"] = positions
    s["pos_no_cash"] = pos_no_cash
    s["pos_long"] = pos_no_cash.where(pos_no_cash > 0, 0).sum(axis=1)
    s["pos_short"] = pos_no_cash.where(pos_no_cash < 0, 0).sum(axis=1)
    total = positions.sum(axis=1).replace(0, np.nan)
    exposure = pos_no_cash.abs().sum(axis=1)
    s["gross_leverage"] = (
        gross_leverage if gross_leverage is not None else (exposure / total).replace([np.inf, -np.inf], np.nan)
    )

    pos_abs = pos_no_cash.abs()
    pos_total = pos_abs.sum(axis=1).replace(0, np.nan)
    pos_pct = pos_abs.div(pos_total, axis=0).fillna(0)
    s["pos_max_concentration"] = pos_pct.max(axis=1)
    s["pos_median_concentration"] = pos_pct.median(axis=1)
    s["pos_alloc"] = pos_no_cash.div(total, axis=0).fillna(0)

    pos_summary = OrderedDict()
    pos_summary["Avg Gross Leverage"] = s["gross_leverage"].mean()
    pos_summary["Max Gross Leverage"] = s["gross_leverage"].max()
    pos_summary["Avg Long Exposure"] = s["pos_long"].mean()
    pos_summary["Avg Short Exposure"] = s["pos_short"].mean()
    pos_summary["Avg Max Position Concentration"] = s["pos_max_concentration"].mean()
    pos_summary["Number of Assets"] = len(pos_no_cash.columns)
    s["position_summary"] = pos_summary
    return s


def _compute_transactions(transactions, positions, turnover=None):
    """Compute transaction analysis."""
    s = {"has_transactions": True}
    txn = transactions.copy()
    txn_norm = txn.copy()
    txn_norm.index = txn_norm.index.normalize()
    s["daily_txn_count"] = txn_norm.groupby(txn_norm.index).size()
    s["daily_txn_value"] = (txn_norm["amount"].abs() * txn_norm["price"]).groupby(txn_norm.index).sum()

    if hasattr(txn.index, "hour"):
        s["txn_hours"] = txn.index.hour

    if turnover is not None:
        s["turnover"] = turnover
    elif positions is not None:
        try:
            s["turnover"] = get_turnover(positions, transactions)
        except (ValueError, TypeError, KeyError, ZeroDivisionError) as e:
            logger.warning("Failed to calculate turnover from transactions: %s", e)

    txn_summary = OrderedDict()
    txn_summary["Total Transactions"] = len(transactions)
    txn_summary["Total Transaction Days"] = len(s["daily_txn_count"])
    txn_summary["Avg Daily Trades"] = float(s["daily_txn_count"].mean())
    txn_summary["Max Daily Trades"] = int(s["daily_txn_count"].max())
    txn_summary["Avg Daily Volume"] = float(s["daily_txn_value"].mean())
    txn_summary["Max Daily Volume"] = float(s["daily_txn_value"].max())
    if "symbol" in transactions.columns:
        txn_summary["Unique Symbols Traded"] = int(transactions["symbol"].nunique())
    s["txn_summary"] = txn_summary
    return s


def _compute_trades(trades):
    """Compute trade-level statistics."""
    s = {}
    ts = OrderedDict()
    n_trades = len(trades)
    winners = trades[trades["pnlcomm"] > 0]
    losers = trades[trades["pnlcomm"] <= 0]
    n_win = len(winners)
    n_loss = len(losers)

    ts["Total Trades"] = n_trades
    ts["Winning Trades"] = n_win
    ts["Losing Trades"] = n_loss
    ts["Win Rate"] = n_win / n_trades if n_trades > 0 else 0
    ts["Total PnL"] = float(trades["pnlcomm"].sum())
    ts["Avg PnL per Trade"] = float(trades["pnlcomm"].mean())
    ts["Median PnL per Trade"] = float(trades["pnlcomm"].median())
    ts["PnL Std Dev"] = float(trades["pnlcomm"].std())
    ts["Avg Win"] = float(winners["pnlcomm"].mean()) if n_win > 0 else 0
    ts["Avg Loss"] = float(losers["pnlcomm"].mean()) if n_loss > 0 else 0
    ts["Max Win"] = float(winners["pnlcomm"].max()) if n_win > 0 else 0
    ts["Max Loss"] = float(losers["pnlcomm"].min()) if n_loss > 0 else 0
    avg_loss = ts["Avg Loss"]
    ts["Profit/Loss Ratio"] = abs(ts["Avg Win"] / avg_loss) if avg_loss != 0 else np.nan
    ts["Expectancy"] = ts["Win Rate"] * ts["Avg Win"] + (1 - ts["Win Rate"]) * ts["Avg Loss"]

    if "commission" in trades.columns:
        ts["Total Commission"] = float(trades["commission"].sum())
        ts["Avg Commission per Trade"] = float(trades["commission"].mean())

    if "long" in trades.columns:
        long_mask = trades["long"] == 1
        short_mask = ~long_mask
        ts["Long Trades"] = int(long_mask.sum())
        ts["Short Trades"] = int(short_mask.sum())
        if long_mask.sum() > 0:
            long_trades = trades[long_mask]
            ts["Long Win Rate"] = float((long_trades["pnlcomm"] > 0).sum() / len(long_trades))
            ts["Long Avg PnL"] = float(long_trades["pnlcomm"].mean())
            ts["Long Total PnL"] = float(long_trades["pnlcomm"].sum())
        if short_mask.sum() > 0:
            short_trades = trades[short_mask]
            ts["Short Win Rate"] = float((short_trades["pnlcomm"] > 0).sum() / len(short_trades))
            ts["Short Avg PnL"] = float(short_trades["pnlcomm"].mean())
            ts["Short Total PnL"] = float(short_trades["pnlcomm"].sum())

    if "barlen" in trades.columns:
        ts["Avg Holding Bars"] = float(trades["barlen"].mean())
        ts["Median Holding Bars"] = float(trades["barlen"].median())
        ts["Max Holding Bars"] = int(trades["barlen"].max())
        ts["Min Holding Bars"] = int(trades["barlen"].min())

    s["trade_stats"] = ts
    s["trade_pnl"] = trades["pnlcomm"].values
    if "long" in trades.columns:
        s["trade_pnl_long"] = trades.loc[trades["long"] == 1, "pnlcomm"].values
        s["trade_pnl_short"] = trades.loc[trades["long"] == 0, "pnlcomm"].values
    if "barlen" in trades.columns:
        s["trade_barlen"] = trades["barlen"].values
    return s


_PERIOD_DEFS = [
    ("1W", 5),
    ("1M", 21),
    ("3M", 63),
    ("6M", 126),
    ("1Y", 252),
    ("3Y", 756),
    ("5Y", 1260),
]


def _compute_period_returns(returns, benchmark_rets, period):
    """Compute period returns and win rates."""
    s = {}
    end_date = returns.index[-1]
    _tz = getattr(end_date, "tzinfo", None)
    _ytd_ts = pd.Timestamp(end_date.year, 1, 1, tz=_tz)
    ytd_mask = returns.index >= _ytd_ts

    pr = OrderedDict()
    for label, days in _period_defs(period):
        pr[label] = float(cum_returns_final(returns.iloc[-days:])) if len(returns) >= days else np.nan
    if ytd_mask.sum() > 0:
        pr["YTD"] = float(cum_returns_final(returns[ytd_mask]))
    pr["Since Inception"] = float(cum_returns_final(returns))
    s["period_returns"] = pr

    if benchmark_rets is not None:
        _bm_tz = getattr(benchmark_rets.index[-1], "tzinfo", None)
        _bm_ytd_ts = pd.Timestamp(end_date.year, 1, 1, tz=_bm_tz)
        bpr = OrderedDict()
        for label, days in _period_defs(period):
            bpr[label] = (
                float(cum_returns_final(benchmark_rets.iloc[-days:])) if len(benchmark_rets) >= days else np.nan
            )
        bm_ytd = benchmark_rets[benchmark_rets.index >= _bm_ytd_ts]
        if len(bm_ytd) > 0:
            bpr["YTD"] = float(cum_returns_final(bm_ytd))
        bpr["Since Inception"] = float(cum_returns_final(benchmark_rets))
        s["benchmark_period_returns"] = bpr

    wr = OrderedDict()
    for label, days in _period_defs(period):
        if len(returns) >= days:
            r = returns.iloc[-days:]
            wr[label] = float((r > 0).sum() / len(r))
        else:
            wr[label] = np.nan
    ytd_r = returns[ytd_mask]
    if len(ytd_r) > 0:
        wr["YTD"] = float((ytd_r > 0).sum() / len(ytd_r))
    wr["Since Inception"] = float((returns > 0).sum() / len(returns))
    s["period_win_rates"] = wr
    return s


def _perf_tag(sh):
    if np.isnan(sh):
        return "N/A"
    return "excellent" if sh > 1.5 else ("good" if sh > 1.0 else ("fair" if sh > 0.5 else "poor"))


def _risk_tag(dd):
    if np.isnan(dd):
        return "N/A"
    a = abs(dd)
    return (
        "risk control: excellent"
        if a < 0.1
        else ("risk control: good" if a < 0.2 else ("risk control: fair" if a < 0.3 else "risk control: poor"))
    )


def _compute_summary_text(perf, benchmark_rets):
    """Generate human-readable summary text."""
    _ann = perf.get("Annual Return", np.nan)
    _shp = perf.get("Sharpe Ratio", np.nan)
    _mdd = perf.get("Max Drawdown", np.nan)
    _vol = perf.get("Annual Volatility", np.nan)
    _sor = perf.get("Sortino Ratio", np.nan)
    _cal = perf.get("Calmar Ratio", np.nan)

    txt = (
        f"Over the report period, annual return is {_ann * 100:.2f}% ({_perf_tag(_shp)}). "
        f"Sharpe={_shp:.2f}, Sortino={_sor:.2f}, Calmar={_cal:.2f}. "
        f"Max drawdown={abs(_mdd) * 100:.2f}%, annual volatility={_vol * 100:.2f}%, {_risk_tag(_mdd)}."
    )
    if benchmark_rets is not None:
        _a = perf.get("Alpha", np.nan)
        _b = perf.get("Beta", np.nan)
        txt += f" Alpha={_a:.4f}, Beta={_b:.4f}."
    return txt


def _validate_returns_input(value, *, parameter):
    """Own one finite, chronological report input without entering a facade."""

    if not isinstance(value, pd.Series):
        raise InputContractError("must be a pandas Series", operation_id="report.compute_sections", parameter=parameter)
    if value.empty:
        raise InputContractError(
            "must contain at least one observation",
            operation_id="report.compute_sections",
            parameter=parameter,
        )
    if not isinstance(value.index, pd.DatetimeIndex):
        raise InputContractError(
            "must use a DatetimeIndex",
            operation_id="report.compute_sections",
            parameter=parameter,
        )
    if not value.index.is_unique or not value.index.is_monotonic_increasing:
        raise InputContractError(
            "index must be unique and increasing",
            operation_id="report.compute_sections",
            parameter=parameter,
        )
    try:
        copied = value.astype(float).copy(deep=True)
    except (TypeError, ValueError) as error:
        raise InputContractError(
            "must contain numeric values",
            operation_id="report.compute_sections",
            parameter=parameter,
        ) from error
    if not bool(np.isfinite(copied.to_numpy()).all()):
        raise InputContractError(
            "must contain only finite values",
            operation_id="report.compute_sections",
            parameter=parameter,
        )
    return copied


def compute_sections(
    returns,
    benchmark_rets,
    positions,
    transactions,
    trades,
    rolling_window,
    period="daily",
    *,
    disclosure_context=None,
):
    """Compute all statistics and time series needed by the report renderers.

    Returns
    -------
    ReportModel
        A dict-compatible, structured model consumed by the HTML/PDF and
        other renderers.  Compute once here, render many times there.
    """
    returns = _validate_returns_input(returns, parameter="returns")
    benchmark_rets = (
        _validate_returns_input(benchmark_rets, parameter="benchmark_rets") if benchmark_rets is not None else None
    )
    positions = positions.copy(deep=True) if positions is not None else None
    transactions = transactions.copy(deep=True) if transactions is not None else None
    gross_leverage = gross_lev(positions) if positions is not None else None
    turnover = None
    if positions is not None and transactions is not None:
        try:
            turnover = get_turnover(positions, transactions)
        except (ValueError, TypeError, KeyError, ZeroDivisionError) as error:
            logger.warning("Failed to calculate turnover from transactions: %s", error)

    sections = {}

    # ------ Basics ------
    sections["date_range"] = (
        returns.index[0].strftime("%Y-%m-%d"),
        returns.index[-1].strftime("%Y-%m-%d"),
    )
    sections["period"] = period
    sections["period_title"] = _period_title(period)
    sections["period_unit"] = _period_unit(period)
    sections["period_unit_plural"] = _period_unit_plural(period)
    sections["n_periods"] = len(returns)
    sections["n_days"] = len(returns)
    sections["n_months"] = _approx_months(len(returns), period)
    sections["performance_disclosure"] = _disclosure_payload(
        _resolved_performance_disclosure(returns, benchmark_rets, period, disclosure_context)
    )

    # ------ Core performance ------
    perf = _compute_core_perf(
        returns,
        benchmark_rets,
        period,
        positions=positions,
        transactions=transactions,
        gross_leverage=gross_leverage,
        turnover=turnover,
    )
    sections["perf_stats"] = perf

    # ------ Extended stats ------
    sections["extended_stats"] = _compute_extended_stats(returns, period)

    # ------ Time series ------
    sections.update(_compute_time_series(returns, rolling_window, period))

    # ------ Benchmark ------
    if benchmark_rets is not None:
        sections.update(_compute_benchmark(returns, benchmark_rets, perf, rolling_window, period))

    # ------ Positions ------
    if positions is not None:
        sections.update(_compute_positions(positions, gross_leverage))

    # ------ Transactions ------
    if transactions is not None:
        sections.update(_compute_transactions(transactions, positions, turnover if positions is not None else None))

    # ------ Trades ------
    if trades is not None and len(trades) > 0:
        sections.update(_compute_trades(trades))

    # ------ Period returns ------
    sections.update(_compute_period_returns(returns, benchmark_rets, period))

    # ------ Summary text ------
    sections["summary_text"] = _compute_summary_text(perf, benchmark_rets)

    return ReportModel(sections)
