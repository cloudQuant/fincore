"""Compute the canonical portfolio report model from direct domain functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from fincore.exceptions import InputContractError
from fincore.metrics.alpha_beta import alpha_beta
from fincore.metrics.drawdown import gen_drawdown_table, max_drawdown
from fincore.metrics.frequencies import ANNUALIZATION_FACTORS
from fincore.metrics.ratios import (
    calmar_ratio,
    down_capture,
    information_ratio,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    stability_of_timeseries,
    up_capture,
)
from fincore.metrics.returns import aggregate_returns, cum_returns, cum_returns_final
from fincore.metrics.risk import annual_volatility, tail_ratio, tracking_error, value_at_risk
from fincore.metrics.rolling import rolling_beta, rolling_sharpe, rolling_volatility
from fincore.metrics.yearly import annual_return
from fincore.portfolio.positions import gross_lev
from fincore.portfolio.transactions import get_turnover
from fincore.report.models import ReportDocument, ReportSection

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = ["build_portfolio_report"]

_OPERATION_ID = "report.portfolio.build_portfolio_report"


def _input_error(message: str, parameter: str) -> InputContractError:
    return InputContractError(message, operation_id=_OPERATION_ID, parameter=parameter)


def _validated_returns(value: pd.Series, *, parameter: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise _input_error("must be a pandas Series", parameter)
    if value.empty:
        raise _input_error("must contain at least one return", parameter)
    if not isinstance(value.index, pd.DatetimeIndex):
        raise _input_error("must use a DatetimeIndex", parameter)
    if not value.index.is_unique or not value.index.is_monotonic_increasing:
        raise _input_error("index must be unique and increasing", parameter)
    try:
        copied = value.astype(float).copy(deep=True)
    except (TypeError, ValueError) as error:
        raise _input_error("must contain numeric values", parameter) from error
    if not bool(np.isfinite(copied.to_numpy()).all()):
        raise _input_error("must contain only finite values", parameter)
    return copied


def _validated_positions(value: pd.DataFrame, *, returns: pd.Series) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise _input_error("must be a pandas DataFrame", "positions")
    if not isinstance(value.index, pd.DatetimeIndex) or not value.index.is_unique:
        raise _input_error("must use a unique DatetimeIndex", "positions")
    if "cash" not in value.columns:
        raise _input_error("must contain a cash column", "positions")
    copied = value.astype(float).copy(deep=True)
    if not bool(np.isfinite(copied.to_numpy()).all()):
        raise _input_error("must contain only finite values", "positions")
    if not copied.index.equals(returns.index):
        copied = copied.reindex(returns.index)
        if copied.isna().any().any():
            raise _input_error("must cover every return timestamp", "positions")
    return copied


def _validated_transactions(value: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise _input_error("must be a pandas DataFrame", "transactions")
    if not isinstance(value.index, pd.DatetimeIndex):
        raise _input_error("must use a DatetimeIndex", "transactions")
    required = {"amount", "price"}
    if missing := sorted(required - set(value.columns)):
        raise _input_error(f"is missing required columns: {missing!r}", "transactions")
    copied = value.copy(deep=True)
    for column in required:
        copied[column] = pd.to_numeric(copied[column], errors="raise")
    if not bool(np.isfinite(copied.loc[:, list(required)].to_numpy(dtype=float)).all()):
        raise _input_error("must contain only finite amount and price values", "transactions")
    return copied


def _scalar(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def _safe_metric(function: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    """Convert mathematically undefined scalar metrics into reportable ``NaN``."""

    try:
        return _scalar(function(*args, **kwargs))
    except (FloatingPointError, ZeroDivisionError):
        return float("nan")


def _performance_metrics(returns: pd.Series, benchmark_returns: pd.Series | None, period: str) -> dict[str, float]:
    metrics = {
        "annual_return": _safe_metric(annual_return, returns, period=period),
        "cumulative_return": _safe_metric(cum_returns_final, returns),
        "annual_volatility": _safe_metric(annual_volatility, returns, period=period),
        "sharpe_ratio": _safe_metric(sharpe_ratio, returns, period=period),
        "calmar_ratio": _safe_metric(calmar_ratio, returns, period=period),
        "stability": _safe_metric(stability_of_timeseries, returns),
        "max_drawdown": _safe_metric(max_drawdown, returns),
        "omega_ratio": _safe_metric(omega_ratio, returns),
        "sortino_ratio": _safe_metric(sortino_ratio, returns, period=period),
        "tail_ratio": _safe_metric(tail_ratio, returns),
        "value_at_risk": _safe_metric(value_at_risk, returns),
    }
    if benchmark_returns is not None:
        try:
            alpha_value, beta_value = alpha_beta(returns, benchmark_returns, period=period)
        except (FloatingPointError, ZeroDivisionError):
            alpha_value, beta_value = float("nan"), float("nan")
        metrics["alpha"] = _scalar(alpha_value)
        metrics["beta"] = _scalar(beta_value)
    return metrics


def _benchmark_section(
    returns: pd.Series, benchmark_returns: pd.Series, *, period: str, rolling_window: int
) -> ReportSection:
    aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")
    if aligned_returns.empty:
        raise _input_error("must share at least one timestamp with returns", "benchmark_returns")
    up = _scalar(up_capture(aligned_returns, aligned_benchmark, period=period))
    down = _scalar(down_capture(aligned_returns, aligned_benchmark, period=period))
    metrics = {
        "information_ratio": _scalar(information_ratio(aligned_returns, aligned_benchmark, period=period)),
        "tracking_error": _scalar(tracking_error(aligned_returns, aligned_benchmark, period=period)),
        "up_capture": up,
        "down_capture": down,
        "capture_ratio": up / down if np.isfinite(down) and down != 0 else float("nan"),
    }
    return ReportSection(
        key="benchmark",
        title="Benchmark comparison",
        metrics=metrics,
        series={
            "benchmark_cumulative_returns": cum_returns(aligned_benchmark, starting_value=1.0),
            "rolling_beta": rolling_beta(aligned_returns, aligned_benchmark, rolling_window=rolling_window),
        },
        units={"benchmark_cumulative_returns": "growth_multiple", "rolling_beta": "beta"},
        legends={"benchmark_cumulative_returns": "Benchmark", "rolling_beta": "Strategy beta"},
    )


def _portfolio_section(
    positions: pd.DataFrame,
    transactions: pd.DataFrame | None,
) -> ReportSection:
    leverage = gross_lev(positions)
    non_cash = positions.drop(columns="cash")
    series: dict[str, pd.Series] = {"gross_leverage": leverage}
    metrics: dict[str, float | int] = {
        "asset_count": len(non_cash.columns),
        "average_gross_leverage": _scalar(leverage.mean()),
        "maximum_gross_leverage": _scalar(leverage.max()),
    }
    if transactions is not None:
        turnover = get_turnover(positions, transactions)
        series["turnover"] = turnover
        metrics["average_turnover"] = _scalar(turnover.mean())
    return ReportSection(
        key="portfolio",
        title="Portfolio exposure",
        metrics=metrics,
        series=series,
        units=dict.fromkeys(series, "ratio"),
        legends={"gross_leverage": "Gross leverage", **({"turnover": "Turnover"} if "turnover" in series else {})},
    )


def _transactions_section(transactions: pd.DataFrame) -> ReportSection:
    normalized = transactions.copy(deep=True)
    normalized.index = normalized.index.normalize()
    daily_count = normalized.groupby(level=0).size().astype(float)
    daily_value = (normalized["amount"].abs() * normalized["price"]).groupby(level=0).sum()
    metrics: dict[str, int | float] = {
        "transaction_count": len(transactions),
        "trading_days": len(daily_count),
        "average_daily_transactions": _scalar(daily_count.mean()),
        "average_daily_notional": _scalar(daily_value.mean()),
    }
    if "symbol" in transactions.columns:
        metrics["symbol_count"] = int(transactions["symbol"].nunique())
    return ReportSection(
        key="transactions",
        title="Transactions",
        metrics=metrics,
        series={"daily_transaction_count": daily_count, "daily_transaction_notional": daily_value},
        units={"daily_transaction_count": "count", "daily_transaction_notional": "notional"},
        legends={"daily_transaction_count": "Trades", "daily_transaction_notional": "Notional"},
    )


def build_portfolio_report(
    returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    positions: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    title: str = "Portfolio Report",
    period: str = "daily",
    rolling_window: int = 63,
    metadata: Mapping[str, Any] | None = None,
) -> ReportDocument:
    """Compute one canonical portfolio report without a facade or tear sheet.

    The returned document contains all financial calculations needed by the
    renderer.  It is therefore safe to render repeatedly with HTML, PDF,
    XLSX, Matplotlib, Plotly, or Bokeh without rerunning a metric kernel.
    """

    if period not in ANNUALIZATION_FACTORS:
        raise _input_error(f"must be one of {tuple(ANNUALIZATION_FACTORS)!r}", "period")
    if not isinstance(rolling_window, int) or isinstance(rolling_window, bool) or rolling_window < 1:
        raise _input_error("must be a positive integer", "rolling_window")
    validated_returns = _validated_returns(returns, parameter="returns")
    validated_benchmark = (
        _validated_returns(benchmark_returns, parameter="benchmark_returns") if benchmark_returns is not None else None
    )
    validated_positions = _validated_positions(positions, returns=validated_returns) if positions is not None else None
    validated_transactions = _validated_transactions(transactions) if transactions is not None else None

    cumulative = cum_returns(validated_returns, starting_value=1.0)
    drawdown_basis = cum_returns(validated_returns, starting_value=0.0)
    drawdown = (1.0 + drawdown_basis) / (1.0 + drawdown_basis).cummax() - 1.0
    sections: list[ReportSection] = [
        ReportSection(
            key="performance",
            title="Performance",
            metrics=_performance_metrics(validated_returns, validated_benchmark, period),
            tables={"drawdowns": gen_drawdown_table(validated_returns, top=5)},
            series={
                "returns": validated_returns,
                "cumulative_returns": cumulative,
                "drawdown": drawdown,
                "rolling_sharpe": rolling_sharpe(
                    validated_returns, rolling_sharpe_window=rolling_window, period=period
                ),
                "rolling_volatility": rolling_volatility(
                    validated_returns,
                    rolling_vol_window=rolling_window,
                    period=period,
                ),
                "monthly_returns": aggregate_returns(validated_returns, "monthly"),
            },
            units={
                "returns": "decimal_return",
                "cumulative_returns": "growth_multiple",
                "drawdown": "decimal_return",
                "rolling_sharpe": "ratio",
                "rolling_volatility": "annualized_decimal_return",
                "monthly_returns": "decimal_return",
            },
            legends={
                "returns": "Strategy return",
                "cumulative_returns": "Strategy",
                "drawdown": "Drawdown",
                "rolling_sharpe": "Rolling Sharpe",
                "rolling_volatility": "Rolling volatility",
                "monthly_returns": "Monthly return",
            },
        )
    ]
    if validated_benchmark is not None:
        sections.append(
            _benchmark_section(
                validated_returns,
                validated_benchmark,
                period=period,
                rolling_window=rolling_window,
            )
        )
    if validated_positions is not None:
        sections.append(_portfolio_section(validated_positions, validated_transactions))
    if validated_transactions is not None:
        sections.append(_transactions_section(validated_transactions))
    return ReportDocument(
        domain="portfolio",
        title=title,
        sections=tuple(sections),
        metadata={
            "period": period,
            "rolling_window": rolling_window,
            "observations": len(validated_returns),
            **dict(metadata or {}),
        },
    )
