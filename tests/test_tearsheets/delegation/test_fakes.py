"""Fake pyfolio objects for testing tearsheet delegation.

Split from test_sheets_delegation.py for maintainability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class _FakePyfolioFull:
    """Fake pyfolio with all methods for full tear sheet testing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.unadjusted_returns = None

    def adjust_returns_for_slippage(self, returns, positions, transactions, slippage):
        self.calls.append(("adjust_returns_for_slippage", {"slippage": slippage}))
        return returns * 0.0

    def create_returns_tear_sheet(self, returns, **kwargs):
        self.calls.append(("create_returns_tear_sheet", dict(kwargs)))

    def create_interesting_times_tear_sheet(self, returns, **kwargs):
        self.calls.append(("create_interesting_times_tear_sheet", dict(kwargs)))

    def create_position_tear_sheet(self, returns, positions, **kwargs):
        self.calls.append(("create_position_tear_sheet", dict(kwargs)))

    def create_txn_tear_sheet(self, returns, positions, transactions, **kwargs):
        self.unadjusted_returns = kwargs.get("unadjusted_returns")
        self.calls.append(("create_txn_tear_sheet", dict(kwargs)))

    def create_round_trip_tear_sheet(self, **kwargs):
        self.calls.append(("create_round_trip_tear_sheet", dict(kwargs)))

    def create_capacity_tear_sheet(self, *args, **kwargs):
        self.calls.append(("create_capacity_tear_sheet", dict(kwargs)))

    def create_risk_tear_sheet(self, *args, **kwargs):
        self.calls.append(("create_risk_tear_sheet", dict(kwargs)))

    def create_perf_attrib_tear_sheet(self, *args, **kwargs):
        self.calls.append(("create_perf_attrib_tear_sheet", dict(kwargs)))

    def create_bayesian_tear_sheet(self, *args, **kwargs):
        self.calls.append(("create_bayesian_tear_sheet", dict(kwargs)))


class _FakePyfolioSimple:
    """Fake pyfolio with methods for simple tear sheet testing."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def adjust_returns_for_slippage(self, returns, positions, transactions, slippage):
        self.calls.append("adjust_returns_for_slippage")
        return returns

    def show_perf_stats(self, *args, **kwargs):
        self.calls.append("show_perf_stats")

    def plot_rolling_returns(self, *args, **kwargs):
        self.calls.append("plot_rolling_returns")

    def plot_rolling_beta(self, *args, **kwargs):
        self.calls.append("plot_rolling_beta")

    def plot_rolling_sharpe(self, *args, **kwargs):
        self.calls.append("plot_rolling_sharpe")

    def plot_drawdown_underwater(self, *args, **kwargs):
        self.calls.append("plot_drawdown_underwater")

    def get_percent_alloc(self, positions):
        self.calls.append("get_percent_alloc")
        denom = positions.abs().sum(axis=1).replace(0, np.nan)
        return positions.div(denom, axis=0).fillna(0)

    def plot_exposures(self, *args, **kwargs):
        self.calls.append("plot_exposures")

    def show_and_plot_top_positions(self, *args, **kwargs):
        self.calls.append("show_and_plot_top_positions")

    def plot_holdings(self, *args, **kwargs):
        self.calls.append("plot_holdings")

    def plot_long_short_holdings(self, *args, **kwargs):
        self.calls.append("plot_long_short_holdings")

    def plot_turnover(self, *args, **kwargs):
        self.calls.append("plot_turnover")

    def plot_txn_time_hist(self, *args, **kwargs):
        self.calls.append("plot_txn_time_hist")


class _FakePyfolioInterestingTimes:
    """Fake pyfolio for interesting times testing."""

    def extract_interesting_date_ranges(self, returns):
        return {}


class _FakePyfolioCapacity:
    """Fake pyfolio for capacity tear sheet testing."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_max_days_to_liquidate_by_ticker(self, *args, **kwargs):
        self.calls.append("get_max_days_to_liquidate_by_ticker")
        return pd.DataFrame({"days_to_liquidate": [0.5, 2.0]}, index=["AAA", "BBB"])

    def get_low_liquidity_transactions(self, *args, **kwargs):
        self.calls.append("get_low_liquidity_transactions")
        return pd.DataFrame({"max_pct_bar_consumed": [1.0, 10.0]}, index=["AAA", "BBB"])

    def plot_capacity_sweep(self, *args, **kwargs):
        self.calls.append("plot_capacity_sweep")


class _FakePyfolioReturns:
    """Fake pyfolio for returns tear sheet testing."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def show_perf_stats(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("show_perf_stats")

    def show_worst_drawdown_periods(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("show_worst_drawdown_periods")

    def plot_rolling_returns(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_rolling_returns")

    def plot_returns(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_returns")

    def plot_rolling_beta(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_rolling_beta")

    def plot_rolling_volatility(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_rolling_volatility")

    def plot_rolling_sharpe(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_rolling_sharpe")

    def plot_drawdown_periods(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_drawdown_periods")

    def plot_drawdown_underwater(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_drawdown_underwater")

    def plot_monthly_returns_heatmap(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_monthly_returns_heatmap")

    def plot_annual_returns(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_annual_returns")

    def plot_monthly_returns_dist(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_monthly_returns_dist")

    def plot_return_quantiles(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_return_quantiles")

    def plot_perf_stats(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_perf_stats")


class _FakePyfolioPositions:
    """Fake pyfolio for positions tear sheet testing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get_percent_alloc(self, positions):
        self.calls.append(("get_percent_alloc", {}))
        denom = positions.abs().sum(axis=1).replace(0, np.nan)
        return positions.div(denom, axis=0).fillna(0)

    def plot_exposures(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append(("plot_exposures", dict(kwargs)))

    def show_and_plot_top_positions(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append(("show_and_plot_top_positions", dict(kwargs)))

    def plot_max_median_position_concentration(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append(("plot_max_median_position_concentration", dict(kwargs)))

    def plot_holdings(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append(("plot_holdings", dict(kwargs)))

    def plot_long_short_holdings(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append(("plot_long_short_holdings", dict(kwargs)))

    def plot_gross_leverage(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append(("plot_gross_leverage", dict(kwargs)))

    def get_sector_exposures(self, positions, sector_mappings):  # noqa: ARG002
        idx = positions.index
        return pd.DataFrame(
            {"tech": [10.0] * len(idx), "fin": [5.0] * len(idx), "cash": [100.0] * len(idx)}, index=idx
        )

    def plot_sector_allocations(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append(("plot_sector_allocations", dict(kwargs)))


class _FakePyfolioTxns:
    """Fake pyfolio for transactions tear sheet testing."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def plot_turnover(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_turnover")

    def plot_daily_volume(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_daily_volume")

    def plot_daily_turnover_hist(self, *args, **kwargs):  # noqa: ARG002
        raise ValueError("nope")

    def plot_txn_time_hist(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_txn_time_hist")

    def plot_slippage_sweep(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_slippage_sweep")

    def plot_slippage_sensitivity(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_slippage_sensitivity")


class _FakePyfolioRoundTrips:
    """Fake pyfolio for round-trip tear sheet testing."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def print_round_trip_stats(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("print_round_trip_stats")

    def show_profit_attribution(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("show_profit_attribution")

    def plot_round_trip_lifetimes(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_round_trip_lifetimes")

    def plot_prob_profit_trade(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append("plot_prob_profit_trade")
