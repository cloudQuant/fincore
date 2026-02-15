import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import fincore.tearsheets.sheets as sheets


class _FakePyfolioFull:
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


def test_create_full_tear_sheet_delegates_through_optional_sections(monkeypatch) -> None:
    # Avoid depending on the real intraday/position heuristics.
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])

    idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
    returns = pd.Series(np.linspace(0.001, -0.001, len(idx)), index=idx, name="r")
    positions = pd.DataFrame({"AAA": [10, 12, 11, 13, 10], "cash": [100, 100, 100, 100, 100]}, index=idx)
    transactions = pd.DataFrame({"amount": [1], "price": [10.0], "symbol": ["AAA"]}, index=[idx[0]])
    market_data = {
        "price": pd.DataFrame({"AAA": [10.0]}, index=[idx[0]]),
        "volume": pd.DataFrame({"AAA": [100]}, index=[idx[0]]),
    }

    pyf = _FakePyfolioFull()
    sheets.create_full_tear_sheet(
        pyf,
        returns,
        positions=positions,
        transactions=transactions,
        market_data=market_data,
        slippage=0.01,
        round_trips=True,
        bayesian=True,
        style_factor_panel=object(),
        factor_returns=object(),
        factor_loadings=object(),
    )

    called = [name for name, _ in pyf.calls]
    assert "adjust_returns_for_slippage" in called
    assert "create_returns_tear_sheet" in called
    assert "create_interesting_times_tear_sheet" in called
    assert "create_position_tear_sheet" in called
    assert "create_txn_tear_sheet" in called
    assert "create_round_trip_tear_sheet" in called
    assert "create_capacity_tear_sheet" in called
    assert "create_risk_tear_sheet" in called
    assert "create_perf_attrib_tear_sheet" in called
    assert "create_bayesian_tear_sheet" in called
    assert pyf.unadjusted_returns is not None


class _FakePyfolioSimple:
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


def test_create_simple_tear_sheet_runs_without_real_pyfolio(monkeypatch) -> None:
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])

    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    returns = pd.Series([0.001, -0.001, 0.002, -0.002, 0.001, 0.0], index=idx, name="r")
    benchmark = returns * 0.5
    positions = pd.DataFrame({"AAA": [10, 10, 11, 11, 12, 12], "cash": [100, 100, 100, 100, 100, 100]}, index=idx)
    transactions = pd.DataFrame({"amount": [1], "price": [10.0], "symbol": ["AAA"]}, index=[idx[1]])

    pyf = _FakePyfolioSimple()
    sheets.create_simple_tear_sheet(
        pyf,
        returns,
        positions=positions,
        transactions=transactions,
        benchmark_rets=benchmark,
        slippage=0.01,
    )

    assert "show_perf_stats" in pyf.calls
    assert "plot_rolling_returns" in pyf.calls
    assert "plot_rolling_beta" in pyf.calls
    assert "plot_rolling_sharpe" in pyf.calls
    assert "plot_drawdown_underwater" in pyf.calls
    assert "get_percent_alloc" in pyf.calls
    assert "plot_turnover" in pyf.calls
    assert "plot_txn_time_hist" in pyf.calls
    plt.close("all")


class _FakePyfolioInterestingTimes:
    def extract_interesting_date_ranges(self, returns):
        return {}


def test_create_interesting_times_tear_sheet_warns_and_returns_when_no_overlap(monkeypatch) -> None:
    monkeypatch.setattr(sheets, "print_table", lambda *_args, **_kwargs: None)
    pyf = _FakePyfolioInterestingTimes()
    idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx, name="r")
    with pytest.warns(UserWarning):
        sheets.create_interesting_times_tear_sheet(pyf, returns)


class _FakePyfolioCapacity:
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


def test_create_capacity_tear_sheet_runs_with_stubbed_pyfolio(monkeypatch) -> None:
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])
    monkeypatch.setattr(sheets, "print_table", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sheets, "format_asset", lambda x: str(x))

    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    returns = pd.Series([0.001, -0.001, 0.002, -0.002, 0.001, 0.0], index=idx, name="r")
    positions = pd.DataFrame(
        {"AAA": [100, 100, 100, 100, 100, 100], "cash": [1_000, 1_000, 1_000, 1_000, 1_000, 1_000]}, index=idx
    )
    transactions = pd.DataFrame({"amount": [1], "price": [10.0], "symbol": ["AAA"]}, index=[idx[1]])
    market_data = {
        "price": pd.DataFrame({"AAA": [10.0]}, index=[idx[1]]),
        "volume": pd.DataFrame({"AAA": [100]}, index=[idx[1]]),
    }

    pyf = _FakePyfolioCapacity()
    sheets.create_capacity_tear_sheet(pyf, returns, positions, transactions, market_data)
    assert "get_max_days_to_liquidate_by_ticker" in pyf.calls
    assert "get_low_liquidity_transactions" in pyf.calls
    assert "plot_capacity_sweep" in pyf.calls
    plt.close("all")


def test_create_bayesian_tear_sheet_requires_live_start_date() -> None:
    with pytest.raises(NotImplementedError):
        sheets.create_bayesian_tear_sheet(object(), pd.Series(dtype=float))


class _FakePyfolioReturns:
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


def test_create_returns_tear_sheet_smoke_with_benchmark_live_and_bootstrap(monkeypatch) -> None:
    monkeypatch.setattr(sheets, "clip_returns_to_benchmark", lambda r, b: r.loc[b.index])

    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    returns = pd.Series(np.linspace(0.001, -0.001, len(idx)), index=idx, name="r")
    benchmark = returns.loc[idx[2:]] * 0.5

    pyf = _FakePyfolioReturns()
    fig = sheets.create_returns_tear_sheet(
        pyf,
        returns,
        benchmark_rets=benchmark,
        live_start_date="2024-01-10",
        bootstrap=True,
        run_flask_app=True,
    )
    assert fig is not None
    assert "show_perf_stats" in pyf.calls
    assert "show_worst_drawdown_periods" in pyf.calls
    assert pyf.calls.count("plot_rolling_returns") == 3
    assert "plot_rolling_beta" in pyf.calls
    assert "plot_perf_stats" in pyf.calls
    plt.close("all")


def test_create_returns_tear_sheet_bootstrap_requires_benchmark() -> None:
    pyf = _FakePyfolioReturns()
    idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx, name="r")
    with pytest.raises(ValueError, match="bootstrap requires"):
        sheets.create_returns_tear_sheet(pyf, returns, benchmark_rets=None, bootstrap=True)


class _FakePyfolioPositions:
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
        return pd.DataFrame({"tech": [10.0] * len(idx), "fin": [5.0] * len(idx), "cash": [100.0] * len(idx)}, index=idx)

    def plot_sector_allocations(self, *args, **kwargs):  # noqa: ARG002
        self.calls.append(("plot_sector_allocations", dict(kwargs)))


def test_create_position_tear_sheet_smoke_with_sector_mappings_and_hide_positions(monkeypatch) -> None:
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])

    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx, name="r")
    positions = pd.DataFrame({"AAA": [10, 10, 0, 0, 5, 0], "cash": [100] * len(idx)}, index=idx)

    pyf = _FakePyfolioPositions()
    fig = sheets.create_position_tear_sheet(
        pyf,
        returns,
        positions,
        hide_positions=True,
        sector_mappings={"AAA": "tech"},
        run_flask_app=True,
    )
    assert fig is not None
    top_calls = [c for c, _ in pyf.calls if c == "show_and_plot_top_positions"]
    assert top_calls
    # hide_positions forces show_and_plot_top_pos=0
    top_kwargs = [kw for c, kw in pyf.calls if c == "show_and_plot_top_positions"][0]
    assert top_kwargs.get("show_and_plot") == 0
    assert any(c == "plot_sector_allocations" for c, _ in pyf.calls)
    plt.close("all")


class _FakePyfolioTxns:
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


def test_create_txn_tear_sheet_smoke_warns_on_turnover_hist_failure(monkeypatch) -> None:
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])

    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx, name="r")
    positions = pd.DataFrame({"AAA": [10] * len(idx), "cash": [100] * len(idx)}, index=idx)
    txns = pd.DataFrame({"amount": [1], "price": [10.0], "symbol": ["AAA"]}, index=[idx[0]])
    unadj = returns.copy()

    pyf = _FakePyfolioTxns()
    with pytest.warns(UserWarning, match="Unable to generate turnover plot"):
        fig = sheets.create_txn_tear_sheet(
            pyf,
            returns,
            positions,
            txns,
            unadjusted_returns=unadj,
            run_flask_app=True,
        )
    assert fig is not None
    assert "plot_turnover" in pyf.calls
    assert "plot_daily_volume" in pyf.calls
    assert "plot_txn_time_hist" in pyf.calls
    assert "plot_slippage_sweep" in pyf.calls
    assert "plot_slippage_sensitivity" in pyf.calls
    plt.close("all")


class _FakePyfolioRoundTrips:
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


def test_create_round_trip_tear_sheet_warns_and_returns_when_too_few_trades(monkeypatch) -> None:
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])
    from fincore.empyrical import Empyrical

    monkeypatch.setattr(Empyrical, "add_closing_transactions", staticmethod(lambda _p, t: t))
    monkeypatch.setattr(
        Empyrical, "extract_round_trips", staticmethod(lambda *_a, **_k: pd.DataFrame({"pnl": [1, 2, 3]}))
    )

    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx, name="r")
    positions = pd.DataFrame({"AAA": [10] * len(idx), "cash": [100] * len(idx)}, index=idx)
    txns = pd.DataFrame({"amount": [1], "price": [10.0], "symbol": ["AAA"]}, index=[idx[0]])

    pyf = _FakePyfolioRoundTrips()
    with pytest.warns(UserWarning, match="Fewer than 5 round-trip"):
        out = sheets.create_round_trip_tear_sheet(pyf, returns, positions, txns)
    assert out is None


def test_create_round_trip_tear_sheet_smoke_with_sector_mappings(monkeypatch) -> None:
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])
    from fincore.empyrical import Empyrical

    monkeypatch.setattr(Empyrical, "add_closing_transactions", staticmethod(lambda _p, t: t))
    trades = pd.DataFrame(
        {
            "duration": pd.to_timedelta([1, 2, 3, 4, 5], unit="D"),
            "pnl": [1.0, -1.0, 2.0, -2.0, 0.5],
            "returns": pd.Series([0.01, -0.01, 0.02, -0.02, 0.005]),
        }
    )
    monkeypatch.setattr(Empyrical, "extract_round_trips", staticmethod(lambda *_a, **_k: trades))
    monkeypatch.setattr(Empyrical, "apply_sector_mappings_to_round_trips", staticmethod(lambda t, _m: t))

    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx, name="r")
    positions = pd.DataFrame({"AAA": [10] * len(idx), "cash": [100] * len(idx)}, index=idx)
    txns = pd.DataFrame({"amount": [1], "price": [10.0], "symbol": ["AAA"]}, index=[idx[0]])

    pyf = _FakePyfolioRoundTrips()
    fig = sheets.create_round_trip_tear_sheet(
        pyf,
        returns,
        positions,
        txns,
        sector_mappings={"AAA": "tech"},
        run_flask_app=True,
    )
    assert fig is not None
    assert "print_round_trip_stats" in pyf.calls
    assert pyf.calls.count("show_profit_attribution") == 2
    assert "plot_round_trip_lifetimes" in pyf.calls
    assert "plot_prob_profit_trade" in pyf.calls
    plt.close("all")
