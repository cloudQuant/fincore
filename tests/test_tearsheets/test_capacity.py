import pandas as pd

from fincore.tearsheets.capacity import plot_capacity_sweep, plot_cones


class _FakeEmpyrical:
    def daily_txns_with_bar_data(self, transactions, market_data):
        return pd.DataFrame({"dummy": [1]})

    def apply_slippage_penalty(self, returns, txn_daily_w_bar, start_pv, bt_starting_capital):
        return returns

    def sharpe_ratio(self, _returns):
        return 0.5


def test_plot_capacity_sweep_runs_and_returns_axes() -> None:
    emp = _FakeEmpyrical()
    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx)
    transactions = pd.DataFrame({"amount": [1], "price": [10.0], "symbol": ["AAA"]}, index=[idx[0]])
    market_data = {
        "price": pd.DataFrame({"AAA": [10.0]}, index=[idx[0]]),
        "volume": pd.DataFrame({"AAA": [100]}, index=[idx[0]]),
    }

    ax = plot_capacity_sweep(
        emp,
        returns,
        transactions,
        market_data,
        bt_starting_capital=1000.0,
        min_pv=1,
        max_pv=4,
        step_size=1,
    )
    assert ax is not None


class _FakeEmpyricalCones:
    def cum_returns(self, returns, starting_value=1.0):
        if isinstance(returns, pd.DataFrame):
            series = returns.iloc[:, 0]
        else:
            series = returns
        return (1 + series).cumprod() * starting_value


def test_plot_cones_runs_and_returns_figure_when_no_ax() -> None:
    emp = _FakeEmpyricalCones()
    idx = pd.date_range("2024-01-01", periods=12, freq="B", tz="UTC")

    # Small negative drift to make a cone crossing likely when the lower bound is above 1.
    oos_returns = pd.Series([-0.002] * len(idx), index=idx, name="oos")

    bounds = pd.DataFrame(
        {
            1.0: 1.05,
            1.5: 1.075,
            -1.0: 0.95,
            -1.5: 0.925,
            2.0: 1.10,
            -2.0: 1.02,  # Higher than initial cum-return (~1.0) to trigger a crossing.
        },
        index=idx,
    )

    fig = plot_cones(emp, name="cone", bounds=bounds, oos_returns=oos_returns, num_strikes=1)
    assert fig is not None


def test_plot_cones_runs_and_returns_axes_when_ax_provided() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emp = _FakeEmpyricalCones()
    idx = pd.date_range("2024-01-01", periods=8, freq="B", tz="UTC")
    oos_returns = pd.Series([0.001, -0.001] * (len(idx) // 2), index=idx, name="oos")
    bounds = pd.DataFrame(
        {1.0: 1.05, 1.5: 1.075, 2.0: 1.1, -1.0: 0.95, -1.5: 0.925, -2.0: 0.9},
        index=idx,
    )

    _, ax = plt.subplots(figsize=(6, 4))
    out_ax = plot_cones(emp, name=None, bounds=bounds, oos_returns=oos_returns, ax=ax, num_strikes=0)
    assert out_ax is ax
    plt.close("all")


def test_plot_capacity_sweep_breaks_on_negative_sharpe_line_63() -> None:
    """Test plot_capacity_sweep breaks early when sharpe < -1 (line 63)."""

    class _FakeEmpyricalNegativeSharpe:
        def daily_txns_with_bar_data(self, transactions, market_data):
            return pd.DataFrame({"dummy": [1]})

        def apply_slippage_penalty(self, returns, txn_daily_w_bar, start_pv, bt_starting_capital):
            # Returns more negative as capital base increases
            return returns * (1 - start_pv / 100000)

        def sharpe_ratio(self, _returns):
            # Return very negative sharpe after certain capital base
            return -2.0

    emp = _FakeEmpyricalNegativeSharpe()
    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx)
    transactions = pd.DataFrame({"amount": [1], "price": [10.0], "symbol": ["AAA"]}, index=[idx[0]])
    market_data = {
        "price": pd.DataFrame({"AAA": [10.0]}, index=[idx[0]]),
        "volume": pd.DataFrame({"AAA": [100]}, index=[idx[0]]),
    }

    ax = plot_capacity_sweep(
        emp,
        returns,
        transactions,
        market_data,
        bt_starting_capital=1000.0,
        min_pv=100000,
        max_pv=500000,
        step_size=100000,
    )
    assert ax is not None


def test_plot_cones_breaks_when_no_crossing_line_143() -> None:
    """Test plot_cones breaks when crossing.sum() <= 0 (line 143)."""
    import matplotlib

    matplotlib.use("Agg")

    class _FakeEmpyricalConesNoCrossing:
        def cum_returns(self, returns, starting_value=1.0):
            if isinstance(returns, pd.DataFrame):
                series = returns.iloc[:, 0]
            else:
                series = returns
            # Always above bounds, so no crossing
            return (1 + series).cumprod() * starting_value * 1.5

    emp = _FakeEmpyricalConesNoCrossing()
    idx = pd.date_range("2024-01-01", periods=8, freq="B", tz="UTC")
    oos_returns = pd.Series([0.001] * len(idx), index=idx, name="oos")

    # Bounds that are below returns (no crossing)
    bounds = pd.DataFrame(
        {1.0: 0.95, 1.5: 0.925, 2.0: 0.9, -1.0: 0.85, -1.5: 0.825, -2.0: 0.8},
        index=idx,
    )

    fig = plot_cones(emp, name="cone", bounds=bounds, oos_returns=oos_returns, num_strikes=2)
    assert fig is not None
