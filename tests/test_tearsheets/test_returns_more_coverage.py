import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import fincore.tearsheets.returns as ret_ts


class _DummyEmpyrical:
    def aggregate_returns(self, returns: pd.Series, period: str) -> pd.Series:
        def agg(x: pd.Series) -> float:
            return float((1 + x.fillna(0)).prod() - 1)

        if period == "monthly":
            m = returns.resample(ret_ts.get_month_end_freq()).apply(agg)
            mi = pd.MultiIndex.from_arrays([m.index.year, m.index.month], names=["year", "month"])
            return pd.Series(m.values, index=mi)
        if period == "yearly":
            y = returns.resample("YE").apply(agg)
            return pd.Series(y.values, index=y.index.year)
        if period == "weekly":
            w = returns.resample("W").apply(agg)
            return w
        raise ValueError(period)

    def cum_returns(self, returns: pd.Series, starting_value: float = 1.0) -> pd.Series:
        return starting_value * (1 + returns.fillna(0)).cumprod()

    def rolling_beta(self, returns: pd.Series, factor_returns: pd.Series, rolling_window: int) -> pd.Series:  # noqa: ARG002
        return pd.Series(0.5, index=returns.index)

    def rolling_volatility(self, returns: pd.Series, rolling_window: int) -> pd.Series:  # noqa: ARG002
        return pd.Series(0.1, index=returns.index)

    def rolling_sharpe(self, returns: pd.Series, rolling_window: int) -> pd.Series:  # noqa: ARG002
        return pd.Series(1.0, index=returns.index)

    def gen_drawdown_table(self, returns: pd.Series, top: int = 10) -> pd.DataFrame:  # noqa: ARG002
        return pd.DataFrame(
            {"Peak date": [returns.index[0]], "Recovery date": [pd.NaT]},
            index=range(1),
        )

    def perf_stats(self, *_args, **_kwargs) -> pd.Series:
        return pd.Series(
            {
                "Annual return": 0.10,
                "Cumulative returns": 0.20,
                "Annual volatility": 0.30,
                "Max drawdown": -0.10,
                "Daily turnover": np.nan,
                "Sharpe ratio": 1.5,
            }
        )

    def perf_stats_bootstrap(self, *_args, **_kwargs) -> pd.DataFrame:
        # Needs "Kurtosis" so plot_perf_stats can drop it.
        return pd.DataFrame(
            {
                "Annual return": [0.10, 0.11, 0.09],
                "Kurtosis": [1.0, 1.2, 0.8],
            }
        )


def _make_returns(start: str = "2023-10-02", periods: int = 120) -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq="B", tz="UTC")
    vals = np.sin(np.linspace(0, 6, len(idx))) * 0.01
    return pd.Series(vals, index=idx, name="r")


def test_returns_plots_cover_ax_none_and_live_start_date_str(monkeypatch) -> None:
    emp = _DummyEmpyrical()
    returns = _make_returns()

    # Make seaborn plotting fast and non-interactive.
    monkeypatch.setattr(ret_ts.sns, "heatmap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ret_ts.sns, "boxplot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ret_ts.sns, "swarmplot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ret_ts.sns, "barplot", lambda *_args, **_kwargs: None)

    # Cover ax=None branches.
    assert ret_ts.plot_monthly_returns_heatmap(emp, returns) is not None
    assert ret_ts.plot_annual_returns(emp, returns) is not None
    assert ret_ts.plot_monthly_returns_dist(emp, returns) is not None

    assert ret_ts.plot_returns(returns, live_start_date=str(returns.index[60].date())) is not None

    # plot_rolling_returns: exercise factor plot, live split, and cone shading.
    factor = returns * 0.5
    factor.name = "SPY"

    def fake_cone(is_returns, horizon, cone_std, starting_value):  # noqa: ARG001
        data = {}
        for std in cone_std:
            data[float(std)] = np.full(horizon, starting_value * 1.1)
            data[float(-std)] = np.full(horizon, starting_value * 0.9)
        return pd.DataFrame(data)

    monkeypatch.setattr(ret_ts.Empyrical, "forecast_cone_bootstrap", staticmethod(fake_cone))
    ax = ret_ts.plot_rolling_returns(
        emp,
        returns,
        factor_returns=factor,
        live_start_date=str(returns.index[60].date()),
        cone_std=1.0,
        cone_function=None,  # cover default assignment branch
    )
    assert ax is not None

    with pytest.raises(ValueError, match="volatility_match requires"):
        ret_ts.plot_rolling_returns(emp, returns, factor_returns=None, volatility_match=True)

    assert ret_ts.plot_rolling_beta(emp, returns, factor) is not None
    assert ret_ts.plot_rolling_volatility(emp, returns, factor_returns=None) is not None
    assert ret_ts.plot_rolling_sharpe(emp, returns, factor_returns=factor) is not None

    assert ret_ts.plot_drawdown_periods(emp, returns, top=1) is not None
    assert ret_ts.plot_drawdown_underwater(emp, returns) is not None

    assert ret_ts.plot_return_quantiles(emp, returns, live_start_date=str(returns.index[60].date())) is not None
    assert ret_ts.plot_monthly_returns_timeseries(emp, returns) is not None
    assert ret_ts.plot_perf_stats(emp, returns, factor) is not None
    plt.close("all")


def test_show_perf_stats_live_split_and_header_rows(monkeypatch) -> None:
    emp = _DummyEmpyrical()
    returns = _make_returns()

    idx = returns.index
    positions = pd.DataFrame({"AAA": 1.0, "cash": 0.0}, index=idx)
    transactions = pd.DataFrame({"amount": [1], "price": [1.0], "symbol": ["AAA"]}, index=[idx[10]])

    captured = {}

    def fake_print_table(df, **kwargs):
        captured["df"] = df
        captured["kwargs"] = kwargs

    monkeypatch.setattr(ret_ts, "print_table", fake_print_table)

    ret_ts.show_perf_stats(
        emp,
        returns,
        factor_returns=returns * 0.5,
        positions=positions,
        transactions=transactions,
        live_start_date=str(idx[60].date()),
        bootstrap=False,
        header_rows={"User header": "X"},
    )

    assert "df" in captured
    # Ensure header rows merged path was taken.
    hdr = captured["kwargs"]["header_rows"]
    assert "User header" in hdr
    assert "Start date" in hdr
    assert "End date" in hdr
