"""Tests for fincore.tearsheets.perf_attrib plotting and display."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.empyrical import Empyrical
from fincore.tearsheets import perf_attrib as tp


@pytest.fixture(autouse=True)
def _mpl_cleanup():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def test_perf_attrib_plot_helpers_smoke():
    import matplotlib.pyplot as plt

    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    total = pd.Series(np.random.RandomState(0).normal(0, 0.01, len(idx)), index=idx)
    common = pd.Series(np.random.RandomState(1).normal(0, 0.005, len(idx)), index=idx)
    specific = total - common
    cost = pd.Series(0.0001, index=idx)

    df = pd.DataFrame(
        {
            "total_returns": total,
            "common_returns": common,
            "specific_returns": specific,
            "FactorA": np.random.RandomState(2).normal(0, 0.001, len(idx)),
            "FactorB": np.random.RandomState(3).normal(0, 0.001, len(idx)),
        },
        index=idx,
    )

    emp = Empyrical(total)

    _, ax = plt.subplots()
    ax = tp.plot_perf_attrib_returns(emp, df, cost=cost, ax=ax)
    assert ax is not None

    _, ax2 = plt.subplots()
    ax2 = tp.plot_alpha_returns(specific, ax=ax2)
    assert ax2 is not None

    _, ax3 = plt.subplots()
    ax3 = tp.plot_factor_contribution_to_perf(emp, df, ax=ax3)
    assert ax3 is not None

    exposures = pd.DataFrame({"A": np.linspace(-1, 1, len(idx)), "B": np.linspace(0, 1, len(idx))}, index=idx)
    _, ax4 = plt.subplots()
    ax4 = tp.plot_risk_exposures(exposures, ax=ax4)
    assert ax4 is not None


def test_show_perf_attrib_stats_calls_print_table(monkeypatch):
    calls: list[tuple[str, pd.DataFrame | pd.Series | None]] = []

    def fake_print_table(table, name=None, **kwargs):
        calls.append((name, table))

    monkeypatch.setattr(tp, "print_table", fake_print_table)

    class DummyEmp:
        def perf_attrib(self, *args, **kwargs):
            idx = pd.date_range("2020-01-01", periods=3, freq="D")
            risk = pd.DataFrame({"f1": [0.1, 0.2, 0.3]}, index=idx)
            perf = pd.DataFrame(
                {
                    "total_returns": [0.01, 0.0, -0.01],
                    "common_returns": [0.005, 0.0, -0.005],
                    "specific_returns": [0.005, 0.0, -0.005],
                },
                index=idx,
            )
            return risk, perf

        def create_perf_attrib_stats(self, perf_attrib_data, risk_exposures):
            # Minimal tables containing required row labels.
            perf_stats = pd.Series(
                {
                    "Annualized Specific Return": 0.1,
                    "Annualized Common Return": 0.2,
                    "Annualized Total Return": 0.3,
                    "Specific Sharpe Ratio": 1.23,
                }
            )
            risk_stats = pd.DataFrame(
                {
                    "Average Risk Factor Exposure": [0.1],
                    "Annualized Return": [0.2],
                    "Cumulative Return": [0.3],
                },
                index=["f1"],
            )
            return perf_stats, risk_stats

    emp = DummyEmp()
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    returns = pd.Series([0.01, 0.0, -0.01], index=idx)
    positions = pd.DataFrame({"cash": [1, 1, 1]}, index=idx)
    factor_returns = pd.DataFrame({"f1": [0.0, 0.0, 0.0]}, index=idx)
    factor_loadings = pd.DataFrame({"f1": [1.0, 1.0, 1.0]}, index=pd.MultiIndex.from_product([idx, ["A"]]))

    tp.show_perf_attrib_stats(emp, returns, positions, factor_returns, factor_loadings)
    assert [c[0] for c in calls] == ["Summary Statistics", "Exposures Summary"]


def test_plot_functions_with_default_ax(monkeypatch):
    """Test plotting functions when ax=None (uses plt.gca())."""
    import matplotlib.pyplot as plt

    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    total = pd.Series(np.random.RandomState(0).normal(0, 0.01, len(idx)), index=idx)
    common = pd.Series(np.random.RandomState(1).normal(0, 0.005, len(idx)), index=idx)
    specific = total - common
    cost = pd.Series(0.0001, index=idx)

    df = pd.DataFrame(
        {
            "total_returns": total,
            "common_returns": common,
            "specific_returns": specific,
            "FactorA": np.random.RandomState(2).normal(0, 0.001, len(idx)),
            "FactorB": np.random.RandomState(3).normal(0, 0.001, len(idx)),
        },
        index=idx,
    )

    emp = Empyrical(total)

    # Test plot_perf_attrib_returns with ax=None (line 35)
    plt.figure()
    ax1 = tp.plot_perf_attrib_returns(emp, df, cost=cost, ax=None)
    assert ax1 is not None
    plt.close()

    # Test plot_alpha_returns with ax=None (line 78)
    plt.figure()
    ax2 = tp.plot_alpha_returns(specific, ax=None)
    assert ax2 is not None
    plt.close()

    # Test plot_factor_contribution_to_perf with ax=None (line 114)
    plt.figure()
    ax3 = tp.plot_factor_contribution_to_perf(emp, df, ax=None)
    assert ax3 is not None
    plt.close()

    # Test plot_risk_exposures with ax=None (line 152)
    exposures = pd.DataFrame({"A": np.linspace(-1, 1, len(idx)), "B": np.linspace(0, 1, len(idx))}, index=idx)
    plt.figure()
    ax4 = tp.plot_risk_exposures(exposures, ax=None)
    assert ax4 is not None
    plt.close()
