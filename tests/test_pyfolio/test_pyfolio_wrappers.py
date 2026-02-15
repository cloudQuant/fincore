import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fincore.pyfolio as pyfolio_mod
from fincore.pyfolio import Pyfolio


def test_pyfolio_fallback_display_and_markdown() -> None:
    # Cover simple fallback helpers.
    pyfolio_mod._fallback_display("x")
    assert pyfolio_mod._fallback_markdown("md") == "md"


def test_pyfolio_wrapper_methods_delegate_to_module_functions(monkeypatch) -> None:
    pyf = Pyfolio()
    sentinel = object()

    def make_stub(name: str):
        def _stub(*_args, **_kwargs):
            return (name, sentinel)

        return _stub

    # Bayesian/risk plotting wrappers.
    monkeypatch.setattr(pyfolio_mod, "_plot_best", make_stub("plot_best"))
    monkeypatch.setattr(pyfolio_mod, "_plot_stoch_vol", make_stub("plot_stoch_vol"))
    monkeypatch.setattr(pyfolio_mod, "_plot_bayes_cone_internal", make_stub("_plot_bayes_cone"))
    monkeypatch.setattr(pyfolio_mod, "_plot_bayes_cone", make_stub("plot_bayes_cone"))
    monkeypatch.setattr(pyfolio_mod, "_plot_style_factor_exposures", make_stub("plot_style_factor_exposures"))
    monkeypatch.setattr(pyfolio_mod, "_plot_sector_exposures_longshort", make_stub("plot_sector_exposures_longshort"))
    monkeypatch.setattr(pyfolio_mod, "_plot_sector_exposures_gross", make_stub("plot_sector_exposures_gross"))
    monkeypatch.setattr(pyfolio_mod, "_plot_sector_exposures_net", make_stub("plot_sector_exposures_net"))
    monkeypatch.setattr(pyfolio_mod, "_plot_cap_exposures_longshort", make_stub("plot_cap_exposures_longshort"))
    monkeypatch.setattr(pyfolio_mod, "_plot_cap_exposures_gross", make_stub("plot_cap_exposures_gross"))
    monkeypatch.setattr(pyfolio_mod, "_plot_cap_exposures_net", make_stub("plot_cap_exposures_net"))
    monkeypatch.setattr(pyfolio_mod, "_plot_volume_exposures_longshort", make_stub("plot_volume_exposures_longshort"))
    monkeypatch.setattr(pyfolio_mod, "_plot_volume_exposures_gross", make_stub("plot_volume_exposures_gross"))

    assert pyf.plot_best()[0] == "plot_best"
    assert pyf.plot_stoch_vol(data=pd.Series([0.0]))[0] == "plot_stoch_vol"
    assert pyf._plot_bayes_cone(pd.Series([0.0]), pd.Series([0.0]), preds=np.zeros((1, 1)))[0] == "_plot_bayes_cone"
    assert pyf.plot_bayes_cone(pd.Series([0.0]), pd.Series([0.0]), ppc=np.zeros((1, 1)))[0] == "plot_bayes_cone"
    assert pyf.plot_style_factor_exposures(pd.DataFrame())[0] == "plot_style_factor_exposures"
    assert pyf.plot_sector_exposures_longshort(pd.DataFrame(), pd.DataFrame())[0] == "plot_sector_exposures_longshort"
    assert pyf.plot_sector_exposures_gross(pd.DataFrame())[0] == "plot_sector_exposures_gross"
    assert pyf.plot_sector_exposures_net(pd.DataFrame())[0] == "plot_sector_exposures_net"
    assert pyf.plot_cap_exposures_longshort(pd.DataFrame(), pd.DataFrame())[0] == "plot_cap_exposures_longshort"
    assert pyf.plot_cap_exposures_gross(pd.DataFrame())[0] == "plot_cap_exposures_gross"
    assert pyf.plot_cap_exposures_net(pd.DataFrame())[0] == "plot_cap_exposures_net"
    assert (
        pyf.plot_volume_exposures_longshort(pd.DataFrame(), pd.DataFrame(), 0.1)[0] == "plot_volume_exposures_longshort"
    )
    assert pyf.plot_volume_exposures_gross(pd.DataFrame(), 0.1)[0] == "plot_volume_exposures_gross"

    # Tear sheet wrappers.
    monkeypatch.setattr(pyfolio_mod, "_create_capacity_tear_sheet", make_stub("create_capacity_tear_sheet"))
    monkeypatch.setattr(pyfolio_mod, "_create_bayesian_tear_sheet", make_stub("create_bayesian_tear_sheet"))
    monkeypatch.setattr(pyfolio_mod, "_create_risk_tear_sheet", make_stub("create_risk_tear_sheet"))
    monkeypatch.setattr(pyfolio_mod, "_create_perf_attrib_tear_sheet", make_stub("create_perf_attrib_tear_sheet"))

    idx = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
    r = pd.Series([0.0, 0.0, 0.0], index=idx)
    pos = pd.DataFrame({"AAA": 1.0, "cash": 0.0}, index=idx)
    txns = pd.DataFrame({"amount": [1], "price": [1.0], "symbol": ["AAA"]}, index=[idx[0]])
    md = {"price": pd.DataFrame({"AAA": [1.0]}, index=[idx[0]]), "volume": pd.DataFrame({"AAA": [100]}, index=[idx[0]])}

    assert pyf.create_capacity_tear_sheet(r, pos, txns, md, set_context=False)[0] == "create_capacity_tear_sheet"
    assert (
        pyf.create_bayesian_tear_sheet(r, live_start_date=idx[1], set_context=False)[0] == "create_bayesian_tear_sheet"
    )
    assert pyf.create_risk_tear_sheet(pos, set_context=False)[0] == "create_risk_tear_sheet"
    assert (
        pyf.create_perf_attrib_tear_sheet(r, pos, pd.DataFrame(), pd.DataFrame(), set_context=False)[0]
        == "create_perf_attrib_tear_sheet"
    )

    # Misc wrappers later in the file.
    monkeypatch.setattr(pyfolio_mod, "_plot_sector_allocations", make_stub("plot_sector_allocations"))
    monkeypatch.setattr(pyfolio_mod, "_plot_capacity_sweep", make_stub("plot_capacity_sweep"))
    monkeypatch.setattr(pyfolio_mod, "_plot_monthly_returns_timeseries", make_stub("plot_monthly_returns_timeseries"))
    monkeypatch.setattr(pyfolio_mod, "_plot_cones", make_stub("plot_cones"))
    monkeypatch.setattr(pyfolio_mod, "_show_perf_attrib_stats", make_stub("show_perf_attrib_stats"))
    monkeypatch.setattr(pyfolio_mod, "_plot_perf_attrib_returns", make_stub("plot_perf_attrib_returns"))
    monkeypatch.setattr(pyfolio_mod, "_plot_alpha_returns", make_stub("plot_alpha_returns"))
    monkeypatch.setattr(pyfolio_mod, "_plot_factor_contribution_to_perf", make_stub("plot_factor_contribution_to_perf"))
    monkeypatch.setattr(pyfolio_mod, "_plot_risk_exposures", make_stub("plot_risk_exposures"))

    assert pyf.plot_sector_allocations(r, pd.DataFrame())[0] == "plot_sector_allocations"
    assert pyf.plot_capacity_sweep(r, txns, md, bt_starting_capital=1.0)[0] == "plot_capacity_sweep"
    assert pyf.plot_monthly_returns_timeseries(r)[0] == "plot_monthly_returns_timeseries"
    assert pyf.plot_cones("x", bounds=pd.DataFrame(), oos_returns=r)[0] == "plot_cones"
    assert pyf.show_perf_attrib_stats(r, pos, pd.DataFrame(), pd.DataFrame())[0] == "show_perf_attrib_stats"
    assert pyf.plot_perf_attrib_returns(pd.DataFrame())[0] == "plot_perf_attrib_returns"
    assert pyf.plot_alpha_returns(pd.Series([0.0]))[0] == "plot_alpha_returns"
    assert pyf.plot_factor_contribution_to_perf(pd.DataFrame())[0] == "plot_factor_contribution_to_perf"
    assert pyf.plot_risk_exposures(pd.DataFrame())[0] == "plot_risk_exposures"
    plt.close("all")
