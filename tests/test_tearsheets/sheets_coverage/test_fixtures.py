"""Shared fixtures and fake classes for tearsheet coverage tests."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class _FakePyfolioInterestingTimesHappy:
    """Fake pyfolio for interesting times tests."""

    def extract_interesting_date_ranges(self, returns: pd.Series) -> dict[str, pd.Series]:
        # Return two small windows to exercise the plotting loop.
        mid = max(int(len(returns) / 2), 2)
        return {
            "Event A": returns.iloc[:mid],
            "Event B": returns.iloc[-mid:],
        }

    def cum_returns(self, rets: pd.Series) -> pd.Series:
        return (1 + rets.fillna(0)).cumprod() - 1


class _FakePyfolioBayes:
    """Fake pyfolio for bayesian tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run_model(self, model_name, *args, **kwargs):
        self.calls.append(f"run_model:{model_name}")
        if model_name == "t":
            # ppc_t: (n_samples, n_days) at least 5 columns.
            ppc = np.zeros((200, 5), dtype=float)
            ppc[:, 0] = np.linspace(-0.02, 0.02, 200)
            ppc[:, 1] = 0.001
            ppc[:, 2] = -0.001
            ppc[:, 3] = 0.0
            ppc[:, 4] = 0.0005
            return {"trace": "t"}, ppc
        if model_name == "best":
            return {"trace": "best"}
        if model_name == "alpha_beta":
            return {"alpha": np.linspace(-0.001, 0.001, 300), "beta": np.linspace(0.8, 1.2, 300)}
        raise AssertionError(f"unexpected model_name={model_name!r}")

    def plot_bayes_cone(self, *args, **kwargs):
        self.calls.append("plot_bayes_cone")

    def plot_best(self, trace, axs):
        self.calls.append("plot_best")
        assert trace["trace"] == "best"
        # Minimal draw to ensure axes are usable.
        axs[0].plot([0, 1], [0, 1])

    def model_stoch_vol(self, returns):
        self.calls.append("model_stoch_vol")
        return None, {"trace": "stoch_vol"}

    def plot_stoch_vol(self, *args, **kwargs):
        self.calls.append("plot_stoch_vol")


class _FakePyfolioPerfAttrib:
    """Fake pyfolio for performance attribution tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def perf_attrib(self, *args, **kwargs):
        self.calls.append("perf_attrib")
        # exposures/perf_attrib_data are time-indexed.
        returns = args[0]
        idx = returns.index
        cols = ["MKT", "SMB", "TECH"]
        exposures = pd.DataFrame(0.0, index=idx, columns=cols)
        perf_attrib_data = pd.DataFrame(0.0, index=idx, columns=cols)
        return exposures, perf_attrib_data

    def show_perf_attrib_stats(self, *args, **kwargs):
        self.calls.append("show_perf_attrib_stats")

    def plot_perf_attrib_returns(self, *args, **kwargs):
        self.calls.append("plot_perf_attrib_returns")

    def plot_factor_contribution_to_perf(self, *args, **kwargs):
        self.calls.append("plot_factor_contribution_to_perf")

    def plot_risk_exposures(self, *args, **kwargs):
        self.calls.append("plot_risk_exposures")


class _FakePyfolioRisk:
    """Fake pyfolio for risk tear sheet tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def compute_style_factor_exposures(self, positions, df):
        self.calls.append("compute_style_factor_exposures")
        return pd.DataFrame({"x": 0.0}, index=positions.index)

    def plot_style_factor_exposures(self, sfe, name, ax):
        self.calls.append("plot_style_factor_exposures")
        ax.plot([0, 1], [0, 1])

    def compute_sector_exposures(self, positions, sectors):
        self.calls.append("compute_sector_exposures")
        idx = positions.index
        z = pd.DataFrame(0.0, index=idx, columns=["Tech"])
        return z, z, z, z

    def plot_sector_exposures_longshort(self, *args, **kwargs):
        self.calls.append("plot_sector_exposures_longshort")

    def plot_sector_exposures_gross(self, *args, **kwargs):
        self.calls.append("plot_sector_exposures_gross")

    def plot_sector_exposures_net(self, *args, **kwargs):
        self.calls.append("plot_sector_exposures_net")

    def compute_cap_exposures(self, positions, caps):
        self.calls.append("compute_cap_exposures")
        idx = positions.index
        z = pd.DataFrame(0.0, index=idx, columns=["Large"])
        return z, z, z, z

    def plot_cap_exposures_longshort(self, *args, **kwargs):
        self.calls.append("plot_cap_exposures_longshort")

    def plot_cap_exposures_gross(self, *args, **kwargs):
        self.calls.append("plot_cap_exposures_gross")

    def plot_cap_exposures_net(self, *args, **kwargs):
        self.calls.append("plot_cap_exposures_net")

    def compute_volume_exposures(self, positions, volumes, percentile):
        self.calls.append("compute_volume_exposures")
        idx = positions.index
        z = pd.DataFrame(0.0, index=idx, columns=["p"])
        return z, z, z

    def plot_volume_exposures_longshort(self, *args, **kwargs):
        self.calls.append("plot_volume_exposures_longshort")

    def plot_volume_exposures_gross(self, *args, **kwargs):
        self.calls.append("plot_volume_exposures_gross")
