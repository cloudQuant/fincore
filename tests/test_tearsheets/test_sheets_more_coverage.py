import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import fincore.tearsheets.sheets as sheets


class _FakePyfolioInterestingTimesHappy:
    def extract_interesting_date_ranges(self, returns: pd.Series) -> dict[str, pd.Series]:
        # Return two small windows to exercise the plotting loop.
        mid = max(int(len(returns) / 2), 2)
        return {
            "Event A": returns.iloc[:mid],
            "Event B": returns.iloc[-mid:],
        }

    def cum_returns(self, rets: pd.Series) -> pd.Series:
        return (1 + rets.fillna(0)).cumprod() - 1


def test_create_interesting_times_tear_sheet_happy_path_with_benchmark(monkeypatch) -> None:
    monkeypatch.setattr(sheets, "print_table", lambda *_args, **_kwargs: None)
    pyf = _FakePyfolioInterestingTimesHappy()

    idx = pd.date_range("2024-01-01", periods=8, freq="B", tz="UTC")
    returns = pd.Series(np.linspace(0.001, -0.001, len(idx)), index=idx, name="r")
    benchmark = returns * 0.5

    fig = sheets.create_interesting_times_tear_sheet(pyf, returns, benchmark_rets=benchmark, run_flask_app=True)
    assert fig is not None
    assert len(fig.axes) == 2
    plt.close("all")


class _FakePyfolioBayes:
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


def test_create_bayesian_tear_sheet_requires_live_start_date() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    returns = pd.Series([0.001] * len(idx), index=idx, name="r")
    with pytest.raises(NotImplementedError):
        sheets.create_bayesian_tear_sheet(_FakePyfolioBayes(), returns)


def test_create_bayesian_tear_sheet_happy_path_with_benchmark_and_stoch_vol(monkeypatch) -> None:
    # Keep the test fast and deterministic: avoid seaborn overhead.
    monkeypatch.setattr(sheets.sns, "histplot", lambda *args, **kwargs: None)
    monkeypatch.setattr(sheets, "timer", lambda *_args, **_kwargs: _args[1])

    idx = pd.date_range("2022-01-03", periods=650, freq="B", tz="UTC")
    returns = pd.Series(0.0001, index=idx, name="r")
    benchmark = returns * 0.5
    live_start_date = idx[500]

    pyf = _FakePyfolioBayes()
    fig = sheets.create_bayesian_tear_sheet(
        pyf,
        returns,
        benchmark_rets=benchmark,
        live_start_date=live_start_date,
        samples=10,
        run_flask_app=True,
        stoch_vol=True,
        progressbar=False,
    )
    assert fig is not None
    assert "run_model:t" in pyf.calls
    assert "run_model:best" in pyf.calls
    assert "run_model:alpha_beta" in pyf.calls
    assert "model_stoch_vol" in pyf.calls
    plt.close("all")


class _FakePyfolioPerfAttrib:
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


def test_create_perf_attrib_tear_sheet_with_and_without_partitions(monkeypatch) -> None:
    monkeypatch.setattr(sheets, "display", lambda *_args, **_kwargs: None)

    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    returns = pd.Series(0.001, index=idx, name="r")
    positions = pd.DataFrame({"AAA": 1.0, "cash": 0.0}, index=idx)
    factor_returns = pd.DataFrame({"MKT": 0.0, "SMB": 0.0, "TECH": 0.0}, index=idx)
    factor_loadings = pd.DataFrame({"AAA": [1.0, 0.5, 0.2]}, index=["MKT", "SMB", "TECH"])

    pyf = _FakePyfolioPerfAttrib()
    fig = sheets.create_perf_attrib_tear_sheet(
        pyf,
        returns,
        positions,
        factor_returns,
        factor_loadings,
        run_flask_app=True,
        factor_partitions={"style": ["MKT", "SMB"], "industry": ["TECH"]},
    )
    assert fig is not None

    fig2 = sheets.create_perf_attrib_tear_sheet(
        pyf,
        returns,
        positions,
        factor_returns,
        factor_loadings,
        run_flask_app=True,
        factor_partitions=None,
    )
    assert fig2 is not None
    assert "perf_attrib" in pyf.calls
    plt.close("all")


class _FakePyfolioRisk:
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


def test_create_risk_tear_sheet_handles_optional_panels(monkeypatch) -> None:
    monkeypatch.setattr(sheets, "check_intraday", lambda *_args, **_kwargs: _args[2])

    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    positions = pd.DataFrame({"AAA": 1.0, "cash": 0.0}, index=idx)
    sectors = pd.DataFrame({"AAA": ["Tech"] * len(idx)}, index=idx)
    caps = pd.DataFrame({"AAA": [1e9] * len(idx)}, index=idx)
    volumes = pd.DataFrame({"AAA": [1000] * len(idx)}, index=idx)
    style_panel = {"Momentum": pd.DataFrame({"AAA": 0.1}, index=idx)}

    pyf = _FakePyfolioRisk()
    fig = sheets.create_risk_tear_sheet(
        pyf,
        positions=positions,
        style_factor_panel=style_panel,
        sectors=sectors,
        caps=caps,
        volumes=volumes,
        percentile=None,
        returns=None,
        transactions=None,
        run_flask_app=True,
    )
    assert fig is not None
    assert "compute_style_factor_exposures" in pyf.calls
    assert "compute_sector_exposures" in pyf.calls
    assert "compute_cap_exposures" in pyf.calls
    assert "compute_volume_exposures" in pyf.calls

    # Also exercise the path without style_factor_panel (previously crashed due to undefined variables).
    fig2 = sheets.create_risk_tear_sheet(
        pyf,
        positions=positions,
        style_factor_panel=None,
        sectors=sectors,
        caps=None,
        volumes=None,
        returns=None,
        transactions=None,
        run_flask_app=True,
    )
    assert fig2 is not None
    plt.close("all")


def test_fallback_display_and_markdown_functions() -> None:
    """Test fallback display and markdown functions (lines 57, 61)."""
    # Test _fallback_display - should print the objects
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        sheets._fallback_display("test", 123)
    assert "test" in f.getvalue()

    # Test _fallback_markdown - should return the text as-is
    result = sheets._fallback_markdown("# Test Markdown")
    assert result == "# Test Markdown"


def test_interesting_times_tear_sheet_without_benchmark_line_678() -> None:
    """Test interesting_times_tear_sheet without benchmark (line 678)."""
    pyf = _FakePyfolioInterestingTimesHappy()

    idx = pd.date_range("2024-01-01", periods=8, freq="B", tz="UTC")
    returns = pd.Series(np.linspace(0.001, -0.001, len(idx)), index=idx, name="r")

    fig = sheets.create_interesting_times_tear_sheet(pyf, returns, benchmark_rets=None, run_flask_app=True)
    assert fig is not None
    plt.close("all")


def test_risk_tear_sheet_empty_index_warns_and_returns_line_939() -> None:
    """Test risk tear sheet with non-overlapping indices (lines 939-940)."""
    import warnings

    pyf = _FakePyfolioRisk()

    # Create positions and sectors with non-overlapping indices
    idx_pos = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    idx_sec = pd.date_range("2023-01-01", periods=10, freq="B", tz="UTC")

    positions = pd.DataFrame({"AAA": 1.0, "cash": 0.0}, index=idx_pos)
    sectors = pd.DataFrame({"AAA": ["Tech"] * len(idx_sec)}, index=idx_sec)

    # Should warn and return early
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = sheets.create_risk_tear_sheet(
            pyf,
            positions=positions,
            style_factor_panel=None,
            sectors=sectors,
            caps=None,
            volumes=None,
            returns=None,
            transactions=None,
            run_flask_app=True,
        )
        # Should return None due to warning
        assert result is None
        assert any("No overlapping index" in str(warning.message) for warning in w)


def test_risk_tear_sheet_no_panels_raises_line_968() -> None:
    """Test risk tear sheet with no panels raises ValueError (line 968)."""
    pyf = _FakePyfolioRisk()

    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    positions = pd.DataFrame({"AAA": 1.0, "cash": 0.0}, index=idx)

    # No style_factor_panel, sectors, caps, or volumes
    with pytest.raises(ValueError, match="requires at least one of"):
        sheets.create_risk_tear_sheet(
            pyf,
            positions=positions,
            style_factor_panel=None,
            sectors=None,
            caps=None,
            volumes=None,
            returns=None,
            transactions=None,
        )


def test_bayesian_tear_sheet_with_small_dataset_line_889() -> None:
    """Test bayesian tear sheet when df_train.size <= returns_cutoff (line 889)."""
    # Use a small dataset to hit the else branch at line 888-889
    import warnings

    # Suppress seaborn warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(sheets, "timer", lambda *_args, **_kwargs: _args[1])
    monkeypatch.setattr(sheets.sns, "histplot", lambda *args, **kwargs: None)

    # Create small dataset (< 400 returns for stoch_vol)
    idx = pd.date_range("2022-01-03", periods=100, freq="B", tz="UTC")
    returns = pd.Series(0.0001, index=idx, name="r")
    benchmark = returns * 0.5
    live_start_date = idx[50]

    pyf = _FakePyfolioBayes()
    fig = sheets.create_bayesian_tear_sheet(
        pyf,
        returns,
        benchmark_rets=benchmark,
        live_start_date=live_start_date,
        samples=10,
        run_flask_app=True,
        stoch_vol=True,
        progressbar=False,
    )
    assert fig is not None
    plt.close("all")
    monkeypatch.undo()


def test_bayesian_tear_sheet_run_flask_app_false_line_903() -> None:
    """Test bayesian tear sheet with run_flask_app=False (line 903)."""
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

    idx = pd.date_range("2022-01-03", periods=500, freq="B", tz="UTC")
    returns = pd.Series(0.0001, index=idx, name="r")
    live_start_date = idx[400]

    pyf = _FakePyfolioBayes()
    # run_flask_app=False means function returns None
    result = sheets.create_bayesian_tear_sheet(
        pyf,
        returns,
        live_start_date=live_start_date,
        samples=10,
        run_flask_app=False,
        stoch_vol=False,
        progressbar=False,
    )
    # Should return None when run_flask_app=False
    assert result is None
