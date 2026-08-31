"""Executable documentation examples for the 0.5 canonical surface."""

from __future__ import annotations

import json
import os
import re
import runpy
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

import fincore

matplotlib.use("Agg", force=True)

_ROOT = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).resolve().parents[2])).resolve()
_FACTOR_ANALYSIS_QUICKSTART = _ROOT / "examples" / "factor_analysis_quickstart.py"
_METRICS_REPORT = _ROOT / "examples" / "metrics_report.py"
_PORTFOLIO_OPTIMIZATION = _ROOT / "examples" / "portfolio_optimization.py"
_RISK_VALIDATION = _ROOT / "examples" / "risk_validation.py"
_MAINTAINED_MARKDOWN = (
    _ROOT / "README.md",
    _ROOT / "docs" / "API_STABILITY.md",
    _ROOT / "docs" / "MIGRATION.md",
    _ROOT / "docs" / "api.md",
    _ROOT / "docs" / "development.md",
    _ROOT / "docs" / "examples.md",
    _ROOT / "docs" / "user_guide.md",
    *sorted((_ROOT / "mkdocs_docs").rglob("*.md")),
)
_EXECUTABLE_FENCE = re.compile(r"```(?:python|bash)\n(.*?)```", re.DOTALL)
_RETIRED_EXECUTABLE_REFERENCE = re.compile(
    r"(?:"
    r"(?:from|import)\s+fincore\.(?:empyrical|pyfolio|alphalens)"
    r"|from\s+fincore\s+import\s+[^\n]*(?:Empyrical|Pyfolio|analyze|create_strategy_report|sharpe_ratio|max_drawdown)"
    r"|fincore\.(?:sharpe_ratio|max_drawdown|analyze|create_strategy_report)\b"
    r"|fincore\[(?:viz|pyfolio|alphalens|alphalens-pyfolio)\]"
    r"|\.\[dev,viz\]"
    r")"
)


def _load_factor_analysis_quickstart() -> dict[str, object]:
    return runpy.run_path(str(_FACTOR_ANALYSIS_QUICKSTART))


def _series(periods: int = 12) -> pd.Series:
    index = pd.date_range("2024-01-02", periods=periods, freq="B")
    return pd.Series(np.resize([0.01, -0.005, 0.002, 0.004, -0.001], periods), index=index, name="strategy")


def test_readme_quick_start_uses_direct_metric_modules() -> None:
    from fincore.metrics.drawdown import max_drawdown
    from fincore.metrics.ratios import sharpe_ratio
    from fincore.metrics.yearly import annual_return

    returns = _series()

    assert np.isfinite(sharpe_ratio(returns))
    assert max_drawdown(returns) < 0
    assert np.isfinite(annual_return(returns))


def test_documented_root_is_only_a_canonical_namespace_index() -> None:
    assert fincore.__all__ == [
        "__version__",
        "attribution",
        "data",
        "errors",
        "extensions",
        "factor_analysis",
        "metrics",
        "optimization",
        "performance",
        "portfolio",
        "report",
        "risk",
        "runtime",
        "simulation",
        "viz",
    ]


def test_maintained_markdown_has_no_retired_executable_surface() -> None:
    """Historical names may be prose, never runnable 0.5 documentation."""

    violations = [
        str(path.relative_to(_ROOT))
        for path in _MAINTAINED_MARKDOWN
        if any(
            _RETIRED_EXECUTABLE_REFERENCE.search(block)
            for block in _EXECUTABLE_FENCE.findall(path.read_text(encoding="utf-8"))
        )
    ]

    assert not violations, "retired executable examples: " + ", ".join(violations)


def test_portfolio_report_workflow_builds_once_and_writes_html(tmp_path: Path) -> None:
    from fincore.report.portfolio.compute import build_portfolio_report
    from fincore.report.renderers.html import write_html

    returns = _series()
    positions = pd.DataFrame({"AAA": 100.0, "BBB": -30.0, "cash": 80.0}, index=returns.index)
    document = build_portfolio_report(returns, positions=positions, rolling_window=3)
    artifact = write_html(document, tmp_path / "portfolio-report.html")

    assert document.domain == "portfolio"
    assert "annual_return" in document.section("performance").metrics
    assert artifact.named_artifacts["file"].is_file()
    assert "Portfolio Report" in artifact.named_artifacts["html"]


def test_performance_cashflow_semantics_example() -> None:
    from fincore.performance.cashflows import cashflow_adjusted_returns, cashflow_adjusted_twr

    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"], utc=True)
    valuations = pd.Series([100.0, 110.0, 121.0], index=dates)
    cashflows = pd.Series([10.0], index=[dates[1]])

    period_returns = cashflow_adjusted_returns(valuations, cashflows, timing="end")
    total_return = cashflow_adjusted_twr(valuations, cashflows, timing="end")

    assert period_returns.round(12).tolist() == [0.0, 0.1]
    assert round(total_return, 12) == 0.1


def test_risk_validation_report_example(tmp_path: Path) -> None:
    from fincore.risk.diagnostics import walk_forward_var
    from fincore.risk.report import build_risk_validation_report
    from fincore.risk.specs import RiskModelSpec

    returns = pd.Series(
        np.linspace(-0.02, 0.02, 60),
        index=pd.date_range("2024-01-02", periods=60, freq="B", tz="UTC"),
    )
    spec = RiskModelSpec(confidence_level=0.95, distribution="normal", window=40, refit_cadence=5)
    walk_forward = walk_forward_var(returns, spec)
    audit_report = build_risk_validation_report(walk_forward)
    output = audit_report.write_json(tmp_path / "risk-validation.json")

    assert audit_report.status == "ok"
    assert json.loads(output.read_text(encoding="utf-8"))["inputs_digest"] == walk_forward.inputs_digest


def test_portfolio_optimization_example() -> None:
    from fincore.optimization.frontier import efficient_frontier
    from fincore.optimization.objectives import optimize
    from fincore.optimization.risk_parity import risk_parity

    returns = pd.DataFrame({"asset_a": [0.01, -0.005, 0.004, 0.002], "asset_b": [0.003, 0.002, -0.001, 0.005]})

    assert "frontier_returns" in efficient_frontier(returns, n_points=5)
    assert "weights" in risk_parity(returns)
    assert "weights" in optimize(returns, objective="max_sharpe")


def test_factor_analysis_quickstart_builds_offline_canonical_inputs() -> None:
    example = _load_factor_analysis_quickstart()
    factor, prices = example["synthetic_factor_inputs"]()

    assert isinstance(factor, pd.Series)
    assert isinstance(prices, pd.DataFrame)
    assert factor.index.names == ["date", "asset"]


def test_canonical_core_examples_execute_without_legacy_imports(tmp_path: Path) -> None:
    """The retained examples exercise only direct 0.5 domain APIs."""

    metrics_example = runpy.run_path(str(_METRICS_REPORT))
    optimization_example = runpy.run_path(str(_PORTFOLIO_OPTIMIZATION))
    risk_example = runpy.run_path(str(_RISK_VALIDATION))

    metrics_example["main"](tmp_path / "portfolio-report.html")
    optimization_example["main"]()
    risk_example["main"]()

    assert (tmp_path / "portfolio-report.html").is_file()


def test_factor_analysis_quickstart_prepares_and_analyzes_model() -> None:
    example = _load_factor_analysis_quickstart()
    prepared, model = example["enhanced_prepare_and_analyze"]()

    assert prepared.loss_report.total_loss <= 0.35
    assert model.forward_periods == ("1D",)
    assert not model.information_coefficient.empty


def test_factor_analysis_quickstart_builds_portfolio_inputs() -> None:
    example = _load_factor_analysis_quickstart()
    inputs = example["factor_portfolio_inputs"]()

    assert inputs.returns.index.isin(inputs.positions.index).all()
    assert "cash" in inputs.positions.columns


def test_factor_analysis_quickstart_renders_and_closes_headless_summary() -> None:
    example = _load_factor_analysis_quickstart()
    artifacts = example["summary_tear_sheet"]()

    assert artifacts.figures
    assert "quantile_statistics" in artifacts.tables
    assert not matplotlib.pyplot.fignum_exists(artifacts.figures[0].number)


def test_factor_analysis_quickstart_names_the_visualization_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    example = _load_factor_analysis_quickstart()
    import fincore.factor_analysis.render_matplotlib as render_matplotlib
    from fincore.exceptions import DependencyError

    real_import = render_matplotlib.importlib.import_module

    def missing_matplotlib(name: str, *args: object, **kwargs: object) -> object:
        if name == "matplotlib.pyplot":
            raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(render_matplotlib.importlib, "import_module", missing_matplotlib)
    with pytest.raises(DependencyError, match=r"pip install fincore\[visualization\]"):
        render_matplotlib._plot_dependencies()
    assert example["OPTIONAL_EXTRA_INSTALL"] == "fincore[visualization]"


def test_factor_cost_and_capacity_ledger_example() -> None:
    from fincore.factor_analysis.costs import FactorCostModel, apply_factor_costs

    dates = pd.date_range("2024-01-02", periods=2, freq="B", tz="UTC", name="date")
    weights = pd.Series(
        [0.60, -0.40, 0.20, -0.80],
        index=pd.MultiIndex.from_product((dates, ("A", "B")), names=("date", "asset")),
    )
    gross_returns = pd.Series([0.010, -0.005], index=dates)
    dollar_volume = pd.DataFrame({"A": [1_000.0, 1_500.0], "B": [2_000.0, 1_000.0]}, index=dates)
    borrow_rates = pd.DataFrame({"A": [0.0, 0.0], "B": [0.002, 0.003]}, index=dates)
    borrow_available = pd.DataFrame(True, index=dates, columns=("A", "B"))

    ledger = apply_factor_costs(
        gross_returns,
        weights,
        dollar_volume,
        portfolio_value=250.0,
        model=FactorCostModel(
            half_spread_bps=10.0, impact_coefficient=0.01, impact_exponent=0.5, max_participation=0.50
        ),
        borrow_rates=borrow_rates,
        borrow_available=borrow_available,
    )

    assert (ledger.participation <= ledger.model.max_participation).all().all()
    pd.testing.assert_series_equal(ledger.net_returns, ledger.gross_returns - ledger.total_cost, check_names=False)


def test_fama_macbeth_newey_west_example() -> None:
    from fincore.factor_analysis.inference import fama_macbeth

    dates = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")
    assets = ["a", "b", "c"]
    exposures = pd.DataFrame(np.tile([-1.0, 0.0, 1.0], (len(dates), 1)), index=dates, columns=assets)
    returns = pd.DataFrame(
        [[-0.02, 0.01, 0.04], [-0.01, 0.0, 0.01], [-0.03, 0.01, 0.05], [-0.02, 0.0, 0.02], [-0.01, 0.02, 0.05]],
        index=dates,
        columns=assets,
    )

    result = fama_macbeth(returns, exposures, covariance="newey-west", newey_west_lags=3)

    assert result.attrs["covariance"] == "newey-west"
    assert result.attrs["newey_west_lags"] == 3
    assert result.attrs["n_cross_sections"] >= 4
