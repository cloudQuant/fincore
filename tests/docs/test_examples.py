"""Executable documentation examples.

Every code block shown in README.md, docs/MIGRATION.md, and the MkDocs
getting-started pages that claims to run against the current fincore release is
mirrored here as a real test. Documentation code must never be written first and
guessed later: this file is the executable contract for those snippets.

Keep each test aligned with the corresponding documentation block.  When a
doc example changes, update the matching test (and vice versa).
"""

from __future__ import annotations

import json
import runpy
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

import fincore

# Real tear-sheet chains must stay headless and deterministic.
matplotlib.use("Agg", force=True)


_ROOT = Path(__file__).resolve().parents[2]
_FACTOR_ANALYSIS_QUICKSTART = _ROOT / "examples" / "factor_analysis_quickstart.py"


def _load_factor_analysis_quickstart() -> dict[str, object]:
    """Load the documented offline quickstart without executing its CLI entry point."""

    return runpy.run_path(str(_FACTOR_ANALYSIS_QUICKSTART))


def _series(start: str = "2024-01-02", periods: int = 5, seed: int = 0) -> pd.Series:
    index = pd.date_range(start, periods=periods, freq="B")
    return pd.Series([0.01, -0.005, 0.002, 0.004, -0.001][:periods], index=index)


def _benchmark(periods: int = 5) -> pd.Series:
    index = pd.date_range("2024-01-02", periods=periods, freq="B")
    return pd.Series([0.008, -0.003, 0.001, 0.002, 0.0][:periods], index=index)


# ---------------------------------------------------------------------------
# README Quick Start: flat API
# ---------------------------------------------------------------------------


def test_readme_quick_start_flat_api() -> None:
    # import fincore
    # import pandas as pd
    returns = pd.Series([0.01, -0.005, 0.002, 0.004])

    sharpe = fincore.sharpe_ratio(returns)
    max_dd = fincore.max_drawdown(returns)

    assert np.isfinite(sharpe)
    assert max_dd < 0


# ---------------------------------------------------------------------------
# README / migration: strict module API (from fincore import empyrical)
# ---------------------------------------------------------------------------


def test_strict_module_api_example() -> None:
    import pandas as pd

    from fincore import empyrical

    returns = pd.Series([0.01, -0.005, 0.002, 0.004])

    sharpe = empyrical.sharpe_ratio(returns)
    max_dd = empyrical.max_drawdown(returns)

    assert np.isfinite(sharpe)
    assert max_dd < 0


def test_migration_flat_import_example() -> None:
    # docs/MIGRATION.md "enhanced flat API" example runs unchanged.
    import pandas as pd

    from fincore import max_drawdown, sharpe_ratio

    returns = pd.Series([0.01, -0.005, 0.002, 0.004])
    assert np.isfinite(sharpe_ratio(returns))
    assert max_drawdown(returns) < 0


def test_quickstart_flat_api_example() -> None:
    # mkdocs_docs/getting-started/quickstart.md "Flat API" block.
    import pandas as pd

    import fincore

    returns = pd.Series([0.01, -0.005, 0.002, 0.004])

    sr = fincore.sharpe_ratio(returns)
    md = fincore.max_drawdown(returns)
    ar = fincore.annual_return(returns)

    assert np.isfinite(sr)
    assert md < 0
    assert np.isfinite(ar)


# ---------------------------------------------------------------------------
# Performance-return semantics guide
# ---------------------------------------------------------------------------


def test_performance_cashflow_semantics_example() -> None:
    # mkdocs_docs/guide/performance-semantics.md "Cashflow-adjusted" block.
    from fincore.performance.cashflows import cashflow_adjusted_returns, cashflow_adjusted_twr

    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"], utc=True)
    valuations = pd.Series([100.0, 110.0, 121.0], index=dates)
    cashflows = pd.Series([10.0], index=[dates[1]])

    period_returns = cashflow_adjusted_returns(valuations, cashflows, timing="end")
    total_return = cashflow_adjusted_twr(valuations, cashflows, timing="end")

    assert period_returns.round(12).tolist() == [0.0, 0.1]
    assert round(total_return, 12) == 0.1


def test_performance_transaction_ledger_example() -> None:
    # mkdocs_docs/guide/performance-semantics.md transaction ledger block.
    from fincore.performance.cashflows import cashflow_adjusted_twr

    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"], utc=True)
    ledger = pd.DataFrame(
        {"amount": [10.0, -5.0], "timing": ["start", "end"]},
        index=[dates[1], dates[1]],
    )
    one_period = cashflow_adjusted_twr(
        pd.Series([100.0, 116.0], index=dates[:2]),
        ledger,
    )

    assert round(one_period, 12) == 0.1


# ---------------------------------------------------------------------------
# Risk-validation guide
# ---------------------------------------------------------------------------


def test_risk_validation_report_example(tmp_path: Path) -> None:
    # mkdocs_docs/guide/risk-validation.md "Auditable walk-forward VaR" block.
    from fincore.risk import RiskModelSpec, build_risk_validation_report, walk_forward_var

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


# ---------------------------------------------------------------------------
# Quick Start: classic API (Empyrical class-level call)
# ---------------------------------------------------------------------------


def test_classic_api_class_level_example() -> None:
    from fincore import Empyrical

    returns = _series()
    benchmark = _benchmark()

    sharpe = Empyrical.sharpe_ratio(returns, risk_free=0.02 / 252)
    alpha, beta = Empyrical.alpha_beta(returns, benchmark)

    assert np.isfinite(sharpe)
    assert np.isfinite(alpha) and np.isfinite(beta)


# ---------------------------------------------------------------------------
# Quick Start: instance API (state-bound Empyrical instance)
# ---------------------------------------------------------------------------


def test_instance_api_example() -> None:
    from fincore import Empyrical

    returns = _series()

    emp = Empyrical(returns=returns)
    sharpe = emp.sharpe_ratio()
    max_dd = emp.max_drawdown()

    assert np.isfinite(sharpe)
    assert max_dd < 0


def test_class_and_instance_calls_share_the_same_metric() -> None:
    from fincore import Empyrical

    returns = _series()

    class_level = Empyrical.sharpe_ratio(returns)
    instance_level = Empyrical(returns=returns).sharpe_ratio()

    assert class_level == instance_level


# ---------------------------------------------------------------------------
# AnalysisContext: analyze / export / plot chain
# ---------------------------------------------------------------------------


def test_analysis_context_export_chain() -> None:
    returns = _series()
    benchmark = _benchmark()

    ctx = fincore.analyze(returns, factor_returns=benchmark)

    assert np.isfinite(ctx.sharpe_ratio)
    assert ctx.max_drawdown < 0

    # JSON round-trip (text) and file export.
    payload = json.loads(ctx.to_json())
    assert "Sharpe ratio" in payload

    with tempfile.TemporaryDirectory() as tmp:
        json_path = Path(tmp) / "report.json"
        ctx.to_json(path=json_path)
        assert json_path.is_file() and json_path.stat().st_size > 0

        html_path = Path(tmp) / "report.html"
        ctx.to_html(path=html_path)
        assert html_path.is_file() and html_path.stat().st_size > 0

    assert ctx.perf_stats() is not None
    assert ctx.to_dict()


def test_analysis_context_plot_returns_report_artifacts() -> None:
    from fincore.report.artifacts import ReportArtifacts

    ctx = fincore.analyze(_series(), factor_returns=_benchmark())

    artifacts = ctx.plot(backend="matplotlib")

    assert isinstance(artifacts, ReportArtifacts)


def test_analysis_context_replace_data_invalidates_cache() -> None:
    returns = _series()

    ctx = fincore.analyze(returns)
    before = ctx.sharpe_ratio

    ctx.replace_data(returns=returns + 0.001)
    after = ctx.sharpe_ratio

    assert before != after


def test_analysis_context_snapshot_immune_to_external_mutation() -> None:
    returns = _series()

    ctx = fincore.analyze(returns)
    before = ctx.sharpe_ratio

    # Mutating the caller's series must not stale the cached snapshot.
    returns.iloc[0] = 99.0
    assert ctx.sharpe_ratio == before


# ---------------------------------------------------------------------------
# RollingEngine batch computation
# ---------------------------------------------------------------------------


def test_rolling_engine_example() -> None:
    from fincore.core.engine import RollingEngine

    rng = np.random.default_rng(7)
    index = pd.date_range("2024-01-02", periods=60, freq="B")
    returns = pd.Series(rng.normal(0.001, 0.02, 60), index=index)
    benchmark = pd.Series(rng.normal(0.0005, 0.015, 60), index=index)

    engine = RollingEngine(returns, factor_returns=benchmark, window=30)
    results = engine.compute(["sharpe", "volatility", "max_drawdown", "beta"])

    assert set(results) == {"sharpe", "volatility", "max_drawdown", "beta"}


# ---------------------------------------------------------------------------
# Pyfolio main chain: enhanced class and functional facade share workflows
# ---------------------------------------------------------------------------


def _pyfolio_block_data() -> tuple[pd.Series, pd.Series]:
    # The exact data of the README "Pyfolio main chain" block (which reuses
    # the AnalysisContext block's series above it).
    index = pd.date_range("2024-01-02", periods=5, freq="B")
    returns = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001], index=index)
    benchmark = pd.Series([0.008, -0.003, 0.001, 0.002, 0.0], index=index)
    return returns, benchmark


def test_pyfolio_class_main_chain() -> None:
    # README block, executed as written:
    # from fincore import Pyfolio  # requires fincore[pyfolio]
    from fincore import Pyfolio

    returns, benchmark = _pyfolio_block_data()

    pyfolio = Pyfolio(returns=returns, benchmark_rets=benchmark)
    pyfolio.create_returns_tear_sheet(returns, benchmark_rets=benchmark)


def test_pyfolio_functional_facade_main_chain() -> None:
    import fincore.pyfolio as pyfolio

    returns, benchmark = _pyfolio_block_data()

    pyfolio.create_returns_tear_sheet(returns, benchmark_rets=benchmark)


# ---------------------------------------------------------------------------
# Portfolio optimization examples
# ---------------------------------------------------------------------------


def test_portfolio_optimization_examples() -> None:
    # README block, executed as written (variable names and arguments match).
    from fincore.optimization import efficient_frontier, optimize, risk_parity

    returns_df = pd.DataFrame(
        {
            "asset_a": [0.01, -0.005, 0.004, 0.002],
            "asset_b": [0.003, 0.002, -0.001, 0.005],
        }
    )
    ef = efficient_frontier(returns_df, n_points=5)
    rp = risk_parity(returns_df)
    w = optimize(returns_df, objective="max_sharpe")

    assert isinstance(ef, dict)
    assert isinstance(rp, dict)
    assert isinstance(w, dict)


# ---------------------------------------------------------------------------
# API-stability surfaces referenced by the docs
# ---------------------------------------------------------------------------


def test_stability_documented_top_level_imports() -> None:
    from fincore import Empyrical, Pyfolio, analyze, create_strategy_report

    assert callable(analyze)
    assert callable(create_strategy_report)
    assert Empyrical is not None
    assert Pyfolio is not None


def test_stability_documented_flat_api_names() -> None:
    names = [
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "annual_return",
        "annual_volatility",
        "cum_returns",
        "cum_returns_final",
        "alpha",
        "beta",
        "alpha_beta",
        "calmar_ratio",
        "omega_ratio",
        "information_ratio",
        "stability_of_timeseries",
        "tail_ratio",
        "value_at_risk",
        "capture",
        "downside_risk",
        "simple_returns",
        "aggregate_returns",
    ]
    for name in names:
        assert callable(getattr(fincore, name)), name


# ---------------------------------------------------------------------------
# Factor-analysis migration quickstart: deterministic, offline, headless
# ---------------------------------------------------------------------------


def test_factor_analysis_quickstart_runs_strict_facade() -> None:
    example = _load_factor_analysis_quickstart()

    clean = example["strict_quickstart"]()

    assert isinstance(clean, pd.DataFrame)
    assert {"factor", "factor_quantile", "1D"}.issubset(clean.columns)
    assert clean.index.names == ["date", "asset"]


def test_factor_analysis_quickstart_prepares_and_analyzes_enhanced_model() -> None:
    example = _load_factor_analysis_quickstart()

    prepared, model = example["enhanced_prepare_and_analyze"]()

    assert prepared.loss_report.total_loss <= 0.35
    assert model.forward_periods == ("1D",)
    assert not model.information_coefficient.empty


def test_factor_analysis_quickstart_builds_pyfolio_bridge_inputs() -> None:
    example = _load_factor_analysis_quickstart()

    inputs = example["pyfolio_bridge"]()

    assert inputs.returns.index.isin(inputs.positions.index).all()
    assert "cash" in inputs.positions.columns


def test_factor_analysis_quickstart_renders_and_closes_headless_summary() -> None:
    example = _load_factor_analysis_quickstart()

    artifacts = example["summary_tear_sheet"]()

    assert artifacts.figures
    assert "quantile_statistics" in artifacts.tables
    assert not matplotlib.pyplot.fignum_exists(artifacts.figures[0].number)


def test_factor_analysis_quickstart_documents_missing_extra_message(monkeypatch: pytest.MonkeyPatch) -> None:
    example = _load_factor_analysis_quickstart()
    from fincore.exceptions import DependencyError
    from fincore.factor_analysis import render_matplotlib

    real_import = render_matplotlib.importlib.import_module

    def missing_matplotlib(name: str, *args: object, **kwargs: object) -> object:
        if name == "matplotlib.pyplot":
            raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(render_matplotlib.importlib, "import_module", missing_matplotlib)
    with pytest.raises(DependencyError, match=r"pip install fincore\[alphalens\]"):
        render_matplotlib._plot_dependencies()
    assert example["OPTIONAL_EXTRA_INSTALL"] == "fincore[alphalens]"


def test_factor_cost_and_capacity_ledger_example() -> None:
    # mkdocs_docs/concepts/factor-research-protocol.md cost/capacity block.
    from fincore.factor_analysis import FactorCostModel, apply_factor_costs

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
            half_spread_bps=10.0,
            impact_coefficient=0.01,
            impact_exponent=0.5,
            max_participation=0.50,
        ),
        borrow_rates=borrow_rates,
        borrow_available=borrow_available,
    )

    assert (ledger.participation <= ledger.model.max_participation).all().all()
    pd.testing.assert_series_equal(ledger.net_returns, ledger.gross_returns - ledger.total_cost, check_names=False)


def test_fama_macbeth_newey_west_example() -> None:
    # mkdocs_docs/concepts/factor-research-protocol.md Newey-West block.
    from fincore.factor_analysis import fama_macbeth

    dates = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")
    assets = ["a", "b", "c"]
    exposures = pd.DataFrame(np.tile([-1.0, 0.0, 1.0], (len(dates), 1)), index=dates, columns=assets)
    returns = pd.DataFrame(
        [[-0.02, 0.01, 0.04], [-0.01, 0.0, 0.01], [-0.03, 0.01, 0.05], [-0.02, 0.0, 0.02], [-0.01, 0.02, 0.05]],
        index=dates,
        columns=assets,
    )

    result = fama_macbeth(
        returns,
        exposures,
        covariance="newey-west",
        newey_west_lags=3,
    )

    assert result.attrs["covariance"] == "newey-west"
    assert result.attrs["newey_west_lags"] == 3
    assert result.attrs["n_cross_sections"] >= 4
