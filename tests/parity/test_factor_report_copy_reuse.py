"""Copy-on-read reuse contracts for factor report projection."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def _factor_data(*, scale: float = 1.0, offset: float = 0.0) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=12)
    assets = ("A", "B", "C", "D")
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    rows = np.arange(len(index))
    asset_offsets = rows % len(assets)
    date_offsets = rows // len(assets)
    factor_by_asset = np.array([-1.4, -0.3, 0.6, 1.7])
    one_day_by_asset = np.array([0.015, -0.01, 0.02, -0.005])
    two_day_by_asset = np.array([0.025, -0.015, 0.03, -0.01])

    return pd.DataFrame(
        {
            "factor": factor_by_asset[asset_offsets] + offset + date_offsets * 0.03,
            "factor_quantile": np.tile(np.arange(1, len(assets) + 1), len(dates)),
            "1D": one_day_by_asset[asset_offsets] * scale + (date_offsets % 3) * 0.0007,
            "2D": two_day_by_asset[asset_offsets] * scale - (date_offsets % 4) * 0.0003,
        },
        index=index,
    )


def _scalar_metrics(value: pd.Series | pd.DataFrame, *, prefix: str) -> dict[str, float]:
    if isinstance(value, pd.Series):
        return {f"{prefix}.{key}": float(item) for key, item in value.items()}
    if value.empty:
        return {}
    first = value.iloc[0]
    return {f"{prefix}.{key}": float(item) for key, item in first.items()}


def _uncached_public_api_reference(model: Any, *, title: str) -> Any:
    """Recreate the previous projection using repeated public model reads."""

    from fincore.factor_analysis.render_matplotlib import (
        build_information_table,
        build_quantile_statistics_table,
        build_returns_table,
        build_turnover_tables,
    )
    from fincore.report.models import ReportDocument, ReportSection

    periods = tuple(str(period) for period in model.forward_periods)
    cumulative_series = {
        f"factor_cumulative_returns.{period}": values
        for period, values in model.factor_cumulative_returns.items()
    }
    cumulative_units = dict.fromkeys(cumulative_series, "growth_multiple")
    cumulative_legends = {
        key: f"Factor {period}"
        for key, period in (
            (key, key.rsplit(".", maxsplit=1)[-1]) for key in cumulative_series
        )
    }
    ic_series = {f"ic.{period}": model.information_coefficient[period] for period in periods}
    ic_units = dict.fromkeys(ic_series, "correlation")
    ic_legends = {key: f"{period} IC" for key, period in zip(ic_series, periods, strict=True)}

    information_summary = build_information_table(model.information_coefficient)
    quantile_statistics = build_quantile_statistics_table(model.factor_data)
    returns_summary = build_returns_table(
        model.alpha_beta, model.mean_returns_by_quantile, model.mean_return_spread
    )
    autocorrelation: dict[object, pd.Series] = {
        int(lag): model.rank_autocorrelation[lag] for lag in model.rank_autocorrelation.columns
    }
    turnover_input: dict[object, pd.DataFrame] = {
        int(lag): values for lag, values in model.quantile_turnover.items()
    }
    turnover_summary, autocorrelation_summary = build_turnover_tables(
        autocorrelation, turnover_input
    )
    turnover_by_quantile = pd.DataFrame(
        {f"{lag}D": values.mean() for lag, values in sorted(model.quantile_turnover.items())}
    )
    turnover_by_quantile.index.name = "factor_quantile"

    observed = len(model.factor_data)
    finite = int(model.factor_data["factor"].notna().sum())
    coverage_metrics: dict[str, Any] = {
        "observation_count": observed,
        "finite_factor_count": finite,
        "coverage_ratio": finite / observed if observed else float("nan"),
        "quantile_count": int(model.factor_data["factor_quantile"].nunique()),
    }
    for period in model.forward_periods:
        if period in model.factor_data.columns:
            coverage_metrics[f"forward_return_count.{period}"] = int(
                model.factor_data[period].notna().sum()
            )

    summary_metrics: dict[str, Any] = {
        "forward_period_count": len(periods),
        "result_fingerprint": model.result_fingerprint,
        **_scalar_metrics(
            model.mean_information_coefficient, prefix="mean_information_coefficient"
        ),
        **coverage_metrics,
    }
    summary_tables = {
        "quantile_statistics": quantile_statistics,
        "alpha_beta": model.alpha_beta,
        "mean_returns_by_quantile": model.mean_returns_by_quantile,
        "information_coefficient": model.information_coefficient,
    }

    dates = pd.DatetimeIndex(model.factor_data.index.get_level_values("date"))
    assets = model.factor_data.index.get_level_values("asset").unique()
    dataset_metadata: dict[str, Any]
    if not len(dates):
        dataset_metadata = {"asset_count": 0, "date_count": 0, "start": "", "end": ""}
    else:
        dataset_metadata = {
            "asset_count": len(assets),
            "date_count": int(dates.nunique()),
            "start": dates.min().isoformat(),
            "end": dates.max().isoformat(),
        }

    return ReportDocument(
        domain="factor",
        title=title,
        sections=(
            ReportSection(
                key="factor_summary",
                title="Factor summary",
                metrics=summary_metrics,
                tables=summary_tables,
                series=cumulative_series,
                units=cumulative_units,
                legends=cumulative_legends,
            ),
            ReportSection(
                key="factor_information",
                title="Information coefficient",
                tables={"information_summary": information_summary},
                series=ic_series,
                units=ic_units,
                legends=ic_legends,
            ),
            ReportSection(
                key="factor_returns",
                title="Factor returns",
                tables={
                    "returns_summary": returns_summary,
                    "alpha_beta": model.alpha_beta,
                    "mean_returns_by_quantile": model.mean_returns_by_quantile,
                },
            ),
            ReportSection(
                key="factor_turnover",
                title="Turnover and autocorrelation",
                tables={
                    "mean_turnover_by_quantile": turnover_by_quantile,
                    "rank_autocorrelation_summary": autocorrelation_summary,
                    "turnover_summary": turnover_summary,
                },
            ),
        ),
        metadata={
            "forward_periods": periods,
            "factor_result_fingerprint": model.result_fingerprint,
            **dataset_metadata,
        },
    )


def test_factor_report_reads_each_public_model_field_once_and_preserves_semantics(
    monkeypatch,
) -> None:
    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.models import FactorAnalysisModel
    from fincore.report.factor.compute import build_factor_report

    model = analyze_factor(
        _factor_data(),
        periods=("1D", "2D"),
        turnover_periods=(1, 5),
        include_portfolio_inputs=False,
    )
    expected_fields = {
        "forward_periods",
        "factor_data",
        "factor_cumulative_returns",
        "information_coefficient",
        "alpha_beta",
        "mean_returns_by_quantile",
        "mean_return_spread",
        "rank_autocorrelation",
        "quantile_turnover",
        "mean_information_coefficient",
        "result_fingerprint",
    }
    original_getattribute = FactorAnalysisModel.__getattribute__
    field_reads: Counter[str] = Counter()

    def counted_getattribute(instance: FactorAnalysisModel, name: str) -> object:
        if instance is model and name in expected_fields:
            field_reads[name] += 1
        return original_getattribute(instance, name)

    monkeypatch.setattr(FactorAnalysisModel, "__getattribute__", counted_getattribute)
    actual = build_factor_report(model, title="Copy reuse")

    assert field_reads == Counter(dict.fromkeys(expected_fields, 1))
    monkeypatch.setattr(FactorAnalysisModel, "__getattribute__", original_getattribute)

    expected = _uncached_public_api_reference(model, title="Copy reuse")
    assert actual.semantic_payload() == expected.semantic_payload()
    assert actual.semantic_digest == expected.semantic_digest


def test_factor_report_results_are_isolated_between_models_and_from_model_data() -> None:
    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.report.factor.compute import build_factor_report

    first_model = analyze_factor(
        _factor_data(),
        periods=("1D",),
        turnover_periods=(1,),
        include_portfolio_inputs=False,
    )
    second_model = analyze_factor(
        _factor_data(scale=1.7, offset=0.2),
        periods=("2D",),
        turnover_periods=(1, 5),
        include_portfolio_inputs=False,
    )
    assert first_model.config.fingerprint != second_model.config.fingerprint
    assert first_model.factor_data is not second_model.factor_data

    first_source = first_model.alpha_beta
    second_source = second_model.alpha_beta
    first_report = build_factor_report(first_model)
    second_report = build_factor_report(second_model)
    first_digest = first_report.semantic_digest
    second_digest = second_report.semantic_digest
    assert first_report.metadata["forward_periods"] == ("1D",)
    assert second_report.metadata["forward_periods"] == ("2D",)

    first_report.section("factor_returns").tables["alpha_beta"].iloc[0, 0] = 123456.0

    assert first_model.alpha_beta.equals(first_source)
    assert second_model.alpha_beta.equals(second_source)
    assert second_report.section("factor_returns").tables["alpha_beta"].equals(second_source)
    assert build_factor_report(first_model).semantic_digest == first_digest
    assert build_factor_report(second_model).semantic_digest == second_digest
