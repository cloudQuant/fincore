"""Bilingual self-contained HTML factor-report contracts."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from fincore.factor_analysis.data import FactorLossReport


def _factor_data() -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=80)
    assets = [f"S{index:02d}" for index in range(30)]
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, (len(dates), len(assets))), axis=0),
        index=dates,
        columns=assets,
    )
    index = pd.MultiIndex.from_product([dates, assets], names=("date", "asset"))
    factor = pd.Series(rng.normal(size=len(index)), index=index, name="factor")
    return factor, prices


def _empty_quantile_data(pattern: str) -> tuple[pd.Series, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=12)
    assets = [f"S{index:02d}" for index in range(6)]
    index = pd.MultiIndex.from_product([dates, assets], names=("date", "asset"))
    if pattern == "binary":
        values = np.tile([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], len(dates))
    elif pattern == "constant":
        values = np.full(len(index), 1.0)
    else:  # pragma: no cover - test helper guard
        raise ValueError(f"unknown quantile fixture pattern: {pattern}")
    factor = pd.Series(values, index=index, name="factor")
    rng = np.random.default_rng(4)
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, (len(dates), len(assets))), axis=0),
        index=dates,
        columns=assets,
    )
    return factor, prices


def _assert_empty_quantile_html(rendered: str, language: str, loss_report: FactorLossReport) -> None:
    assert f'<html lang="{language}">' in rendered
    assert "</html>" in rendered
    assert "Input cells" in rendered and "Finite factor cells" in rendered
    assert "Forward-return cells" in rendered and "Binning cells" in rendered
    assert "100.00%" in rendered
    for count in (
        loss_report.input_count,
        loss_report.finite_factor_count,
        loss_report.forward_returns_count,
        loss_report.binning_count,
    ):
        assert f"<td>{count:,}</td>" in rendered
    for metric in (
        "Mean Period Wise Return Top Quantile (bps)",
        "Mean Period Wise Return Bottom Quantile (bps)",
        "Mean Period Wise Spread (bps)",
    ):
        row = re.compile(rf"<tr>\s*<td>{re.escape(metric)}</td>(?P<cells>(?:\s*<td>[^<]*</td>)+)\s*</tr>")
        match = row.search(rendered)
        assert match is not None
        assert "—" in match.group("cells")
        assert not re.search(r"<td>0(?:\.0+)?</td>", match.group("cells"))


@pytest.mark.parametrize("language", ("zh", "en"))
def test_factor_html_report_is_bilingual_and_self_contained(tmp_path: Path, language: str) -> None:
    from fincore.report.factor import write_factor_report

    factor, prices = _factor_data()
    path = write_factor_report(
        factor,
        prices,
        tmp_path / f"report-{language}.html",
        periods=(1, 5, 20),
        quantiles=5,
        title="Demo Factor",
        language=language,
    )
    rendered = path.read_text(encoding="utf-8")

    assert f'<html lang="{language}">' in rendered
    assert f'body class="lang-{language}"' in rendered
    assert 'data-lang="zh"' in rendered and 'data-lang="en"' in rendered
    assert "信息系数（IC）时间序列" in rendered
    assert "Information Coefficient (IC) Time Series" in rendered
    assert "IC 统计" in rendered and "IC statistics" in rendered
    assert "echarts" in rendered.casefold()
    assert 'src="http' not in rendered and 'href="http' not in rendered
    assert "ic-chart" in rendered and "turnover-chart" in rendered
    assert "data-table" in rendered
    assert path.stat().st_size > 500_000


def test_factor_html_report_from_model_includes_preparation_coverage(tmp_path: Path) -> None:
    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.data import prepare_factor_data
    from fincore.report.factor import write_factor_report_from_model

    factor, prices = _factor_data()
    prepared = prepare_factor_data(factor, prices, periods=(1, 5), quantiles=5, max_loss=1.0)
    model = analyze_factor(prepared.data, periods=None, turnover_periods=(1, 5))
    path = write_factor_report_from_model(
        model,
        tmp_path / "report-model.html",
        language="en",
        loss_report=prepared.loss_report,
    )
    rendered = path.read_text(encoding="utf-8")

    assert '<html lang="en">' in rendered
    assert "IC statistics" in rendered
    assert "Input cells" in rendered
    assert "Finite factor cells" in rendered


@pytest.mark.parametrize("pattern", ("binary", "constant"))
@pytest.mark.parametrize("language", ("zh", "en"))
def test_factor_html_report_handles_finite_inputs_without_quantile_assignments(
    tmp_path: Path, pattern: str, language: str
) -> None:
    from fincore.factor_analysis.data import prepare_factor_data
    from fincore.report.factor import write_factor_report

    factor, prices = _empty_quantile_data(pattern)
    factor_before = factor.copy(deep=True)
    prices_before = prices.copy(deep=True)
    prepared = prepare_factor_data(
        factor,
        prices.shift(-1),
        periods=(1, 5),
        quantiles=3,
        max_loss=1.0,
    )
    assert prepared.data.empty
    assert prepared.loss_report.input_count == len(factor)
    assert prepared.loss_report.finite_factor_count == len(factor)
    assert prepared.loss_report.forward_returns_count > 0
    assert prepared.loss_report.binning_count == 0
    assert prepared.loss_report.total_loss == 1.0

    path = write_factor_report(
        factor,
        prices,
        tmp_path / f"empty-{pattern}-{language}.html",
        periods=(1, 5),
        quantiles=3,
        max_loss=1.0,
        language=language,
    )
    rendered = path.read_text(encoding="utf-8")

    _assert_empty_quantile_html(rendered, language, prepared.loss_report)
    pd.testing.assert_series_equal(factor, factor_before)
    pd.testing.assert_frame_equal(prices, prices_before)


@pytest.mark.parametrize("pattern", ("binary", "constant"))
def test_factor_html_empty_model_can_be_rendered_again_without_recomputation_or_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pattern: str,
) -> None:
    from fincore.factor_analysis import analysis as analysis_module
    from fincore.factor_analysis import data as data_module
    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.data import prepare_factor_data
    from fincore.report.factor import write_factor_report_from_model

    factor, prices = _empty_quantile_data(pattern)
    factor_before = factor.copy(deep=True)
    prices_before = prices.copy(deep=True)
    prepared = prepare_factor_data(
        factor,
        prices.shift(-1),
        periods=(1, 5),
        quantiles=3,
        max_loss=1.0,
    )
    model = analyze_factor(prepared.data, periods=None, turnover_periods=(1, 5))
    assert model.factor_data.empty
    before = json.dumps(model.to_serializable(), ensure_ascii=False, sort_keys=True, allow_nan=False)
    fingerprint = model.result_fingerprint

    def must_not_recompute(*args: object, **kwargs: object) -> None:
        raise AssertionError("model-only rendering must not prepare or analyze the factor again")

    monkeypatch.setattr(data_module, "prepare_factor_data", must_not_recompute)
    monkeypatch.setattr(analysis_module, "analyze_factor", must_not_recompute)

    for language in ("zh", "en"):
        path = write_factor_report_from_model(
            model,
            tmp_path / f"empty-model-{pattern}-{language}.html",
            language=language,
            loss_report=prepared.loss_report,
        )
        _assert_empty_quantile_html(path.read_text(encoding="utf-8"), language, prepared.loss_report)

    assert model.result_fingerprint == fingerprint
    assert json.dumps(model.to_serializable(), ensure_ascii=False, sort_keys=True, allow_nan=False) == before
    pd.testing.assert_series_equal(factor, factor_before)
    pd.testing.assert_frame_equal(prices, prices_before)


def test_factor_html_model_rerenders_without_repreparing_or_reanalyzing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fincore.factor_analysis import analysis as analysis_module
    from fincore.factor_analysis import data as data_module
    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.data import prepare_factor_data
    from fincore.report.factor import write_factor_report_from_model

    factor, prices = _factor_data()
    prepared = prepare_factor_data(factor, prices, periods=(1, 5), quantiles=5, max_loss=1.0)
    model = analyze_factor(prepared.data, periods=None, turnover_periods=(1, 5))
    before = json.dumps(model.to_serializable(), ensure_ascii=False, sort_keys=True, allow_nan=False)
    fingerprint = model.result_fingerprint

    def must_not_recompute(*args: object, **kwargs: object) -> None:
        raise AssertionError("model-only rendering must not prepare or analyze the factor again")

    monkeypatch.setattr(data_module, "prepare_factor_data", must_not_recompute)
    monkeypatch.setattr(analysis_module, "analyze_factor", must_not_recompute)

    chinese = write_factor_report_from_model(
        model,
        tmp_path / "rerender-zh.html",
        title="同一模型中文标题",
        language="zh",
        loss_report=prepared.loss_report,
    )
    english = write_factor_report_from_model(
        model,
        tmp_path / "rerender-en.html",
        title="Same model English title",
        language="en",
        loss_report=prepared.loss_report,
    )

    chinese_html = chinese.read_text(encoding="utf-8")
    english_html = english.read_text(encoding="utf-8")
    assert '<html lang="zh">' in chinese_html
    assert "同一模型中文标题" in chinese_html
    assert '<html lang="en">' in english_html
    assert "Same model English title" in english_html
    assert model.result_fingerprint == fingerprint
    assert json.dumps(model.to_serializable(), ensure_ascii=False, sort_keys=True, allow_nan=False) == before


def test_factor_html_report_rejects_an_unknown_language(tmp_path: Path) -> None:
    from fincore.report.factor import write_factor_report

    factor, prices = _factor_data()
    with pytest.raises(ValueError, match="language"):
        write_factor_report(factor, prices, tmp_path / "bad.html", language="fr")
