"""Bilingual self-contained HTML factor-report contracts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path


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
