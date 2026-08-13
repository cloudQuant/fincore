"""Task 8 compute-once / render-many model contract.

``compute_sections`` produces a structured :class:`ReportModel`; renderers
consume it without recomputing or mutating it, and
``create_strategy_report(return_result=True)`` returns the artifacts bundle.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fincore.report import create_strategy_report
from fincore.report.artifacts import ReportArtifacts
from fincore.report.compute import compute_sections
from fincore.report.model import ReportModel, SectionModel
from fincore.report.render_html import generate_html


def _returns(n: int = 120) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(
        [0.001 if i % 2 == 0 else -0.0007 for i in range(n)],
        index=idx,
        name="strategy",
    )


def test_compute_sections_returns_report_model() -> None:
    model = compute_sections(_returns(), None, None, None, None, 20)

    assert isinstance(model, ReportModel)
    assert isinstance(model, dict)  # dict-compatible for legacy consumers
    assert model.title == "Strategy Report"
    assert "cum_returns" in model
    assert model.to_dict() == dict(model)


def test_section_models_classify_by_shape() -> None:
    model = compute_sections(_returns(), None, None, None, None, 20)

    views = model.section_models
    assert isinstance(views["perf_stats"], SectionModel)
    # Scalar mappings become metric blocks.
    assert "Sharpe Ratio" in views["perf_stats"].metrics
    # Series become series, DataFrames become tables, text becomes meta.
    assert "cum_returns" in views["cum_returns"].series
    assert "dd_table" in views["dd_table"].tables
    assert "summary_text" in views["summary_text"].meta


def test_classify_sections_handles_nested_mappings_and_tuples() -> None:
    from fincore.report.model import classify_sections

    inner = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2))
    views = classify_sections(
        {
            "nested": {"scalar": 1.5, "series": inner, "table": pd.DataFrame({"a": [1]})},
            "dates": ("2024-01-01", "2024-02-01"),
            "flag": True,
        }
    )

    assert views["nested"].metrics["scalar"] == 1.5
    assert "series" in views["nested"].series
    assert "table" in views["nested"].tables
    assert views["dates"].meta["dates"] == ("2024-01-01", "2024-02-01")
    assert views["flag"].meta["flag"] is True
    assert not views["nested"].is_empty()
    assert SectionModel(name="empty").is_empty()


def test_generate_html_with_model_skips_computation(monkeypatch, tmp_path) -> None:
    model = compute_sections(_returns(), None, None, None, None, 20)
    model.title = "Precomputed"

    def forbidden_compute(*_args, **_kwargs):
        pytest.fail("renderer recomputed statistics although a model was supplied")

    monkeypatch.setattr("fincore.report.render_html.compute_sections", forbidden_compute)

    out = tmp_path / "precomputed.html"
    generate_html(
        _returns(),
        benchmark_rets=None,
        positions=None,
        transactions=None,
        trades=None,
        title="From Model",
        output=str(out),
        rolling_window=20,
        model=model,
    )
    assert out.exists()


def test_generate_html_does_not_mutate_the_caller_model(tmp_path) -> None:
    model = compute_sections(_returns(), None, None, None, None, 20)
    keys_before = set(model.keys())

    out = tmp_path / "nomutate.html"
    generate_html(
        _returns(),
        benchmark_rets=None,
        positions=None,
        transactions=None,
        trades=None,
        title="No Mutation",
        output=str(out),
        rolling_window=20,
        model=model,
    )

    assert set(model.keys()) == keys_before
    assert "_title" not in model


def test_create_strategy_report_return_result_bundles_artifacts(tmp_path) -> None:
    out = tmp_path / "bundle.html"

    result = create_strategy_report(
        _returns(),
        title="Artifact Bundle",
        output=str(out),
        rolling_window=20,
        return_result=True,
    )

    assert isinstance(result, ReportArtifacts)
    assert result.backend == "html"
    assert result.files == [out]
    assert result.html is not None
    assert "Artifact Bundle" in result.html
    assert isinstance(result.model, ReportModel)
    result.close()


def test_create_strategy_report_default_returns_path_only(tmp_path) -> None:
    out = tmp_path / "plain.html"

    result = create_strategy_report(
        _returns(),
        title="Plain Path",
        output=str(out),
        rolling_window=20,
    )

    assert result == str(out)
    assert out.exists()
