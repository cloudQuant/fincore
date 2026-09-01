"""Boundary tests for report projection, extensions, and portfolio inputs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _returns(size: int = 72) -> pd.Series:
    index = pd.date_range("2025-01-02", periods=size, freq="B", tz="UTC")
    values = np.resize(np.array([0.012, -0.006, 0.003, 0.001], dtype=float), size)
    return pd.Series(values, index=index, name="strategy", dtype=float)


def _report_document() -> object:
    from fincore.report.models import ReportDocument, ReportSection

    values = pd.Series([0.01, -0.02], index=pd.date_range("2025-01-02", periods=2, tz="UTC"), name="returns")
    section = ReportSection(
        key="performance",
        title="Performance",
        metrics={"return": 0.01, "missing": float("nan"), "infinite": float("inf")},
        tables={"summary": pd.DataFrame({"value": [1, 2]})},
        series={"returns": values},
        units={"return": "decimal_return", "returns": "decimal_return"},
        legends={"summary": "Summary", "returns": "Strategy"},
        notes=("All values are direct canonical computations.",),
    )
    return ReportDocument(
        domain="portfolio",
        title="Canonical <report>",
        sections=(section,),
        metadata={"as_of": pd.Timestamp("2025-01-03", tz="UTC")},
        offline_assets={"theme.css": "body { color: #123; }", "chart.js": "window.chartReady = true;"},
    )


def test_report_model_semantic_payload_copies_values_and_rejects_contract_violations() -> None:
    from fincore.report.models import ReportDocument, ReportSection

    document = _report_document()
    payload = document.semantic_payload()

    assert payload["sections"][0]["metrics"]["missing"] == {"type": "float", "value": "nan"}
    assert payload["offline_assets"]["theme.css"]
    assert document.section("performance").series["returns"].iloc[0] == 0.01
    with pytest.raises(KeyError):
        document.section("missing")
    with pytest.raises(ValueError, match="non-empty"):
        ReportSection(key="", title="x")
    with pytest.raises(ValueError, match="unknown values"):
        ReportSection(key="x", title="X", units={"unknown": "ratio"})
    with pytest.raises(TypeError, match="notes"):
        ReportSection(key="x", title="X", notes=("",))
    section = ReportSection(key="x", title="X")
    with pytest.raises(ValueError, match="unique"):
        ReportDocument(domain="x", title="X", sections=(section, section))
    with pytest.raises(TypeError, match="offline asset"):
        ReportDocument(domain="x", title="X", sections=(), offline_assets={"bad": 1})  # type: ignore[dict-item]


def test_portfolio_report_builds_all_optional_sections_without_renderer_recomputation() -> None:
    from fincore.report.portfolio.compute import build_portfolio_report

    returns = _returns()
    positions = pd.DataFrame(
        {
            "AAA": np.linspace(10.0, 40.0, len(returns)),
            "BBB": np.linspace(-5.0, 5.0, len(returns)),
            "cash": np.full(len(returns), 95.0),
        },
        index=returns.index,
    )
    transactions = pd.DataFrame(
        {"amount": [2, -1, 3], "price": [10.0, 12.0, 11.0], "symbol": ["AAA", "BBB", "AAA"]},
        index=pd.DatetimeIndex([returns.index[1], returns.index[1], returns.index[3]]),
    )
    document = build_portfolio_report(
        returns,
        benchmark_returns=returns.mul(0.7),
        positions=positions,
        transactions=transactions,
        rolling_window=4,
        metadata={"run": "coverage"},
    )

    assert [section.key for section in document.sections] == ["performance", "benchmark", "portfolio", "transactions"]
    assert document.section("portfolio").metrics["asset_count"] == 2
    assert document.section("transactions").metrics["symbol_count"] == 2
    assert document.metadata["run"] == "coverage"


@pytest.mark.parametrize(
    ("kwargs", "parameter", "message"),
    [
        ({"period": "bad"}, "period", "must be one of"),
        ({"rolling_window": 0}, "rolling_window", "positive integer"),
        ({"benchmark_returns": pd.Series([0.1])}, "benchmark_returns", "DatetimeIndex"),
        (
            {"positions": pd.DataFrame({"AAA": [1.0]}, index=pd.date_range("2025-01-01", periods=1))},
            "positions",
            "cash column",
        ),
        (
            {"transactions": pd.DataFrame({"amount": [1]}, index=pd.date_range("2025-01-01", periods=1))},
            "transactions",
            "missing required",
        ),
    ],
)
def test_portfolio_report_rejects_input_contract_violations(
    kwargs: dict[str, object], parameter: str, message: str
) -> None:
    from fincore.exceptions import InputContractError
    from fincore.report.portfolio.compute import build_portfolio_report

    with pytest.raises(InputContractError, match=message) as caught:
        build_portfolio_report(_returns(), **kwargs)  # type: ignore[arg-type]
    assert caught.value.parameter == parameter


def test_portfolio_report_converts_math_failures_to_nan_and_rejects_unaligned_benchmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fincore.exceptions import InputContractError
    from fincore.report.portfolio import compute

    def zero_division(*args: object, **kwargs: object) -> float:
        raise ZeroDivisionError

    assert np.isnan(compute._safe_metric(zero_division, 1))
    monkeypatch.setattr(compute, "alpha_beta", lambda *args, **kwargs: (_ for _ in ()).throw(ZeroDivisionError()))
    report = compute.build_portfolio_report(_returns(), benchmark_returns=_returns().mul(0.5), rolling_window=4)
    assert np.isnan(report.section("performance").metrics["alpha"])

    shifted = _returns().copy()
    shifted.index = shifted.index + pd.Timedelta(days=365)
    with pytest.raises(InputContractError, match="share at least one"):
        compute.build_portfolio_report(_returns(), benchmark_returns=shifted, rolling_window=4)


def test_html_renderer_renders_document_assets_and_validates_boundaries(tmp_path: object) -> None:
    from fincore.report.renderers.html import render_html, write_html

    document = _report_document()
    rendered = render_html(document, offline_assets={"extra.css": "h1 { color: red; }"})
    bundle = write_html(document, tmp_path / "nested" / "report.html")  # type: ignore[operator]

    assert "Canonical &lt;report&gt;" in rendered
    assert 'data-offline-asset="theme.css"' in rendered
    assert "N/A" in rendered and "∞" in rendered
    assert bundle.named_artifacts["file"].read_text(encoding="utf-8").startswith("<!doctype html>")
    with pytest.raises(TypeError, match="ReportDocument"):
        render_html(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="offline_assets"):
        render_html(document, offline_assets={"bad": 1})  # type: ignore[dict-item]


def test_pdf_renderer_uses_the_precomputed_html_document_with_a_lazy_playwright_boundary(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fincore.report.renderers import pdf as pdf_renderer

    writes: list[dict[str, object]] = []

    class FakePage:
        def goto(self, uri: str, **kwargs: object) -> None:
            assert uri.startswith("file:")
            writes.append({"goto": uri, **kwargs})

        def pdf(self, **kwargs: object) -> None:
            writes.append(kwargs)
            Path(str(kwargs["path"])).write_bytes(b"%PDF-1.7\n")

    class FakeBrowser:
        def new_page(self, **kwargs: object) -> FakePage:
            writes.append({"new_page": kwargs})
            return FakePage()

        def close(self) -> None:
            writes.append({"closed": True})

    class FakePlaywright:
        chromium = SimpleNamespace(launch=lambda **kwargs: FakeBrowser())

        def __enter__(self) -> FakePlaywright:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        pdf_renderer,
        "load_optional_module",
        lambda *args, **kwargs: SimpleNamespace(sync_playwright=lambda: FakePlaywright()),
    )
    target = tmp_path / "report.pdf"  # type: ignore[operator]
    bundle = pdf_renderer.write_pdf(_report_document(), target)

    assert target.read_bytes().startswith(b"%PDF")
    assert bundle.named_artifacts["file"] == target
    assert any("goto" in call for call in writes)
    assert {"closed": True} in writes


def test_extension_snapshot_and_portfolio_models_reject_invalid_components_and_preserve_value_isolation() -> None:
    from fincore.extensions.snapshot import ExtensionHook, ExtensionSnapshot, RendererRegistration
    from fincore.portfolio.models import ExposureBundle, PortfolioInputs, VolumeExposureBundle
    from fincore.runtime import OperationSpec

    with pytest.raises(TypeError, match="priority"):
        ExtensionHook(event="audit", callable=lambda: None, priority=True)
    with pytest.raises(TypeError, match="callable"):
        RendererRegistration(name="bad", renderer=object())  # type: ignore[arg-type]
    invalid = OperationSpec(
        operation_id="metrics.invalid",
        capability_id="metrics.invalid",
        domain="metrics",
        callable=lambda: None,
    )
    with pytest.raises(ValueError, match="extension namespace"):
        ExtensionSnapshot(operations=(invalid,))

    calls: list[str] = []
    snapshot = ExtensionSnapshot(
        hooks=(ExtensionHook(event="notify", callable=lambda **kwargs: calls.append("notified")),),
        renderers=(RendererRegistration(name="one", renderer=lambda: None),),
    )
    assert snapshot.execute_hooks("notify") is None
    assert calls == ["notified"]
    assert snapshot.renderer("missing") is None

    returns = _returns(2)
    inputs = PortfolioInputs(returns=returns)
    returns.iloc[0] = 9.0
    assert inputs.materialize()["returns"].iloc[0] != 9.0
    with pytest.raises(TypeError, match="pandas Series"):
        PortfolioInputs(returns=[1])  # type: ignore[arg-type]

    positions = pd.DataFrame({"sector": [1.0, 2.0]}, index=returns.index)
    frames = dict.fromkeys(("long", "short", "gross", "net"), positions)
    ExposureBundle(**frames)  # type: ignore[arg-type]
    with pytest.raises(Exception, match="same index"):
        ExposureBundle(positions, positions.iloc[:1], positions, positions)
    values = pd.Series([1.0, 2.0], index=returns.index)
    VolumeExposureBundle(values, values, values)
    with pytest.raises(Exception, match="same index"):
        VolumeExposureBundle(values, values.iloc[:1], values)
