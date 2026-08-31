"""Task 8 contracts for performance-report disclosure and provenance."""

from __future__ import annotations

import json
from pathlib import PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd

from fincore.performance.disclosures import DisclosureContext
from fincore.report import create_strategy_report
from fincore.report.compute import compute_sections
from fincore.report.provenance import ReportProvenance
from fincore.report.render_html import generate_html


def _returns() -> pd.Series:
    index = pd.date_range("2024-01-02", periods=60, freq="B", tz="UTC")
    values = [0.001 if number % 2 == 0 else -0.0005 for number in range(len(index))]
    return pd.Series(values, index=index, name="strategy")


def test_compute_sections_derives_a_complete_honest_default_disclosure() -> None:
    model = compute_sections(_returns(), None, None, None, None, 20)

    disclosure = model["performance_disclosure"]

    assert disclosure["convention"] == "Simple periodic returns; geometrically compounded"
    assert disclosure["return_type"] == "simple"
    assert disclosure["units"] == "decimal return per period"
    assert disclosure["frequency"] == "daily"
    assert disclosure["sample_period"] == "2024-01-02 to 2024-03-25 (60 daily observations)"
    assert disclosure["data_quality"] == "60 finite observations; unique, increasing DatetimeIndex validated"
    assert disclosure["fees"] == "not supplied; caller-defined return series"
    assert disclosure["cashflows"] == "not supplied; no cashflow adjustment applied"
    assert disclosure["benchmark"] == "none supplied"
    assert disclosure["risk_free"] == "not supplied; ratios use documented defaults"
    assert disclosure["annualized"] is True
    assert disclosure["notes"] == ["GIPS-aware disclosure support; not GIPS compliance certification."]


def test_context_uses_its_legacy_defaults_as_complete_caller_assertions() -> None:
    model = compute_sections(
        _returns(),
        None,
        None,
        None,
        None,
        20,
        disclosure_context=DisclosureContext(annualized=False),
    )

    disclosure = model["performance_disclosure"]

    assert disclosure["convention"] == "TWR"
    assert disclosure["fees"] == "gross-of-fees"
    assert disclosure["cashflows"] == "none"
    assert disclosure["return_type"] == "simple"
    assert disclosure["units"] == "decimal return per period"
    assert disclosure["annualized"] is False


def test_html_report_renders_custom_disclosure_as_escaped_structured_content(tmp_path) -> None:
    disclosure = DisclosureContext(
        convention="TWR after external-flow neutralization",
        return_type="simple",
        units="decimal return per period",
        frequency="daily",
        fees="net-of-fees",
        cashflows="timed transaction ledger",
        benchmark="S&P 500 total return",
        risk_free="USD 3M Treasury",
        annualized=False,
        notes=("<script>never execute</script>",),
    )
    output = tmp_path / "disclosure.html"

    generate_html(
        _returns(),
        benchmark_rets=None,
        positions=None,
        transactions=None,
        trades=None,
        title="Disclosure",
        output=str(output),
        rolling_window=20,
        disclosure_context=disclosure,
    )

    html = output.read_text(encoding="utf-8")
    assert 'href="#disclosure"' in html
    assert 'id="disclosure"' in html
    assert "TWR after external-flow neutralization" in html
    assert "S&amp;P 500 total return" in html
    assert "Annualized" in html and "no" in html
    assert "&lt;script&gt;never execute&lt;/script&gt;" in html
    assert "<script>never execute</script>" not in html


def test_report_manifest_records_the_resolved_disclosure_without_raw_inputs(tmp_path) -> None:
    disclosure = DisclosureContext(
        convention="TWR",
        return_type="simple",
        units="decimal return per period",
        frequency="daily",
        fees="net-of-fees",
        cashflows="timed transaction ledger",
        benchmark="custom benchmark",
        risk_free="USD 3M Treasury",
        annualized=False,
        notes=("reviewed by investment committee",),
    )
    result = create_strategy_report(
        _returns(),
        output=str(tmp_path / "report.html"),
        return_result=True,
        audit_manifest=True,
        disclosure_context=disclosure,
    )

    assert result.model is not None
    assert result.manifest_path is not None
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    recorded = manifest["configuration"]["performance_disclosure"]
    assert recorded == result.model["performance_disclosure"]
    assert recorded["sample_period"] == "2024-01-02 to 2024-03-25 (60 daily observations)"
    assert recorded["data_quality"] == "60 finite observations; unique, increasing DatetimeIndex validated"
    assert "0.001" not in result.manifest_path.read_text(encoding="utf-8")


def test_report_manifest_redacts_nested_disclosure_secrets_and_absolute_paths(tmp_path) -> None:
    context = DisclosureContext(
        notes=("api_token=review-secret-123", "/private/client-ledger.csv"),
    )

    result = create_strategy_report(
        _returns(),
        output=str(tmp_path / "report.html"),
        return_result=True,
        audit_manifest=True,
        disclosure_context=context,
    )

    assert result.manifest_path is not None
    serialized = result.manifest_path.read_text(encoding="utf-8")
    recorded = json.loads(serialized)["configuration"]["performance_disclosure"]
    assert "review-secret-123" not in serialized
    assert "api_token" not in serialized
    assert "/private/client-ledger.csv" not in serialized
    assert recorded["notes"] == ["[redacted]"]


def test_report_manifest_omits_windows_and_file_uri_disclosure_paths(tmp_path) -> None:
    unsafe_notes = (
        r"C:\\Users\\alice\\client-ledger.csv",
        "C:/Users/alice/client-ledger.csv",
        "file:///Users/alice/client-ledger.csv",
    )

    for number, unsafe_note in enumerate(unsafe_notes):
        result = create_strategy_report(
            _returns(),
            output=str(tmp_path / f"report-{number}.html"),
            return_result=True,
            audit_manifest=True,
            disclosure_context=DisclosureContext(notes=(unsafe_note,)),
        )

        assert result.manifest_path is not None
        serialized = result.manifest_path.read_text(encoding="utf-8")
        recorded = json.loads(serialized)["configuration"]["performance_disclosure"]
        assert unsafe_note not in serialized
        assert recorded["notes"] == []


def test_provenance_omits_nested_pure_path_values() -> None:
    provenance = ReportProvenance.build(
        code_version="test",
        configuration={
            "performance_disclosure": {
                "paths": {
                    "posix": PurePosixPath("/Users/alice/client-ledger.csv"),
                    "windows": PureWindowsPath(r"C:\Users\alice\client-ledger.csv"),
                }
            }
        },
        inputs={},
    )

    assert provenance.configuration == {"performance_disclosure": {"paths": {}}}


def test_provenance_sanitizes_stringified_unknown_values() -> None:
    class StringifyingPath:
        def __str__(self) -> str:
            return "file:///Users/alice/client-ledger.csv"

    provenance = ReportProvenance.build(
        code_version="test",
        configuration={"opaque": StringifyingPath()},
        inputs={},
    )

    assert provenance.configuration == {}


def test_report_manifest_normalizes_numpy_boolean_disclosure_values(tmp_path) -> None:
    context = DisclosureContext(annualized=np.bool_(False))

    result = create_strategy_report(
        _returns(),
        output=str(tmp_path / "report.html"),
        return_result=True,
        audit_manifest=True,
        disclosure_context=context,
    )

    assert result.manifest_path is not None
    recorded = json.loads(result.manifest_path.read_text(encoding="utf-8"))["configuration"]["performance_disclosure"]
    assert recorded["annualized"] is False


def test_legacy_precomputed_model_derives_disclosure_from_its_own_metadata(tmp_path) -> None:
    legacy_index = pd.date_range("2020-01-02", periods=60, freq="B", tz="UTC")
    legacy_returns = pd.Series(
        [0.001 if number % 2 == 0 else -0.0005 for number in range(len(legacy_index))],
        index=legacy_index,
    )
    model = compute_sections(legacy_returns, None, None, None, None, 20)
    del model["performance_disclosure"]
    output = tmp_path / "legacy-model.html"

    generate_html(
        _returns(),
        benchmark_rets=None,
        positions=None,
        transactions=None,
        trades=None,
        title="Legacy model",
        output=str(output),
        rolling_window=20,
        model=model,
    )

    html = output.read_text(encoding="utf-8")
    assert 'id="disclosure"' in html
    assert "2020-01-02 to 2020-03-25 (60 daily observations)" in html
    assert "2024-01-02 to 2024-03-25" not in html
    assert "legacy precomputed model; calculation convention unavailable" in html
    assert "performance_disclosure" not in model


def test_legacy_precomputed_model_needs_no_raw_inputs_for_default_disclosure(tmp_path) -> None:
    returns = _returns()
    model = compute_sections(returns, None, None, None, None, 20)
    del model["performance_disclosure"]
    output = tmp_path / "legacy-model.html"

    generate_html(
        None,
        benchmark_rets=None,
        positions=None,
        transactions=None,
        trades=None,
        title="Legacy model",
        output=str(output),
        rolling_window=20,
        model=model,
    )

    assert output.exists()
