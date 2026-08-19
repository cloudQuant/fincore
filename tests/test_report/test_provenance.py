"""Audit-manifest tests for enhanced reports."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from fincore.report import create_strategy_report
from fincore.report.provenance import SCHEMA_VERSION, ReportProvenance

if TYPE_CHECKING:
    from pathlib import Path


def _returns(n: int = 60) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    values = [(0.001 if i % 2 == 0 else -0.0007) + (i % 7) * 1e-5 for i in range(n)]
    return pd.Series(values, index=idx, name="strategy")


def test_audit_manifest_contains_hashes_not_raw_returns(tmp_path: Path) -> None:
    result = create_strategy_report(
        _returns(),
        output=str(tmp_path / "report.html"),
        return_result=True,
        audit_manifest=True,
    )

    assert result.manifest_path is not None
    assert result.manifest_path.is_file()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["inputs"]["returns"]["sha256"]
    assert len(manifest["inputs"]["returns"]["sha256"]) == 64
    # The manifest never copies raw return values into the artifact.
    assert "0.001" not in result.manifest_path.read_text()


def test_audit_manifest_records_schema_commit_and_dependencies(tmp_path: Path) -> None:
    result = create_strategy_report(
        _returns(),
        output=str(tmp_path / "report.html"),
        return_result=True,
        audit_manifest=True,
    )

    manifest = json.loads(result.manifest_path.read_text())

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["code_commit"]
    assert manifest["code_version"]
    assert manifest["dependencies"]["pandas"]
    assert manifest["configuration"]["rolling_window"] == 63


def test_audit_manifest_hashes_optional_inputs(tmp_path: Path) -> None:
    returns = _returns()
    benchmark = _returns() * 0.8

    result = create_strategy_report(
        returns,
        benchmark_rets=benchmark,
        output=str(tmp_path / "report.html"),
        return_result=True,
        audit_manifest=True,
    )

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["inputs"]["returns"]["sha256"]
    assert manifest["inputs"]["benchmark_rets"]["sha256"]
    assert manifest["inputs"]["benchmark_rets"]["length"] == len(benchmark)


def test_default_report_behavior_is_unchanged(tmp_path: Path) -> None:
    """Without return_result + audit_manifest, no sidecar is written."""
    out = tmp_path / "plain.html"

    result = create_strategy_report(_returns(), output=str(out))

    assert result == str(out)
    assert not out.with_suffix(".manifest.json").exists()


def test_return_result_without_audit_manifest_has_no_manifest(tmp_path: Path) -> None:
    result = create_strategy_report(_returns(), output=str(tmp_path / "report.html"), return_result=True)

    assert result.manifest_path is None


def test_report_provenance_never_leaks_credentials() -> None:
    provenance = ReportProvenance.build(
        code_version="0.3.0",
        configuration={"api_key": "secret-value", "rolling_window": 63},
        inputs={"returns": pd.Series([0.01, -0.01])},
    )

    payload = json.dumps(provenance.to_dict())

    assert "secret-value" not in payload
    assert "0.01" not in payload
    assert "api_key" not in payload


def test_sanitize_configuration_strips_absolute_and_relative_paths() -> None:
    provenance = ReportProvenance.build(
        code_version="0.3.0",
        configuration={
            "output": "/tmp/report.html",
            "home": "~/data",
            "windows_path": "\\temp\\out.txt",
            "keep_me": 42,
        },
        inputs={},
    )
    payload = provenance.to_dict()
    assert "output" not in payload["configuration"]
    assert "home" not in payload["configuration"]
    assert "windows_path" not in payload["configuration"]
    assert payload["configuration"]["keep_me"] == 42


def test_git_commit_falls_back_to_unknown_when_git_missing(monkeypatch) -> None:
    import subprocess

    def boom(*args, **kwargs):
        raise OSError("git not installed")

    monkeypatch.setattr(subprocess, "run", boom)
    provenance = ReportProvenance.build(code_version="0.3.0", configuration={}, inputs={})
    assert provenance.code_commit == "unknown"


def test_input_record_handles_dataframe_and_scalar() -> None:
    provenance = ReportProvenance.build(
        code_version="0.3.0",
        configuration={},
        inputs={
            "frame": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            "scalar": 7,
            "none_value": None,
        },
    )
    payload = provenance.to_dict()
    assert payload["inputs"]["frame"]["kind"] == "dataframe"
    assert payload["inputs"]["frame"]["rows"] == 3
    assert payload["inputs"]["frame"]["columns"] == 2
    assert payload["inputs"]["frame"]["sha256"]
    assert payload["inputs"]["scalar"]["kind"] == "int"
    assert "none_value" not in payload["inputs"]


def test_input_record_series_includes_time_bounds() -> None:
    idx = pd.date_range("2024-01-01", periods=3)
    provenance = ReportProvenance.build(
        code_version="0.3.0",
        configuration={},
        inputs={"returns": pd.Series([0.01, -0.01, 0.02], index=idx)},
    )
    payload = provenance.to_dict()
    rec = payload["inputs"]["returns"]
    assert rec["kind"] == "series"
    assert rec["start"] == "2024-01-01 00:00:00"
    assert rec["end"] == "2024-01-03 00:00:00"
