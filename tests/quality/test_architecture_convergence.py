"""Contract tests for the fail-closed 0042-R2 architecture checker."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


SCRIPT = Path(__file__).parents[2] / "scripts" / "check_architecture_convergence.py"


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, check=False, text=True)


def _make_clean_source(tmp_path: Path) -> Path:
    """Create a small, committed source tree with measurable architecture facts."""
    source_root = tmp_path / "source"
    package = source_root / "samplepkg"
    package.mkdir(parents=True)
    (source_root / "pyproject.toml").write_text(
        """[project]\nname = \"samplepkg\"\nversion = \"0.0.0\"\n\n[project.optional-dependencies]\nviz = [\"optlib>=1\"]\n""",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text("from . import first\n", encoding="utf-8")
    (package / "first.py").write_text(
        """try:\n    import optlib\nexcept ImportError:\n    import optlib as fallback_optlib\nelse:\n    import optlib as else_optlib\nfinally:\n    import optlib as final_optlib\n\nfrom . import second\n\ndef distinctive(value):\n    intermediate = value + 1\n    return intermediate * 2\n\ndef duplicate_one(value):\n    temporary = value + 42\n    return temporary\n""",
        encoding="utf-8",
    )
    (package / "second.py").write_text(
        """import optlib\nfrom .first import duplicate_one\n\ndef duplicate_two(item):\n    other_name = item + 9\n    return other_name\n""",
        encoding="utf-8",
    )
    for command in (
        ["git", "init", "--quiet"],
        ["git", "config", "user.email", "architecture@example.invalid"],
        ["git", "config", "user.name", "Architecture Checker"],
        ["git", "add", "."],
        ["git", "commit", "--quiet", "-m", "test source"],
    ):
        completed = _run(command, cwd=source_root)
        assert completed.returncode == 0, completed.stderr
    return source_root


def _collect(source_root: Path, output: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(source_root),
            "--package",
            "samplepkg",
            "--capture",
            str(output),
            *arguments,
        ],
        cwd=source_root,
    )


def _freeze_baseline(captured: dict[str, Any]) -> dict[str, Any]:
    baseline = copy.deepcopy(captured)
    metrics = baseline["measurements"]["summary"]
    baseline["baseline_state"] = "frozen"
    baseline["verdict"] = "architecture_baseline"
    baseline["thresholds"] = {
        "max_duplicate_body_occurrences": metrics["duplicate_body_occurrences"],
        "max_implementation_fingerprint_count": metrics["implementation_fingerprint_count"],
        "max_internal_cycle_count": metrics["internal_cycle_count"],
        "max_logical_loc": metrics["logical_loc"],
        "max_optional_import_leakage_count": metrics["optional_import_leakage_count"],
        "max_physical_loc": metrics["physical_loc"],
    }
    return baseline


def test_capture_is_deterministic_and_collects_all_required_measurement_facts(tmp_path: Path) -> None:
    source_root = _make_clean_source(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_result = _collect(source_root, first)
    second_result = _collect(source_root, second)

    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr
    assert first.read_bytes() == second.read_bytes()
    artifact = json.loads(first.read_text(encoding="utf-8"))

    assert artifact["schema_version"] == 1
    assert artifact["artifact_type"] == "fincore_0042_r2_architecture_measurement"
    assert artifact["baseline_state"] == "captured"
    assert artifact["verdict"] == "measurement_only"
    assert artifact["package"] == "samplepkg"
    assert artifact["source_provenance"]["clean"] is True
    assert len(artifact["source_provenance"]["commit"]) == 40
    assert artifact["source_provenance"]["tree"]
    assert "timestamp" not in first.read_text(encoding="utf-8")
    assert str(source_root) not in first.read_text(encoding="utf-8")

    measurements = artifact["measurements"]
    assert measurements["summary"]["physical_loc"] > 0
    assert measurements["summary"]["logical_loc"] > 0
    assert measurements["summary"]["internal_cycle_count"] == 1
    assert measurements["summary"]["optional_import_leakage_count"] == 4
    assert measurements["summary"]["duplicate_body_occurrences"] >= 1
    assert [item["path"] for item in measurements["files"]] == [
        "samplepkg/__init__.py",
        "samplepkg/first.py",
        "samplepkg/second.py",
    ]
    assert measurements["optional_import_policy"]["effective_module_roots"] == ["optlib"]
    assert {fact["path"] for fact in measurements["optional_import_leakage"]} == {
        "samplepkg/first.py",
        "samplepkg/second.py",
    }
    assert len([fact for fact in measurements["optional_imports"] if fact["guarded"]]) == 1
    assert all(
        "samplepkg" in edge["from"] and "samplepkg" in edge["to"]
        for edge in measurements["internal_import_graph"]["edges"]
    )
    assert measurements["internal_import_graph"]["cycles"] == [["samplepkg.first", "samplepkg.second"]]

    duplicate_groups = measurements["normalized_ast_duplication"]["groups"]
    assert any(
        {occurrence["qualname"] for occurrence in group["occurrences"]} == {"duplicate_one", "duplicate_two"}
        for group in duplicate_groups
    )
    assert {fact["qualname"] for fact in measurements["implementation_fingerprints"]} >= {
        "distinctive",
        "duplicate_one",
        "duplicate_two",
    }


def test_rejects_dirty_source_and_output_inside_source_without_overwriting(tmp_path: Path) -> None:
    source_root = _make_clean_source(tmp_path)
    output = tmp_path / "architecture.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")
    (source_root / "dirty-marker.txt").write_text("dirty\n", encoding="utf-8")

    dirty_result = _collect(source_root, output)

    assert dirty_result.returncode != 0
    assert "clean" in dirty_result.stderr.lower()
    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'

    (source_root / "dirty-marker.txt").unlink()
    inside_result = _collect(source_root, source_root / "architecture.json")

    assert inside_result.returncode != 0
    assert "outside" in inside_result.stderr.lower()
    assert not (source_root / "architecture.json").exists()


def test_baseline_validation_fails_closed_for_missing_pending_platform_schema_and_threshold_errors(
    tmp_path: Path,
) -> None:
    source_root = _make_clean_source(tmp_path)
    captured_path = tmp_path / "captured.json"
    capture_result = _collect(source_root, captured_path)
    assert capture_result.returncode == 0, capture_result.stderr
    captured = json.loads(captured_path.read_text(encoding="utf-8"))
    baseline = _freeze_baseline(captured)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    validated_path = tmp_path / "validated.json"
    validated_result = _collect(source_root, validated_path, "--baseline", str(baseline_path))

    assert validated_result.returncode == 0, validated_result.stderr
    assert json.loads(validated_path.read_text(encoding="utf-8"))["baseline_validation"]["status"] == "passed"

    missing = _collect(source_root, tmp_path / "missing.json", "--baseline", str(tmp_path / "absent.json"))
    assert missing.returncode != 0
    assert "baseline" in missing.stderr.lower()

    pending = _freeze_baseline(captured)
    pending["baseline_state"] = "pending"
    baseline_path.write_text(json.dumps(pending), encoding="utf-8")
    pending_result = _collect(source_root, tmp_path / "pending.json", "--baseline", str(baseline_path))
    assert pending_result.returncode != 0
    assert "frozen" in pending_result.stderr.lower()

    platform_mismatch = _freeze_baseline(captured)
    platform_mismatch["source_provenance"]["platform"]["system"] = "different-platform"
    baseline_path.write_text(json.dumps(platform_mismatch), encoding="utf-8")
    platform_result = _collect(source_root, tmp_path / "platform.json", "--baseline", str(baseline_path))
    assert platform_result.returncode != 0
    assert "platform" in platform_result.stderr.lower()

    schema_mismatch = _freeze_baseline(captured)
    schema_mismatch["schema_version"] = 999
    baseline_path.write_text(json.dumps(schema_mismatch), encoding="utf-8")
    schema_result = _collect(source_root, tmp_path / "schema.json", "--baseline", str(baseline_path))
    assert schema_result.returncode != 0
    assert "schema" in schema_result.stderr.lower()

    missing_measurements = _freeze_baseline(captured)
    del missing_measurements["measurements"]
    baseline_path.write_text(json.dumps(missing_measurements), encoding="utf-8")
    missing_measurements_result = _collect(
        source_root,
        tmp_path / "missing-measurements.json",
        "--baseline",
        str(baseline_path),
    )
    assert missing_measurements_result.returncode != 0
    assert "measurements" in missing_measurements_result.stderr.lower()

    missing_tool_provenance = _freeze_baseline(captured)
    del missing_tool_provenance["tool_provenance"]
    baseline_path.write_text(json.dumps(missing_tool_provenance), encoding="utf-8")
    missing_tool_result = _collect(
        source_root,
        tmp_path / "missing-tool-provenance.json",
        "--baseline",
        str(baseline_path),
    )
    assert missing_tool_result.returncode != 0
    assert "tool_provenance" in missing_tool_result.stderr.lower()

    tooling_mismatch = _freeze_baseline(captured)
    tooling_mismatch["tool_provenance"]["script_sha256"] = "0" * 64
    baseline_path.write_text(json.dumps(tooling_mismatch), encoding="utf-8")
    tooling_mismatch_result = _collect(
        source_root,
        tmp_path / "tooling-mismatch.json",
        "--baseline",
        str(baseline_path),
    )
    assert tooling_mismatch_result.returncode != 0
    assert "tooling" in tooling_mismatch_result.stderr.lower()

    threshold_failure = _freeze_baseline(captured)
    threshold_failure["thresholds"]["max_logical_loc"] = 0
    baseline_path.write_text(json.dumps(threshold_failure), encoding="utf-8")
    threshold_result = _collect(source_root, tmp_path / "threshold.json", "--baseline", str(baseline_path))
    assert threshold_result.returncode != 0
    assert "logical_loc" in threshold_result.stderr


def test_strict_flags_require_explicit_evidence_instead_of_claiming_task8_results(tmp_path: Path) -> None:
    source_root = _make_clean_source(tmp_path)
    captured_path = tmp_path / "captured.json"
    assert _collect(source_root, captured_path).returncode == 0
    baseline = _freeze_baseline(json.loads(captured_path.read_text(encoding="utf-8")))
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    no_cycles = _collect(
        source_root, tmp_path / "no-cycles.json", "--baseline", str(baseline_path), "--require-no-cycles"
    )
    legacy_zero = _collect(
        source_root,
        tmp_path / "legacy-zero.json",
        "--baseline",
        str(baseline_path),
        "--require-legacy-zero",
    )

    assert no_cycles.returncode != 0
    assert "cycle" in no_cycles.stderr.lower()
    assert legacy_zero.returncode != 0
    assert "legacy" in legacy_zero.stderr.lower()
