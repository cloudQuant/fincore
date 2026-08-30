"""Contracts for the raw 0042-R2 module-facts discovery artifact.

The collector must inspect only a clean initial Git HEAD.  These tests use a
separate clone as the source so the test checkout can contain the collector
under development without becoming collector input.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "collect_0042_r2_module_facts.py"
REPOSITORY_ROOT = SCRIPT.parents[1]
FIXTURE = REPOSITORY_ROOT / "tests" / "parity" / "fixtures" / "module-facts-discovery-0042-r2.json"


def _clone_clean_source(tmp_path: Path, revision: str = "HEAD") -> Path:
    """Clone one reachable revision without inheriting its worktree state."""
    expected_head = subprocess.run(
        ["git", "rev-parse", revision],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    source_root = tmp_path / "source"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-local", str(REPOSITORY_ROOT), str(source_root)],
        check=True,
        text=True,
    )
    subprocess.run(["git", "checkout", "--quiet", expected_head], cwd=source_root, check=True, text=True)
    assert not subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=source_root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return source_root


def _collect(source_root: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=source_root,
        capture_output=True,
        check=False,
        text=True,
    )


def _load_artifact(source_root: Path, output: Path) -> dict[str, Any]:
    result = _collect(source_root, output)
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def _head_python_paths(source_root: Path) -> list[str]:
    output = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", "fincore"],
        cwd=source_root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return sorted(path for path in output.splitlines() if path.endswith(".py"))


def _walk_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_walk_mapping_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_walk_mapping_keys(item) for item in value)) if value else set()
    return set()


def test_collects_complete_deterministic_raw_module_facts_from_clean_head(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"

    artifact = _load_artifact(source_root, first_output)
    second = _load_artifact(source_root, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert artifact == second
    assert artifact["schema_version"] == 1
    assert artifact["artifact_type"] == "module_facts_discovery"
    assert artifact["discovery_status"] == "partial"
    assert artifact["not_for_d0"] is True
    assert "disposition" in artifact["partial_reason"].lower()
    assert "docs" in artifact["partial_reason"].lower()
    assert "examples" in artifact["partial_reason"].lower()
    assert "benchmarks" in artifact["partial_reason"].lower()

    provenance = artifact["source_provenance"]
    assert provenance["clean"] is True
    assert len(provenance["commit"]) == 40
    assert len(provenance["tree"]) == 40
    assert artifact["source_archive"]["verified_against_regular_blobs"] is True

    expected_paths = _head_python_paths(source_root)
    modules = artifact["modules"]
    assert artifact["module_count"] == len(expected_paths) == len(modules)
    assert [module["path"] for module in modules] == expected_paths
    assert all(not Path(module["path"]).is_absolute() for module in modules)
    assert all("\\" not in module["path"] and ".." not in Path(module["path"]).parts for module in modules)

    by_path = {module["path"]: module for module in modules}
    assert by_path["fincore/__init__.py"]["module_name"] == "fincore"
    assert by_path["fincore/metrics/__init__.py"]["module_name"] == "fincore.metrics"
    assert by_path["fincore/metrics/basic.py"]["module_name"] == "fincore.metrics.basic"
    assert by_path["fincore/__init__.py"]["is_package"] is True
    assert by_path["fincore/metrics/basic.py"]["is_package"] is False

    for module in modules:
        payload = subprocess.run(
            ["git", "show", f"{provenance['commit']}:{module['path']}"],
            cwd=source_root,
            capture_output=True,
            check=True,
        ).stdout
        assert module["blob_sha256"] == hashlib.sha256(payload).hexdigest()
        assert isinstance(module["ast_facts"]["imports"], list)
        assert isinstance(module["ast_facts"]["type_checking_lines"], list)
        assert isinstance(module["risk_facts"]["dynamic_import_calls"], list)
        assert isinstance(module["risk_facts"]["all_assignments"], list)
        assert isinstance(module["risk_facts"]["module_getattr_lines"], list)
        assert module["static_consumer_paths"] == sorted(module["static_consumer_paths"])
        assert module["static_consumer_count"] == len(module["static_consumer_paths"])
        for import_fact in module["ast_facts"]["imports"]:
            assert import_fact["kind"] in {"import", "from_import"}
            assert import_fact["line"] > 0
            assert import_fact["resolution"]["category"] in {
                "absolute_internal",
                "external_or_unknown",
                "relative_internal",
                "relative_escaped",
            }

    assert by_path["fincore/__init__.py"]["risk_facts"]["module_getattr_lines"]
    assert by_path["fincore/metrics/__init__.py"]["risk_facts"]["module_getattr_lines"]
    assert by_path["fincore/metrics/basic.py"]["risk_facts"]["all_assignments"]
    assert by_path["fincore/metrics/basic.py"]["ast_facts"]["imports"]
    assert any(
        fact["call"] == "__import__" for fact in by_path["fincore/_dispatch.py"]["risk_facts"]["dynamic_import_calls"]
    )
    assert "fincore/api/builtins.py" in by_path["fincore/_registry.py"]["static_consumer_paths"]
    assert any(
        fact["resolution"]["category"] == "relative_internal"
        for module in modules
        for fact in module["ast_facts"]["imports"]
    )
    conditional_exports = by_path["fincore/viz/interactive/__init__.py"]["risk_facts"]["all_assignments"]
    assert any(
        assignment["module_scope"] == "conditional" and "BokehBackend" in assignment["static_values"]
        for assignment in conditional_exports
    )

    forbidden_decision_fields = {
        "decision",
        "disposition",
        "keep",
        "move",
        "delete",
        "owner",
        "oracle",
        "target_operation_id",
    }
    assert not (_walk_mapping_keys(modules) & forbidden_decision_fields)
    serialized = first_output.read_text(encoding="utf-8")
    assert str(source_root) not in serialized
    assert "generated_at" not in _walk_mapping_keys(artifact)
    assert "timestamp" not in _walk_mapping_keys(artifact)


def test_committed_fixture_is_byte_identical_to_its_recorded_clean_source(tmp_path: Path) -> None:
    """The frozen module facts must stay reproducible from their own source commit."""
    expected_bytes = FIXTURE.read_bytes()
    fixture = json.loads(expected_bytes)
    assert fixture["artifact_type"] == "module_facts_discovery"
    assert fixture["discovery_status"] == "partial"
    assert fixture["not_for_d0"] is True
    recorded_commit = fixture["source_provenance"]["commit"]
    assert isinstance(recorded_commit, str) and recorded_commit

    source_root = _clone_clean_source(tmp_path, recorded_commit)
    output = tmp_path / "regenerated.json"

    result = _collect(source_root, output)

    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == expected_bytes


def test_rejects_dirty_source_before_overwriting_output(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    (source_root / "raw-module-facts-dirty-marker.txt").write_text("dirty\n", encoding="utf-8")
    output = tmp_path / "module-facts.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")

    result = _collect(source_root, output)

    assert result.returncode != 0
    assert "clean" in result.stderr.lower()
    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'


def test_rejects_python_symlink_in_initial_head_before_overwriting_output(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    malicious = source_root / "fincore" / "linked_module.py"
    malicious.symlink_to("metrics/basic.py")
    subprocess.run(["git", "config", "user.email", "module-facts@example.invalid"], cwd=source_root, check=True)
    subprocess.run(["git", "config", "user.name", "Module Facts Test"], cwd=source_root, check=True)
    subprocess.run(["git", "add", "fincore/linked_module.py"], cwd=source_root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "test symlink module"], cwd=source_root, check=True)
    assert not subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=source_root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    output = tmp_path / "module-facts.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")

    result = _collect(source_root, output)

    assert result.returncode != 0
    assert "regular git blob" in result.stderr.lower()
    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'


def test_rejects_module_and_package_name_collision_before_overwriting_output(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    colliding_module = source_root / "fincore" / "metrics.py"
    colliding_module.write_text('"""Deliberate module/package collision."""\n', encoding="utf-8")
    subprocess.run(["git", "config", "user.email", "module-facts@example.invalid"], cwd=source_root, check=True)
    subprocess.run(["git", "config", "user.name", "Module Facts Test"], cwd=source_root, check=True)
    subprocess.run(["git", "add", "fincore/metrics.py"], cwd=source_root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "test module collision"], cwd=source_root, check=True)
    output = tmp_path / "module-facts.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")

    result = _collect(source_root, output)

    assert result.returncode != 0
    assert "multiple source paths" in result.stderr.lower()
    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'


def test_rejects_fincore_source_tree_as_output_before_overwriting(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    target = source_root / "fincore" / "metrics" / "basic.py"
    original = target.read_bytes()

    result = _collect(source_root, Path("fincore/metrics/basic.py"))

    assert result.returncode != 0
    assert "source tree" in result.stderr.lower()
    assert target.read_bytes() == original


def test_collector_passes_its_focused_mypy_check() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", str(SCRIPT)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("output_argument", [".git", ".git/HEAD"])
def test_rejects_git_control_directory_as_output_before_overwriting(tmp_path: Path, output_argument: str) -> None:
    source_root = _clone_clean_source(tmp_path)
    git_path = source_root / output_argument
    original = git_path.read_bytes() if git_path.is_file() else None

    result = _collect(source_root, Path(output_argument))

    assert result.returncode != 0
    assert "git control" in result.stderr.lower()
    if original is not None:
        assert git_path.read_bytes() == original
