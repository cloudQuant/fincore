"""Contracts for raw 0042-R2 pytest test-node discovery."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "collect_0042_r2_test_node_facts.py"
REPOSITORY_ROOT = SCRIPT.parents[1]
FIXTURE = REPOSITORY_ROOT / "tests" / "parity" / "fixtures" / "test-node-facts-discovery-0042-r2.json"


def _clone_clean_source(tmp_path: Path) -> Path:
    expected_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
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


def _selected_test_python_paths(source_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", "tests"],
        cwd=source_root,
        capture_output=True,
        check=True,
        text=True,
    )
    return sorted(
        path for path in result.stdout.splitlines() if path.endswith(".py") and not path.startswith("tests/benchmarks/")
    )


def _walk_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_walk_mapping_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_walk_mapping_keys(item) for item in value)) if value else set()
    return set()


def _load_collector_module() -> Any:
    module_name = "fincore_0042_r2_test_node_facts_under_test"
    specification = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def test_cli_describes_raw_test_node_discovery() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "test-node" in result.stdout


def test_collects_deterministic_raw_test_node_facts_from_clean_head(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    first_output = tmp_path / "first.json"

    artifact = _load_artifact(source_root, first_output)

    assert artifact["schema_version"] == 1
    assert artifact["artifact_type"] == "test_node_facts_discovery"
    assert artifact["discovery_status"] == "partial"
    assert artifact["not_for_d0"] is True
    assert "disposition" in artifact["partial_reason"].lower()
    assert "capability" in artifact["partial_reason"].lower()
    assert "benchmarks" in artifact["partial_reason"].lower()

    provenance = artifact["source_provenance"]
    assert provenance["clean"] is True
    assert len(provenance["commit"]) == 40
    assert len(provenance["tree"]) == 40
    assert artifact["source_archive"]["verified_against_regular_blobs"] is True
    assert artifact["source_archive"]["scope"] == "full_repository"

    collection = artifact["collection"]
    assert collection["status"] == "passed"
    assert collection["collection_errors"] == []
    assert collection["marker_expression"] == "not integration_online"
    assert collection["ignored_paths"] == ["tests/benchmarks"]
    assert collection["argv"][:3] == ["<python>", "-m", "pytest"]
    assert "--collect-only" in collection["argv"]
    assert "--ignore=tests/benchmarks" in collection["argv"]

    expected_paths = _selected_test_python_paths(source_root)
    source_test_blobs = artifact["source_test_blobs"]
    assert artifact["source_test_blob_count"] == len(expected_paths) == len(source_test_blobs)
    assert [item["path"] for item in source_test_blobs] == expected_paths
    blob_sha_by_path = {item["path"]: item["sha256"] for item in source_test_blobs}
    assert all(len(sha256) == 64 for sha256 in blob_sha_by_path.values())
    for path in (expected_paths[0], expected_paths[-1], "tests/compat/empyrical/test_public_api.py"):
        payload = subprocess.run(
            ["git", "show", f"{provenance['commit']}:{path}"],
            cwd=source_root,
            capture_output=True,
            check=True,
        ).stdout
        assert blob_sha_by_path[path] == hashlib.sha256(payload).hexdigest()

    nodes = artifact["nodes"]
    assert artifact["node_count"] == len(nodes) > 0
    assert [item["nodeid"] for item in nodes] == sorted(item["nodeid"] for item in nodes)
    assert len({item["nodeid"] for item in nodes}) == len(nodes)
    assert all(item["test_path"] in blob_sha_by_path for item in nodes)
    assert all(item["test_blob_sha256"] == blob_sha_by_path[item["test_path"]] for item in nodes)
    assert all(not item["test_path"].startswith("tests/benchmarks/") for item in nodes)
    assert all("integration_online" not in item["markers"] for item in nodes)
    assert {item["legacy_family"] for item in nodes} >= {"empyrical", "pyfolio", "alphalens", "other"}
    assert all(item["directory_group"] == "tests" or item["directory_group"].startswith("tests/") for item in nodes)
    assert all(not Path(item["test_path"]).is_absolute() for item in nodes)
    assert all("\\" not in item["test_path"] and ".." not in Path(item["test_path"]).parts for item in nodes)

    group_counts = artifact["group_counts"]
    assert sum(item["count"] for item in group_counts["directory"]) == len(nodes)
    assert sum(item["count"] for item in group_counts["legacy_family"]) == len(nodes)
    assert group_counts["marker"]
    assert {item["group"] for item in group_counts["legacy_family"]} >= {
        "empyrical",
        "pyfolio",
        "alphalens",
        "other",
    }

    forbidden_decision_fields = {
        "decision",
        "disposition",
        "keep",
        "move",
        "delete",
        "owner",
        "oracle",
        "target_operation_id",
        "capability_id",
    }
    assert not (_walk_mapping_keys(artifact) & forbidden_decision_fields)
    serialized = first_output.read_text(encoding="utf-8")
    assert str(source_root) not in serialized
    assert "generated_at" not in _walk_mapping_keys(artifact)
    assert "timestamp" not in _walk_mapping_keys(artifact)


def test_mocked_collection_is_deterministic_and_preserves_raw_group_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collector = _load_collector_module()
    source_root = _clone_clean_source(tmp_path)
    raw_nodes = [
        {
            "nodeid": "tests/quality/test_workflow_integrity.py::test_other",
            "test_path": "tests/quality/test_workflow_integrity.py",
            "markers": [],
        },
        {
            "nodeid": "tests/compat/pyfolio/test_public_api.py::test_pyfolio",
            "test_path": "tests/compat/pyfolio/test_public_api.py",
            "markers": ["p1"],
        },
        {
            "nodeid": "tests/compat/empyrical/test_public_api.py::test_empyrical",
            "test_path": "tests/compat/empyrical/test_public_api.py",
            "markers": ["unit", "p0"],
        },
        {
            "nodeid": "tests/compat/alphalens/test_public_api.py::test_alphalens",
            "test_path": "tests/compat/alphalens/test_public_api.py",
            "markers": ["p2"],
        },
    ]
    monkeypatch.setattr(collector, "_run_pytest_collection", lambda _snapshot, _scratch: raw_nodes)

    first = collector._collect_artifact(source_root)
    second = collector._collect_artifact(source_root)

    assert json.dumps(first, ensure_ascii=False, indent=2, sort_keys=True) == json.dumps(
        second, ensure_ascii=False, indent=2, sort_keys=True
    )
    assert [node["nodeid"] for node in first["nodes"]] == sorted(node["nodeid"] for node in raw_nodes)
    assert {node["legacy_family"] for node in first["nodes"]} == {"empyrical", "pyfolio", "alphalens", "other"}
    assert {item["group"] for item in first["group_counts"]["marker"]} == {"<unmarked>", "p0", "p1", "p2", "unit"}


def test_rejects_dirty_source_before_overwriting_output(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    (source_root / "dirty-test-node-facts-marker.txt").write_text("dirty\n", encoding="utf-8")
    output = tmp_path / "test-node-facts.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")

    result = _collect(source_root, output)

    assert result.returncode != 0
    assert "clean" in result.stderr.lower()
    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'


def test_rejects_output_inside_source_worktree_before_collection(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    output = source_root / "test-node-facts.json"

    result = _collect(source_root, output)

    assert result.returncode != 0
    assert "outside" in result.stderr.lower()
    assert not output.exists()
    assert not subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=source_root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout


def test_rejects_non_regular_test_blob_before_overwriting_output(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    (source_root / "tests" / "linked_0042_r2_test.py").symlink_to("compat/empyrical/test_public_api.py")
    subprocess.run(["git", "config", "user.email", "test-node-facts@example.invalid"], cwd=source_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test Node Facts"], cwd=source_root, check=True)
    subprocess.run(["git", "add", "tests/linked_0042_r2_test.py"], cwd=source_root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "test symlink test input"], cwd=source_root, check=True)
    output = tmp_path / "test-node-facts.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")

    result = _collect(source_root, output)

    assert result.returncode != 0
    assert "regular git blob" in result.stderr.lower()
    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'


def test_fails_closed_when_pytest_collection_reports_an_error(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    (source_root / "tests" / "test_broken_0042_r2_collection.py").write_text(
        "def test_broken(:\n    pass\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "config", "user.email", "test-node-facts@example.invalid"], cwd=source_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test Node Facts"], cwd=source_root, check=True)
    subprocess.run(["git", "add", "tests/test_broken_0042_r2_collection.py"], cwd=source_root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "test failed collection"], cwd=source_root, check=True)
    output = tmp_path / "test-node-facts.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")

    result = _collect(source_root, output)

    assert result.returncode != 0
    assert "pytest collection failed" in result.stderr.lower()
    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'


def test_frozen_fixture_is_raw_partial_not_d0_test_node_evidence() -> None:
    artifact = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert artifact["artifact_type"] == "test_node_facts_discovery"
    assert artifact["discovery_status"] == "partial"
    assert artifact["not_for_d0"] is True
    assert artifact["node_count"] == len(artifact["nodes"]) > 0
    assert artifact["collection"]["status"] == "passed"
    assert artifact["collection"]["collection_errors"] == []
    assert not (_walk_mapping_keys(artifact) & {"owner", "disposition", "capability_id", "target_operation_id"})
