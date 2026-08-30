"""Contracts for the 0042-R2 raw legacy-surface discovery artifact.

The fixture this command produces is deliberately *not* the reconciled
inventory consumed by the capability-baseline capture command.  These tests
exercise it from a separate clean clone so its source provenance is real while
the checkout used to run pytest remains free to contain the test and script
under development.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "collect_0042_r2_surface_discovery.py"
REPOSITORY_ROOT = SCRIPT.parents[1]
FIXTURE = REPOSITORY_ROOT / "tests" / "parity" / "fixtures" / "legacy-surface-discovery-0042-r2.json"

REQUIRED_SOURCE_KINDS = frozenset(
    {
        "metric_registry",
        "workflow_registry",
        "performance_operation_specs",
        "alphalens_function_specs",
        "alphalens_workflow_specs",
        "public_api_snapshot",
        "empyrical_compat_manifest",
        "pyfolio_compat_manifest",
        "alphalens_compat_manifest",
        "capability_registry",
        "distribution_extras",
        "installed_consumer_profiles",
        "pyfolio_class_methods",
    }
)

EXPECTED_ARTIFACT_PATHS = {
    "metric_registry": "fincore/_registry.py",
    "workflow_registry": "fincore/contracts/workflows.py",
    "performance_operation_specs": "fincore/api/builtins.py",
    "alphalens_function_specs": "fincore/contracts/factor_analysis.py",
    "alphalens_workflow_specs": "fincore/contracts/factor_workflows.py",
    "public_api_snapshot": "tests/contracts/fixtures/public-api-0.4.0.dev0.json",
    "empyrical_compat_manifest": "tests/compat/fixtures/empyrical-0.6.0-api.json",
    "pyfolio_compat_manifest": "tests/compat/fixtures/pyfolio-0.9.6-api.json",
    "alphalens_compat_manifest": "tests/compat/fixtures/alphalens-0.4.0-cloudquant-api.json",
    "capability_registry": "fincore/capabilities.py",
    "distribution_extras": "pyproject.toml",
    "installed_consumer_profiles": "scripts/test_installed_wheel.py",
    "pyfolio_class_methods": "fincore/_pyfolio_impl.py",
}

EXPECTED_SOURCE_COUNTS = {
    "metric_registry": 237,
    "workflow_registry": 11,
    "performance_operation_specs": 9,
    "alphalens_function_specs": 61,
    "alphalens_workflow_specs": 7,
    "capability_registry": 25,
    "distribution_extras": 15,
    "installed_consumer_profiles": 9,
    "pyfolio_class_methods": 67,
}

_TEST_REPLACE_REF_BASE = "refs/fincore-0042-r2-test-replace"


def _clone_clean_source(tmp_path: Path, revision: str = "HEAD") -> Path:
    """Create an independent clean checkout at one locally reachable revision."""
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


def _collect(
    source_root: Path, output: Path, *, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=source_root,
        capture_output=True,
        check=False,
        text=True,
        env=environment,
    )


def _load_artifact(source_root: Path, output: Path) -> dict[str, Any]:
    result = _collect(source_root, output)
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def _entry_counts(artifact: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for entry in artifact["entries"]:
        counts[entry["source_kind"]] += 1
    return dict(counts)


def _walk_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_walk_mapping_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_walk_mapping_keys(item) for item in value)) if value else set()
    return set()


def _install_replaced_head(source_root: Path, path: str) -> tuple[str, str, bytes, bytes]:
    """Leave a clean canonical A checkout with an A-to-B replace mapping."""
    commit = subprocess.run(
        ["git", "--no-replace-objects", "rev-parse", "HEAD"],
        cwd=source_root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "--no-replace-objects", "rev-parse", "HEAD^{tree}"],
        cwd=source_root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    source_path = source_root / path
    canonical_payload = source_path.read_bytes()
    source_path.write_bytes(canonical_payload + b"\n# 0042-r2 provenance replacement test\n")
    subprocess.run(["git", "config", "user.email", "surface-discovery@example.invalid"], cwd=source_root, check=True)
    subprocess.run(["git", "config", "user.name", "Surface Discovery"], cwd=source_root, check=True)
    subprocess.run(["git", "add", path], cwd=source_root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "replacement B"], cwd=source_root, check=True)
    replacement_commit = subprocess.run(
        ["git", "--no-replace-objects", "rev-parse", "HEAD"],
        cwd=source_root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    replacement_payload = source_path.read_bytes()
    subprocess.run(["git", "--no-replace-objects", "reset", "--hard", "--quiet", commit], cwd=source_root, check=True)
    subprocess.run(["git", "replace", commit, replacement_commit], cwd=source_root, check=True)
    subprocess.run(
        ["git", "--no-replace-objects", "update-ref", f"{_TEST_REPLACE_REF_BASE}/{commit}", replacement_commit],
        cwd=source_root,
        check=True,
    )
    return commit, tree, canonical_payload, replacement_payload


def _poisoned_git_environment(tmp_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_DIR": str(tmp_path / "poisoned-git-dir"),
            "GIT_WORK_TREE": str(tmp_path / "poisoned-worktree"),
            "GIT_INDEX_FILE": str(tmp_path / "poisoned-index"),
            "GIT_REPLACE_REF_BASE": _TEST_REPLACE_REF_BASE,
            "GIT_CONFIG_GLOBAL": str(tmp_path / "poisoned-global-config"),
            "GIT_CONFIG_NOSYSTEM": "0",
        }
    )
    return environment


def test_collects_deterministic_multisource_raw_facts_from_clean_head(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"

    artifact = _load_artifact(source_root, first_output)
    second = _load_artifact(source_root, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert artifact == second
    assert artifact["schema_version"] == 1
    assert artifact["artifact_type"] == "legacy_surface_discovery"
    assert artifact["discovery_status"] == "partial"
    assert artifact["not_for_d0"] is True
    assert "maintained docs" in artifact["partial_reason"]
    assert "examples" in artifact["partial_reason"]
    assert "benchmarks" in artifact["partial_reason"]
    assert "wheel" in artifact["partial_reason"]
    assert "test-node" in artifact["partial_reason"]
    assert set(artifact["required_source_kinds"]) == REQUIRED_SOURCE_KINDS
    assert artifact["source"]["clean"] is True
    assert len(artifact["source"]["commit"]) == 40
    assert len(artifact["source"]["tree"]) == 40
    assert artifact["entries"]
    assert artifact["source_artifacts"]
    assert artifact["discrepancies"]

    artifact_by_kind = {item["source_kind"]: item for item in artifact["source_artifacts"]}
    assert set(artifact_by_kind) == REQUIRED_SOURCE_KINDS
    assert {kind: item["path"] for kind, item in artifact_by_kind.items()} == EXPECTED_ARTIFACT_PATHS
    assert all(not Path(item["path"]).is_absolute() for item in artifact["source_artifacts"])
    assert all("\\" not in item["path"] for item in artifact["source_artifacts"])
    counts = _entry_counts(artifact)
    assert set(EXPECTED_SOURCE_COUNTS) <= set(counts)
    assert all(counts[kind] == count for kind, count in EXPECTED_SOURCE_COUNTS.items())
    assert counts["empyrical_compat_manifest"] > 0
    assert counts["pyfolio_compat_manifest"] > 0
    assert counts["alphalens_compat_manifest"] > 0

    entries_by_path: dict[str, set[str]] = defaultdict(set)
    for entry in artifact["entries"]:
        assert entry["entry_id"]
        assert entry["source_id"] == entry["source_kind"]
        assert entry["origin"]
        assert entry["surface"]
        assert entry["concept"]
        assert entry["source_locator"]
        locator = entry["source_locator"]
        source_artifact = artifact_by_kind[entry["source_kind"]]
        assert locator["artifact_path"] == source_artifact["path"]
        assert locator["artifact_sha256"] == source_artifact["sha256"]
        public_path = entry["surface"].get("public_path")
        if public_path:
            entries_by_path[public_path].add(entry["source_kind"])

    assert any(len(kinds) > 1 for kinds in entries_by_path.values()), "cross-source facts must not be deduplicated"
    assert any(entry["source_kind"] == "alphalens_function_specs" for entry in artifact["entries"])
    assert any(entry["source_kind"] == "pyfolio_class_methods" for entry in artifact["entries"])
    assert {
        "catalog_projection_not_complete_source",
        "snapshot_paths_not_equivalent_to_catalog_bindings",
        "pyfolio_class_methods_not_workflows",
        "factor_contract_manifest_not_one_to_one",
        "distribution_extras_not_installed_profiles",
    } <= {item["discrepancy_id"] for item in artifact["discrepancies"]}


def test_artifact_hashes_are_initial_head_blob_hashes_and_raw_entries_have_no_decisions(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    output = tmp_path / "surface.json"
    artifact = _load_artifact(source_root, output)
    commit = artifact["source"]["commit"]

    for source_artifact in artifact["source_artifacts"]:
        payload = subprocess.run(
            ["git", "show", f"{commit}:{source_artifact['path']}"],
            cwd=source_root,
            capture_output=True,
            check=True,
        ).stdout
        assert source_artifact["sha256"] == hashlib.sha256(payload).hexdigest()

    forbidden_decision_fields = {"owner", "disposition", "target_operation_id", "oracle"}
    assert not (_walk_mapping_keys(artifact["entries"]) & forbidden_decision_fields)
    serialized = output.read_text(encoding="utf-8")
    assert str(source_root) not in serialized
    assert "generated_at" not in serialized
    assert "timestamp" not in serialized


def test_ignores_replacement_and_environment_redirection_for_canonical_source(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    commit, tree, canonical_payload, replacement_payload = _install_replaced_head(source_root, "fincore/_registry.py")
    output = tmp_path / "surface.json"

    result = _collect(source_root, output, environment=_poisoned_git_environment(tmp_path))

    assert result.returncode == 0, result.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["source"] == {"commit": commit, "tree": tree, "clean": True}
    registry = next(item for item in artifact["source_artifacts"] if item["path"] == "fincore/_registry.py")
    assert registry["sha256"] == hashlib.sha256(canonical_payload).hexdigest()
    assert registry["sha256"] != hashlib.sha256(replacement_payload).hexdigest()


def test_committed_fixture_is_byte_identical_to_its_recorded_clean_source(tmp_path: Path) -> None:
    """The frozen raw fixture must remain reproducible from its own source commit."""
    expected_bytes = FIXTURE.read_bytes()
    fixture = json.loads(expected_bytes)
    assert fixture["artifact_type"] == "legacy_surface_discovery"
    assert fixture["discovery_status"] == "partial"
    assert fixture["not_for_d0"] is True
    recorded_commit = fixture["source"]["commit"]
    assert isinstance(recorded_commit, str) and recorded_commit

    source_root = _clone_clean_source(tmp_path, recorded_commit)
    output = tmp_path / "regenerated.json"

    result = _collect(source_root, output)

    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == expected_bytes


def test_rejects_dirty_source_before_overwriting_output(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    (source_root / "raw-discovery-dirty-marker.txt").write_text("dirty\n", encoding="utf-8")
    output = tmp_path / "surface.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")

    result = _collect(source_root, output)

    assert result.returncode != 0
    assert "clean" in result.stderr.lower()
    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'


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
