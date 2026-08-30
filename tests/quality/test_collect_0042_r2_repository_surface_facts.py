"""Contracts for raw 0042-R2 repository-surface facts discovery.

The collector is deliberately narrower than a repository lifecycle decision:
it reads category facts from a separate clean Git checkout and records no
keep/move/delete/retire outcome.  These tests keep the source checkout apart
from the checkout holding the collector so source provenance remains genuine.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

SCRIPT = Path(__file__).parents[2] / "scripts" / "collect_0042_r2_repository_surface_facts.py"
REPOSITORY_ROOT = SCRIPT.parents[1]
FIXTURE = REPOSITORY_ROOT / "tests" / "parity" / "fixtures" / "repository-surface-facts-discovery-0042-r2.json"

EXPECTED_WORKFLOWS = {
    ".github/workflows/ci.yml",
    ".github/workflows/docs.yml",
    ".github/workflows/publish.yml",
    ".github/workflows/test-priority.yml",
}
_TEST_REPLACE_REF_BASE = "refs/fincore-0042-r2-test-replace"
EXPECTED_FROZEN_CATEGORY_COUNTS = {
    "active_maintained_doc": 77,
    "active_workflow": 4,
    "compat_generator_checker": 6,
    "example": 39,
    "historical_provenance_candidate": 144,
    "packaging_release_script": 14,
    "script_candidate": 51,
    "template": 1,
    "type_stub": 1,
}
# This is the corrected scoped baseline: the prior 294-record artifact admitted
# runtime/test paths via a filename heuristic.  Those paths belong to the
# dedicated module- and test-node discovery artifacts, not this repository surface.


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


def _load_collector_module() -> tuple[str, Any]:
    module_name = "fincore_0042_r2_repository_surface_collector_test"
    specification = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module_name, module


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
    subprocess.run(["git", "config", "user.email", "repository-facts@example.invalid"], cwd=source_root, check=True)
    subprocess.run(["git", "config", "user.name", "Repository Facts Test"], cwd=source_root, check=True)
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


def _walk_mapping_keys(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        yield from value
        for nested in value.values():
            yield from _walk_mapping_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_mapping_keys(nested)


def _records_by_path(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["path"]: record for record in artifact["records"]}


def test_collects_deterministic_raw_repository_surface_facts_from_clean_head(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"

    artifact = _load_artifact(source_root, first_output)
    second = _load_artifact(source_root, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert artifact == second
    assert artifact["schema_version"] == 1
    assert artifact["artifact_type"] == "repository_surface_facts_discovery"
    assert artifact["discovery_status"] == "partial"
    assert artifact["not_for_d0"] is True
    assert "decision" in artifact["partial_reason"].lower()
    assert "compatibility" in artifact["partial_reason"].lower()
    assert "scripts/" in artifact["partial_reason"]
    assert "runtime" in artifact["boundaries"]["excluded"]
    assert artifact["source_provenance"]["clean"] is True
    assert len(artifact["source_provenance"]["commit"]) == 40
    assert len(artifact["source_provenance"]["tree"]) == 40
    assert artifact["source_archive"]["verified_against_regular_blobs"] is True
    assert artifact["record_count"] == len(artifact["records"])
    assert artifact["record_count"] == 320
    assert artifact["category_counts"] == EXPECTED_FROZEN_CATEGORY_COUNTS
    assert artifact["records"] == sorted(artifact["records"], key=lambda item: item["path"])

    records_by_path = _records_by_path(artifact)
    assert set(records_by_path) >= EXPECTED_WORKFLOWS
    assert {"pyproject.toml", "MANIFEST.in", "setup.py"} <= set(records_by_path)
    assert {
        "README.md",
        "docs/api.md",
        "mkdocs_docs/index.md",
        ".github/pull_request_template.md",
    } <= set(records_by_path)
    assert {"examples/quick_start.py", "examples/data/positions.csv"} <= set(records_by_path)
    assert "fincore/py.typed" in records_by_path
    assert {
        "scripts/generate_compat_manifest.py",
        "scripts/check_api_diff.py",
        "scripts/generate_alphalens_oracle.py",
        "scripts/check_architecture_convergence.py",
        "scripts/capture_capability_baseline.py",
        "scripts/check_0042_r2_legacy_surface_inventory.py",
        "scripts/profile_workloads.py",
    } <= set(records_by_path)
    assert {
        "CHANGELOG.md",
        "docs/0039-优化完善项目/任务.md",
        "docs/plans/2026-08-30-fincore-0042-r2-breaking-unified-core.md",
        "docs/upstream-provenance.md",
        "docs/迭代计划/0001_添加empyrical然后重构/01_需求文档.md",
        "examples/011_abberation/logs/AbberationStrategy_20260207_162014/run_info.json",
    } <= set(records_by_path)

    for path in EXPECTED_WORKFLOWS:
        assert "active_workflow" in records_by_path[path]["category_tags"]
    tracked_script_paths = set(
        subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", artifact["source_provenance"]["commit"], "--", "scripts"],
            cwd=source_root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.splitlines()
    )
    assert tracked_script_paths <= set(records_by_path)
    for path in tracked_script_paths:
        assert "script_candidate" in records_by_path[path]["category_tags"]
    assert records_by_path["scripts/profile_workloads.py"]["kind"] == "script_candidate"
    assert records_by_path["scripts/check_release_candidate.py"]["kind"] == "packaging_release_script"
    assert records_by_path["scripts/generate_compat_manifest.py"]["kind"] == "compat_generator_checker"
    assert "packaging_release_script" in records_by_path["pyproject.toml"]["category_tags"]
    assert "packaging_release_script" in records_by_path["scripts/check_release_candidate.py"]["category_tags"]
    assert "active_maintained_doc" in records_by_path["README.md"]["category_tags"]
    assert "template" in records_by_path[".github/pull_request_template.md"]["category_tags"]
    assert "example" in records_by_path["examples/quick_start.py"]["category_tags"]
    assert (
        "historical_provenance_candidate"
        not in records_by_path["examples/factor_analysis_quickstart.py"]["category_tags"]
    )
    assert "example" in records_by_path["examples/factor_analysis_quickstart.py"]["category_tags"]
    assert "historical_provenance_candidate" not in records_by_path["mkdocs_docs/api/report.md"]["category_tags"]
    assert "active_maintained_doc" in records_by_path["mkdocs_docs/api/report.md"]["category_tags"]
    assert "type_stub" in records_by_path["fincore/py.typed"]["category_tags"]
    assert "compat_generator_checker" in records_by_path["scripts/generate_compat_manifest.py"]["category_tags"]
    assert "historical_provenance_candidate" in records_by_path["CHANGELOG.md"]["category_tags"]
    assert "historical_provenance_candidate" in records_by_path["docs/0039-优化完善项目/任务.md"]["category_tags"]
    assert (
        "historical_provenance_candidate"
        in records_by_path["docs/plans/2026-08-30-fincore-0042-r2-breaking-unified-core.md"]["category_tags"]
    )
    assert "historical_provenance_candidate" in records_by_path["docs/upstream-provenance.md"]["category_tags"]
    assert (
        "historical_provenance_candidate"
        in records_by_path["docs/迭代计划/0001_添加empyrical然后重构/01_需求文档.md"]["category_tags"]
    )
    assert (
        "historical_provenance_candidate"
        in records_by_path["examples/011_abberation/logs/AbberationStrategy_20260207_162014/run_info.json"][
            "category_tags"
        ]
    )
    assert "active_maintained_doc" not in records_by_path["CHANGELOG.md"]["category_tags"]
    assert "active_maintained_doc" not in records_by_path["docs/0039-优化完善项目/任务.md"]["category_tags"]
    assert "historical_provenance_candidate" not in records_by_path["docs/MIGRATION.md"]["category_tags"]
    assert "active_maintained_doc" in records_by_path["docs/MIGRATION.md"]["category_tags"]
    assert (
        "active_maintained_doc"
        not in records_by_path["docs/plans/2026-08-30-fincore-0042-r2-breaking-unified-core.md"]["category_tags"]
    )
    assert (
        "active_maintained_doc"
        not in records_by_path["docs/迭代计划/0001_添加empyrical然后重构/01_需求文档.md"]["category_tags"]
    )
    assert "fincore/contracts/analysis.py" not in records_by_path
    assert "tests/test_report/test_offline_report.py" not in records_by_path

    source_commit = artifact["source_provenance"]["commit"]
    for path, record in records_by_path.items():
        assert record["kind"] in {
            "active_workflow",
            "packaging_release_script",
            "active_maintained_doc",
            "template",
            "example",
            "type_stub",
            "compat_generator_checker",
            "script_candidate",
            "historical_provenance_candidate",
        }
        assert record["classification_basis"]
        assert record["category_tags"] == sorted(record["category_tags"])
        assert all(not Path(tag).is_absolute() for tag in record["category_tags"])
        payload = subprocess.run(
            ["git", "show", f"{source_commit}:{path}"],
            cwd=source_root,
            capture_output=True,
            check=True,
        ).stdout
        assert record["blob_sha256"] == hashlib.sha256(payload).hexdigest()
        token_facts = record["token_facts"]
        assert set(token_facts) == {"content_kind", "executable_tokens", "import_tokens", "reference_tokens"}
        assert isinstance(token_facts["executable_tokens"], list)
        assert isinstance(token_facts["import_tokens"], list)
        assert isinstance(token_facts["reference_tokens"], list)

    assert records_by_path[".github/workflows/ci.yml"]["token_facts"]["executable_tokens"]
    assert records_by_path["scripts/generate_compat_manifest.py"]["token_facts"]["import_tokens"]
    assert records_by_path["README.md"]["token_facts"]["reference_tokens"]
    assert records_by_path["fincore/py.typed"]["token_facts"]["content_kind"] == "text"
    assert records_by_path["fincore/py.typed"]["token_facts"]["import_tokens"] == []

    forbidden_decision_fields = {
        "decision",
        "disposition",
        "keep",
        "move",
        "delete",
        "retire",
        "owner",
        "oracle",
        "target_operation_id",
    }
    assert not (set(_walk_mapping_keys(artifact)) & forbidden_decision_fields)
    serialized = first_output.read_text(encoding="utf-8")
    assert str(source_root) not in serialized
    assert "generated_at" not in serialized
    assert "timestamp" not in serialized


def test_ignores_replacement_and_environment_redirection_for_canonical_source(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    commit, tree, canonical_payload, replacement_payload = _install_replaced_head(
        source_root, "scripts/check_api_diff.py"
    )
    output = tmp_path / "repository-surface.json"

    result = _collect(source_root, output, environment=_poisoned_git_environment(tmp_path))

    assert result.returncode == 0, result.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["source_provenance"] == {"commit": commit, "tree": tree, "clean": True}
    script = _records_by_path(artifact)["scripts/check_api_diff.py"]
    assert script["blob_sha256"] == hashlib.sha256(canonical_payload).hexdigest()
    assert script["blob_sha256"] != hashlib.sha256(replacement_payload).hexdigest()


def test_fails_closed_when_a_selected_python_script_has_invalid_syntax(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    invalid_script = source_root / "scripts" / "check_api_diff.py"
    invalid_script.write_text("def malformed(:\n", encoding="utf-8")
    subprocess.run(["git", "config", "user.email", "repository-facts@example.invalid"], cwd=source_root, check=True)
    subprocess.run(["git", "config", "user.name", "Repository Facts Test"], cwd=source_root, check=True)
    subprocess.run(["git", "add", "scripts/check_api_diff.py"], cwd=source_root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "test invalid selected Python"], cwd=source_root, check=True)
    output = tmp_path / "repository-surface.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")

    result = _collect(source_root, output)

    assert result.returncode != 0
    assert "cannot parse selected python" in result.stderr.lower()
    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'


def test_collects_literal_and_folded_workflow_run_block_commands(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    workflow = source_root / ".github" / "workflows" / "0042-r2-multiline.yml"
    workflow.write_text(
        """name: multiline-run-token-test
on: [push]
jobs:
  literal:
    runs-on: ubuntu-latest
    steps:
      - run: |+
          echo literal-command
          python -m pytest \\
            --offline
          uses: not-a-workflow-action
        env:
          LEGACY: alphalens
      - run: >+
          folded-command --with-option
          and-more
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "config", "user.email", "repository-facts@example.invalid"], cwd=source_root, check=True)
    subprocess.run(["git", "config", "user.name", "Repository Facts Test"], cwd=source_root, check=True)
    subprocess.run(["git", "add", ".github/workflows/0042-r2-multiline.yml"], cwd=source_root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "test multiline workflow runs"], cwd=source_root, check=True)

    artifact = _load_artifact(source_root, tmp_path / "repository-surface.json")
    tokens = _records_by_path(artifact)[".github/workflows/0042-r2-multiline.yml"]["token_facts"]["executable_tokens"]

    assert {"workflow:run:echo", "workflow:run:python", "workflow:run:folded-command"} <= set(tokens)
    assert "workflow:run:--offline" not in tokens
    assert "workflow:run:env:" not in tokens
    assert "workflow:run:LEGACY:" not in tokens
    assert "workflow:uses:not-a-workflow-action" not in tokens


def test_atomic_write_keeps_validated_parent_descriptor_across_path_swap(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path / "source")
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    protected_parent = tmp_path / "protected"
    protected_parent.mkdir()
    output = output_parent / "repository-surface.json"
    module_name, collector = _load_collector_module()
    target = collector._reject_unsafe_output(source_root, output)
    artifact = collector._collect_artifact(source_root)
    original_parent = tmp_path / "output-original"

    output_parent.rename(original_parent)
    output_parent.symlink_to(protected_parent, target_is_directory=True)
    try:
        collector._atomic_write(target, artifact)
    finally:
        collector._close_output_target(target)
        sys.modules.pop(module_name, None)

    assert (original_parent / output.name).is_file()
    assert not (protected_parent / output.name).exists()


def test_atomic_write_rejects_final_output_symlink_after_initial_validation(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path / "source")
    output = tmp_path / "repository-surface.json"
    protected = tmp_path / "protected.json"
    protected.write_text('{"protected": true}\n', encoding="utf-8")
    module_name, collector = _load_collector_module()
    target = collector._reject_unsafe_output(source_root, output)
    artifact = collector._collect_artifact(source_root)
    output.symlink_to(protected)
    try:
        with pytest.raises(collector.DiscoveryError, match="symbolic link"):
            collector._atomic_write(target, artifact)
    finally:
        collector._close_output_target(target)
        sys.modules.pop(module_name, None)

    assert protected.read_text(encoding="utf-8") == '{"protected": true}\n'
    assert output.is_symlink()


def test_committed_fixture_is_byte_identical_to_its_recorded_clean_source(tmp_path: Path) -> None:
    """The frozen discovery fixture must reproduce from its recorded source commit."""
    expected_bytes = FIXTURE.read_bytes()
    fixture = json.loads(expected_bytes)
    assert fixture["artifact_type"] == "repository_surface_facts_discovery"
    assert fixture["discovery_status"] == "partial"
    assert fixture["not_for_d0"] is True
    assert fixture["record_count"] == 320
    assert fixture["category_counts"] == EXPECTED_FROZEN_CATEGORY_COUNTS
    recorded_commit = fixture["source_provenance"]["commit"]
    assert isinstance(recorded_commit, str) and recorded_commit

    source_root = _clone_clean_source(tmp_path, recorded_commit)
    output = tmp_path / "regenerated.json"

    result = _collect(source_root, output)

    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == expected_bytes


def test_rejects_dirty_source_before_overwriting_output(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    (source_root / "repository-surface-dirty-marker.txt").write_text("dirty\n", encoding="utf-8")
    output = tmp_path / "repository-surface.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")

    result = _collect(source_root, output)

    assert result.returncode != 0
    assert "clean" in result.stderr.lower()
    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'


def test_rejects_selected_symlink_in_initial_head_before_overwriting_output(tmp_path: Path) -> None:
    source_root = _clone_clean_source(tmp_path)
    malicious = source_root / ".github" / "workflows" / "linked.yml"
    malicious.symlink_to("ci.yml")
    subprocess.run(["git", "config", "user.email", "repository-facts@example.invalid"], cwd=source_root, check=True)
    subprocess.run(["git", "config", "user.name", "Repository Facts Test"], cwd=source_root, check=True)
    subprocess.run(["git", "add", ".github/workflows/linked.yml"], cwd=source_root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "test selected symlink"], cwd=source_root, check=True)
    output = tmp_path / "repository-surface.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")

    result = _collect(source_root, output)

    assert result.returncode != 0
    assert "regular git blob" in result.stderr.lower()
    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'


@pytest.mark.parametrize("output_kind", ["symlink", "directory"])
def test_rejects_unsafe_external_output_before_overwriting(tmp_path: Path, output_kind: str) -> None:
    source_root = _clone_clean_source(tmp_path)
    protected = tmp_path / "protected.json"
    protected.write_text('{"previous": true}\n', encoding="utf-8")
    output = tmp_path / "repository-surface.json"
    if output_kind == "symlink":
        output.symlink_to(protected)
    else:
        output.mkdir()

    result = _collect(source_root, output)

    assert result.returncode != 0
    assert "output" in result.stderr.lower()
    assert protected.read_text(encoding="utf-8") == '{"previous": true}\n'
    if output_kind == "symlink":
        assert output.is_symlink()
    else:
        assert output.is_dir()


@pytest.mark.parametrize("output_argument", ["repository-surface.json", ".git", ".git/HEAD"])
def test_rejects_source_worktree_or_git_control_output_before_overwriting(tmp_path: Path, output_argument: str) -> None:
    source_root = _clone_clean_source(tmp_path)
    git_target = source_root / output_argument
    original = git_target.read_bytes() if git_target.is_file() else None

    result = _collect(source_root, Path(output_argument))

    assert result.returncode != 0
    assert "output" in result.stderr.lower()
    if original is not None:
        assert git_target.read_bytes() == original


def test_collector_passes_its_focused_mypy_check() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", str(SCRIPT)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
