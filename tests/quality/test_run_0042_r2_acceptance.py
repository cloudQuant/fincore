"""Fail-closed contracts for the independent 0042-R2 acceptance runner.

The runner must never sign a gate from candidate-supplied expected values.
Until the D0 bundle and D0_TOOLING_SHA are frozen, every bundle-consuming
gate must BLOCK, and identity errors must be usage failures.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts" / "run_0042_r2_acceptance.py"


def _load_runner_module():
    specification = importlib.util.spec_from_file_location("fincore_0042_r2_runner_test", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), *args],
        cwd=cwd or REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _commit(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=0042-R2 acceptance test",
            "-c",
            "user.email=0042-r2-acceptance@example.invalid",
            "commit",
            "-qm",
            message,
        ],
        cwd=repo,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "candidate"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "symbolic-ref", "HEAD", "refs/heads/main"], cwd=repo, check=True)
    (repo / "README.md").write_text("candidate\n", encoding="utf-8")
    _commit(repo, "candidate base")
    return repo


def test_unknown_gate_is_a_usage_error(tmp_path: Path) -> None:
    result = _run(["--gate", "not-a-gate", "--output-dir", str(tmp_path / "out")])

    assert result.returncode == 2
    assert "unknown gate" in result.stderr
    evidence = json.loads((tmp_path / "out" / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["verdict"] == "BLOCKED"


def test_malformed_d0_bundle_is_a_usage_error(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    bundle = tmp_path / "d0-bundle.json"
    bundle.write_text("{}\n", encoding="utf-8")

    result = _run(
        [
            "--gate",
            "tests",
            "--candidate-root",
            str(repo),
            "--expected-bundle",
            str(bundle),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert result.returncode == 2
    assert "expected-bundle must be a directory" in result.stderr
    evidence = json.loads((tmp_path / "out" / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["verdict"] == "BLOCKED"
    assert evidence["gate"] == "tests"


def test_expected_bundle_inside_the_candidate_tree_is_rejected(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    bundle = repo / "bundle.json"
    bundle.write_text("{}\n", encoding="utf-8")

    result = _run(
        [
            "--gate",
            "parity",
            "--candidate-root",
            str(repo),
            "--expected-bundle",
            str(bundle),
        ]
    )

    assert result.returncode == 2
    assert "outside the candidate tree" in result.stderr


def test_missing_expected_bundle_is_a_usage_error(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)

    result = _run(["--gate", "quality", "--candidate-root", str(repo)])

    assert result.returncode == 2
    assert "--expected-bundle" in result.stderr


def test_evidence_is_bound_to_the_runner_blob(tmp_path: Path) -> None:
    result = _run(["--gate", "not-a-gate", "--output-dir", str(tmp_path / "out")])

    assert result.returncode == 2
    evidence = json.loads((tmp_path / "out" / "evidence.json").read_text(encoding="utf-8"))
    expected_digest = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert evidence["runner"]["runner_blob_sha256"] == expected_digest


def test_evidence_child_passes_for_an_allowlisted_single_parent_child(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    parent = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    docs = repo / "docs" / "quality"
    docs.mkdir(parents=True)
    (docs / "0042-r2-acceptance.md").write_text("# acceptance\n", encoding="utf-8")
    child = _commit(repo, "docs: record acceptance evidence")

    result = _run(
        [
            "--gate",
            "evidence-child",
            "--candidate-root",
            str(repo),
            "--tested-parent",
            parent,
            "--evidence-head",
            child,
            "--allow-path",
            "docs/quality/0042-r2-acceptance.md",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert result.returncode == 0, result.stderr
    evidence = json.loads((tmp_path / "out" / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["verdict"] == "PASS"
    assert evidence["details"]["tested_parent"] == parent
    assert evidence["details"]["evidence_head"] == child


def test_evidence_child_blocks_paths_outside_the_allowlist(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    parent = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    (repo / "fincore").mkdir(exist_ok=True)
    (repo / "fincore" / "sneaked.py").write_text("x = 1\n", encoding="utf-8")
    child = _commit(repo, "docs: record acceptance evidence")

    result = _run(
        [
            "--gate",
            "evidence-child",
            "--candidate-root",
            str(repo),
            "--tested-parent",
            parent,
            "--evidence-head",
            child,
            "--allow-path",
            "docs/quality/0042-r2-acceptance.md",
        ]
    )

    assert result.returncode == 3
    assert "allowlist" in result.stderr


def test_evidence_child_blocks_a_wrong_or_missing_parent(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs" / "0042-r2-acceptance.md").write_text("# acceptance\n", encoding="utf-8")
    child = _commit(repo, "docs: record acceptance evidence")

    result = _run(
        [
            "--gate",
            "evidence-child",
            "--candidate-root",
            str(repo),
            "--tested-parent",
            "f" * 40,
            "--evidence-head",
            child,
            "--allow-path",
            "docs/quality/0042-r2-acceptance.md",
        ]
    )

    assert result.returncode == 3

    result = _run(
        [
            "--gate",
            "evidence-child",
            "--candidate-root",
            str(repo),
            "--evidence-head",
            child,
            "--allow-path",
            "docs/quality/0042-r2-acceptance.md",
        ]
    )

    assert result.returncode == 2
    assert "--tested-parent" in result.stderr


def test_evidence_child_blocks_merge_commits_with_two_parents(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    parent = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    subprocess.run(["git", "checkout", "-qb", "side"], cwd=repo, check=True)
    (repo / "side.txt").write_text("side\n", encoding="utf-8")
    _commit(repo, "side change")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs" / "0042-r2-acceptance.md").write_text("# acceptance\n", encoding="utf-8")
    _commit(repo, "docs: record acceptance evidence")
    merge = (
        subprocess.run(
            ["git", "merge", "-q", "--no-ff", "--no-edit", "side"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
    )

    result = _run(
        [
            "--gate",
            "evidence-child",
            "--candidate-root",
            str(repo),
            "--tested-parent",
            parent,
            "--evidence-head",
            merge,
            "--allow-path",
            "docs/quality/0042-r2-acceptance.md",
        ]
    )

    assert result.returncode == 3
    assert "exactly one parent" in result.stderr


def test_d0_bundle_loader_binds_every_artifact_and_the_tooling_identity(tmp_path: Path) -> None:
    """Expected facts come from the external D0 bundle, never the candidate."""

    runner = _load_runner_module()
    bundle = tmp_path / "d0"
    bundle.mkdir()
    artifacts = {
        "capability_baseline": "capability-baseline.json",
        "architecture_baseline": "architecture-baseline.json",
        "performance_baseline": "performance-baseline.json",
        "quality_baseline": "quality-baseline.json",
    }
    for name, relative in artifacts.items():
        (bundle / relative).write_text(json.dumps({"name": name}), encoding="utf-8")
    source_bundle = bundle / "baseline-source.bundle"
    source_bundle.write_bytes(b"baseline source bundle")
    tooling_identity = {
        "commit": "a" * 40,
        "tree": "b" * 40,
        "runner_blob_sha256": "c" * 64,
    }
    manifest = {
        "artifact_type": "fincore_0042_r2_d0_bundle",
        "schema_version": 1,
        "tooling": {
            "commit": tooling_identity["commit"],
            "tree": tooling_identity["tree"],
            "files": {"scripts/run_0042_r2_acceptance.py": tooling_identity["runner_blob_sha256"]},
        },
        "baseline_source": {
            "commit": "d" * 40,
            "tree": "e" * 40,
            "provisioning": {
                "git_bundle": {
                    "path": source_bundle.name,
                    "sha256": hashlib.sha256(source_bundle.read_bytes()).hexdigest(),
                }
            },
        },
        "artifacts": {
            name: {"path": relative, "sha256": hashlib.sha256((bundle / relative).read_bytes()).hexdigest()}
            for name, relative in artifacts.items()
        },
        "python_support_window": ["3.11.8"],
    }
    (bundle / "d0-bundle-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded = runner._load_d0_bundle(bundle, tooling_identity)

    assert loaded["manifest_sha256"] == hashlib.sha256((bundle / "d0-bundle-manifest.json").read_bytes()).hexdigest()
    assert loaded["baseline_source_bundle"] == source_bundle
    assert loaded["artifacts"]["architecture_baseline"] == bundle / "architecture-baseline.json"


def test_materialized_baseline_cleanup_rejects_an_untrusted_path(tmp_path: Path) -> None:
    """The runner may only remove the scratch directory it created itself."""

    runner = _load_runner_module()
    source_root = tmp_path / "untrusted" / "source"
    source_root.mkdir(parents=True)

    with pytest.raises(runner.RunnerBlockedError, match="runner-owned"):
        runner._cleanup_materialized_source(source_root)


def test_baseline_relative_file_normalizes_a_symlinked_root() -> None:
    """A materialized baseline may use a lexical temporary path such as /tmp."""

    runner = _load_runner_module()
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        baseline_root = Path(directory)
        ledger = baseline_root / "ledger.json"
        ledger.write_text("{}\n", encoding="utf-8")

        assert runner._baseline_relative_file(baseline_root, "ledger.json") == ledger.resolve()


def test_architecture_validation_accepts_the_frozen_checker_status_contract() -> None:
    """The detached runner consumes the checker's documented lowercase status."""

    runner = _load_runner_module()

    assert runner._architecture_validation_passed({"status": "passed"})
    assert not runner._architecture_validation_passed({"status": "failed"})
    assert not runner._architecture_validation_passed({"verdict": "PASS"})
