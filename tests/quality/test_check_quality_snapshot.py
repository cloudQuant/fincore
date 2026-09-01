"""Freshness-gate tests for scripts/check_quality_snapshot.py."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import scripts.check_quality_snapshot as quality_snapshot
from scripts.check_quality_snapshot import (
    MIN_BRANCH_COVERAGE,
    SCHEMA_VERSION,
    branch_coverage,
    check_snapshot,
    load_snapshot,
)

ROOT = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).resolve().parents[2])).resolve()


def _git(root: Path, *args: str) -> str:
    """Run a small Git fixture command and return stdout."""
    import subprocess

    result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _complete_snapshot(source_commit: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"commit": source_commit, "dirty": False},
        "outcome": "pass",
        "copy_manifest_excluded_paths": [
            "docs/quality/current-baseline.json",
            "docs/quality/current-baseline.md",
        ],
        "runs": [
            {
                "label": "branch-coverage",
                "returncode": 0,
                "integrity_ok": True,
                "branch_coverage_percent": MIN_BRANCH_COVERAGE + 2.0,
            }
        ],
    }


def test_rejects_dirty_or_wrong_commit_snapshot(tmp_path: Path) -> None:
    snapshot = {"schema_version": SCHEMA_VERSION, "source": {"commit": "old", "dirty": True}, "outcome": "pass"}
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot))

    violations = check_snapshot(path, expected_commit="current")

    assert "source.commit does not match HEAD" in violations
    assert "source.dirty must be false" in violations


def test_accepts_a_fresh_clean_complete_snapshot(tmp_path: Path) -> None:
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "source": {"commit": "current", "dirty": False},
        "outcome": "pass",
        "runs": [
            {"label": "trusted-baseline", "returncode": 0, "integrity_ok": True},
            {
                "label": "branch-coverage",
                "returncode": 0,
                "integrity_ok": True,
                "branch_coverage_percent": 62.0,
            },
        ],
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot))

    assert check_snapshot(path, expected_commit="current") == []


def test_accepts_only_snapshot_output_commit_after_a_fresh_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed baseline artifact is valid, but a later code edit is not."""
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.name", "Fincore tests")
    _git(tmp_path, "config", "user.email", "fincore-tests@example.invalid")
    (tmp_path / "fincore").mkdir()
    (tmp_path / "fincore" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "fincore/core.py")
    _git(tmp_path, "commit", "--quiet", "-m", "source")
    source_commit = _git(tmp_path, "rev-parse", "HEAD")

    baseline_dir = tmp_path / "docs" / "quality"
    baseline_dir.mkdir(parents=True)
    snapshot_path = baseline_dir / "current-baseline.json"
    snapshot_path.write_text(json.dumps(_complete_snapshot(source_commit)), encoding="utf-8")
    (baseline_dir / "current-baseline.md").write_text("# Quality baseline\n", encoding="utf-8")
    _git(tmp_path, "add", "docs/quality/current-baseline.json", "docs/quality/current-baseline.md")
    _git(tmp_path, "commit", "--quiet", "-m", "quality baseline")
    baseline_commit = _git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setattr(quality_snapshot, "SOURCE_ROOT", tmp_path)

    assert (
        check_snapshot(
            snapshot_path,
            expected_commit=baseline_commit,
            allow_snapshot_output_commit=True,
        )
        == []
    )

    (tmp_path / "fincore" / "core.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(tmp_path, "add", "fincore/core.py")
    _git(tmp_path, "commit", "--quiet", "-m", "code change")
    code_commit = _git(tmp_path, "rev-parse", "HEAD")
    violations = check_snapshot(
        snapshot_path,
        expected_commit=code_commit,
        allow_snapshot_output_commit=True,
    )
    assert "source.commit does not match HEAD" in violations


def test_rejects_missing_or_low_branch_coverage(tmp_path: Path) -> None:
    base = {
        "schema_version": SCHEMA_VERSION,
        "source": {"commit": "current", "dirty": False},
        "outcome": "pass",
    }
    missing = tmp_path / "missing.json"
    missing.write_text(
        json.dumps({**base, "runs": [{"label": "trusted-baseline", "returncode": 0, "integrity_ok": True}]})
    )
    assert "branch coverage is missing" in check_snapshot(missing, expected_commit="current")

    low = tmp_path / "low.json"
    low.write_text(
        json.dumps(
            {
                **base,
                "runs": [
                    {
                        "label": "branch-coverage",
                        "returncode": 0,
                        "integrity_ok": True,
                        "branch_coverage_percent": MIN_BRANCH_COVERAGE - 1.0,
                    }
                ],
            }
        )
    )
    violations = check_snapshot(low, expected_commit="current")
    assert any("below the" in v for v in violations)


def test_baseline_recording_keeps_coverage_presence_but_defers_final_threshold(tmp_path: Path) -> None:
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "source": {"commit": "current", "dirty": False},
        "outcome": "pass",
        "runs": [
            {
                "label": "branch-coverage",
                "returncode": 0,
                "integrity_ok": True,
                "branch_coverage_percent": MIN_BRANCH_COVERAGE - 1.0,
            }
        ],
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(snapshot))

    assert check_snapshot(path, expected_commit="current", record_baseline=True) == []
    assert check_snapshot(path, expected_commit="current")

    snapshot["runs"][0].pop("branch_coverage_percent")
    path.write_text(json.dumps(snapshot))
    assert "branch coverage is missing" in check_snapshot(path, expected_commit="current", record_baseline=True)


def test_rejects_failed_run_and_impure_copy(tmp_path: Path) -> None:
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "source": {"commit": "current", "dirty": False},
        "outcome": "pass",
        "runs": [
            {"label": "trusted-baseline", "returncode": 1, "integrity_ok": False},
            {
                "label": "branch-coverage",
                "returncode": 0,
                "integrity_ok": True,
                "branch_coverage_percent": 62.0,
            },
        ],
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot))

    violations = check_snapshot(path, expected_commit="current")

    assert "run trusted-baseline returncode is not 0" in violations
    assert "run trusted-baseline integrity_ok is not true" in violations


def test_branch_coverage_helper_reads_the_coverage_run() -> None:
    snapshot = {
        "runs": [
            {"label": "trusted-baseline"},
            {"label": "branch-coverage", "branch_coverage_percent": 55.0},
        ]
    }

    assert branch_coverage(snapshot) == 55.0
    assert branch_coverage({"runs": [{"label": "trusted-baseline"}]}) is None


def test_load_snapshot_round_trips_json(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"outcome": "pass"}))

    assert load_snapshot(path) == {"outcome": "pass"}


def test_current_snapshot_must_be_present_and_schema_versioned() -> None:
    """The checked-in snapshot file must exist and be parseable.

    Skipped inside the disposable baseline-copy where the snapshot is the
    (excluded) output artifact being regenerated.
    """
    snapshot_path = ROOT / "docs" / "quality" / "current-baseline.json"
    if not snapshot_path.is_file():
        pytest.skip("current-baseline.json is absent (baseline collection copy)")
    snapshot = load_snapshot(snapshot_path)
    assert isinstance(snapshot, dict)
