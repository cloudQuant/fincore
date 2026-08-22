"""Integrity-gate tests for scripts/check_workflow_integrity.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.check_workflow_integrity import WORKFLOWS_DIR, check_workflow

if TYPE_CHECKING:
    from pathlib import Path


def test_detects_duplicate_mapping_key(tmp_path: Path) -> None:
    wf = tmp_path / "dup.yml"
    wf.write_text("jobs:\n  a:\n    runs-on: ubuntu-latest\n  a:\n    runs-on: ubuntu-latest\n")

    violations = check_workflow(wf)

    assert any("duplicate mapping key" in v for v in violations)


def test_detects_missing_needs_reference(tmp_path: Path) -> None:
    wf = tmp_path / "needs.yml"
    wf.write_text("jobs:\n  a:\n    runs-on: ubuntu-latest\n    needs: [b]\n")

    violations = check_workflow(wf)

    assert any("missing job" in v for v in violations)


def test_detects_unsupported_gh_run_download_workflow_flag(tmp_path: Path) -> None:
    """Candidate retrieval must select a run id before downloading its artifact."""
    wf = tmp_path / "invalid-download.yml"
    wf.write_text(
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: |\n"
        "          gh run download --name dist --workflow ci.yml --dir dist/\n"
    )

    violations = check_workflow(wf)

    assert any("gh run download does not support --workflow" in v for v in violations)


def test_valid_workflow_passes(tmp_path: Path) -> None:
    wf = tmp_path / "ok.yml"
    wf.write_text("jobs:\n  a:\n    runs-on: ubuntu-latest\n  b:\n    runs-on: ubuntu-latest\n    needs: [a]\n")

    assert check_workflow(wf) == []


def test_checked_in_workflows_are_valid() -> None:
    for path in WORKFLOWS_DIR.glob("*.yml"):
        assert check_workflow(path) == [], f"{path.name} has violations"
