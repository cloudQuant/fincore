"""Contract tests for the manually dispatched 0042-R2 matrix workflow.

The break these tests catch is a workflow that appears cross-platform but
silently rebuilds a wheel in every cell, runs unfrozen tooling, or omits one
of the three required operating systems.  Hosted Actions cannot be exercised
from this test suite, so the real workflow graph and action inputs are loaded
and validated as the public CI contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
BUILD_JOB = "r2-acceptance-build"
CELL_JOB = "r2-acceptance-matrix-cell"
AGGREGATE_JOB = "r2-acceptance-matrix-aggregate"
DIST_ARTIFACT = "r2-acceptance-dist"


def _d0_tooling_commit() -> str:
    manifest = json.loads(
        (REPOSITORY_ROOT / "docs" / "quality" / "0042-r2-d0-bundle-manifest.json").read_text(encoding="utf-8")
    )
    commit = manifest["tooling"]["commit"]
    assert isinstance(commit, str)
    return commit


def _workflow() -> dict[str, Any]:
    payload = yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(payload, dict)
    return payload


def _jobs() -> dict[str, Any]:
    jobs = _workflow().get("jobs")
    assert isinstance(jobs, dict)
    return jobs


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    steps = job.get("steps")
    assert isinstance(steps, list)
    assert all(isinstance(step, dict) for step in steps)
    return steps


def _step_named(job: dict[str, Any], name: str) -> dict[str, Any]:
    for step in _steps(job):
        if step.get("name") == name:
            return step
    raise AssertionError(f"missing workflow step: {name}")


def _run_text(job: dict[str, Any]) -> str:
    return "\n".join(str(step.get("run", "")) for step in _steps(job))


def test_manual_r2_dispatch_requires_immutable_d0_and_tooling_inputs() -> None:
    workflow = _workflow()

    dispatch = workflow["on"]["workflow_dispatch"]
    inputs = dispatch["inputs"]

    assert workflow["permissions"] == {"contents": "read"}
    assert inputs["d0_bundle_url"]["required"] == "true"
    assert inputs["d0_bundle_sha256"]["required"] == "true"
    assert inputs["tooling_ref"]["default"] == _d0_tooling_commit()


def test_manual_r2_dispatch_skips_ordinary_ci_jobs() -> None:
    r2_jobs = {BUILD_JOB, CELL_JOB, AGGREGATE_JOB}

    for name, job in _jobs().items():
        if name not in r2_jobs:
            assert job["if"] == "github.event_name != 'workflow_dispatch'"


def test_r2_build_job_creates_and_uploads_the_only_distribution() -> None:
    build = _jobs()[BUILD_JOB]

    build_run = _run_text(build)
    upload = _step_named(build, "Upload the immutable R2 distribution")

    assert build["runs-on"] == "ubuntu-latest"
    assert "python -m build --outdir ../r2-acceptance-dist" in build_run
    assert "build-artifacts.sha256" in build_run
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["with"]["name"] == DIST_ARTIFACT
    assert upload["with"]["path"] == "r2-acceptance-dist"


def test_r2_matrix_cells_download_the_single_wheel_and_run_the_frozen_contract() -> None:
    cell = _jobs()[CELL_JOB]

    matrix = cell["strategy"]["matrix"]["include"]
    download = _step_named(cell, "Download the immutable R2 distribution")
    frozen_bytes = _step_named(cell, "Normalize frozen tooling bytes on Windows")
    dependencies = _step_named(cell, "Install R2 matrix-cell dependencies")
    diagnostics = _step_named(cell, "Capture failed matrix contract diagnostics")
    upload = _step_named(cell, "Upload R2 matrix-cell evidence")
    cell_run = _run_text(cell)

    assert cell["needs"] == BUILD_JOB
    assert {(entry["os"], entry["runner"], entry["python"]) for entry in matrix} == {
        ("linux", "ubuntu-latest", "3.11.8"),
        ("macos", "macos-latest", "3.11.8"),
        ("windows", "windows-latest", "3.11.8"),
    }
    assert download["uses"] == "actions/download-artifact@v4"
    assert download["with"]["name"] == DIST_ARTIFACT
    assert frozen_bytes["if"] == "runner.os == 'Windows'"
    assert frozen_bytes["shell"] == "bash"
    assert "git -C tooling config core.autocrlf false" in frozen_bytes["run"]
    assert "git -C tooling checkout-index --force --all" in frozen_bytes["run"]
    assert "git -C tooling status --porcelain=v1 --untracked-files=all" in frozen_bytes["run"]
    assert diagnostics["if"] == "failure()"
    assert diagnostics["working-directory"] == "candidate"
    assert "matrix-contract-diagnostic.log" in diagnostics["run"]
    assert "diagnostic only; it does not alter the frozen contract verdict" in diagnostics["run"]
    assert "--import-mode=importlib" in diagnostics["run"]
    assert "--tb=short" in diagnostics["run"]
    assert "python -m build" not in cell_run
    assert 'python -m pip install "$wheel[' in dependencies["run"]
    assert "./candidate[" not in dependencies["run"]
    assert "--gate matrix-cell" in cell_run
    assert "--candidate-wheel" in cell_run
    assert "--expected-bundle" in cell_run
    assert '--os "${{ matrix.os }}"' in cell_run
    assert "--python-full-version \"$(python -c 'import platform; print(platform.python_version())')\"" in cell_run
    assert "--dependency-lane latest" in cell_run
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["with"]["name"] == "r2-acceptance-matrix-cell-${{ matrix.os }}"


def test_r2_aggregate_only_consumes_matrix_cell_artifacts() -> None:
    aggregate = _jobs()[AGGREGATE_JOB]

    download = _step_named(aggregate, "Download all R2 matrix-cell evidence")
    upload = _step_named(aggregate, "Upload R2 matrix aggregate evidence")
    aggregate_run = _run_text(aggregate)

    assert set(aggregate["needs"]) == {BUILD_JOB, CELL_JOB}
    assert download["uses"] == "actions/download-artifact@v4"
    assert download["with"]["pattern"] == "r2-acceptance-matrix-cell-*"
    # Keep each artifact in a distinct directory.  Every cell contains a
    # matrix-cell.json, so merging them would silently overwrite evidence.
    assert download["with"]["merge-multiple"] == "false"
    assert "python -m build" not in aggregate_run
    assert "--gate matrix-aggregate" in aggregate_run
    assert "--require-os linux macos windows" in aggregate_run
    assert "--require-support-window-from-bundle" in aggregate_run
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["with"]["name"] == "r2-acceptance-matrix-aggregate"
