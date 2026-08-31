"""Contracts for frozen 0042-R2 tools that execute a separate candidate tree."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module():
    script = Path(__file__).parents[2] / "scripts" / "_0042_r2_tooling.py"
    specification = importlib.util.spec_from_file_location("fincore_0042_r2_tooling_test", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_source_root_defaults_to_the_tooling_checkout_and_honors_the_explicit_candidate_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    tooling_root = tmp_path / "tooling"
    candidate_root = tmp_path / "candidate"
    tooling_root.mkdir()
    candidate_root.mkdir()

    monkeypatch.delenv(module.SOURCE_ROOT_ENV, raising=False)
    assert module.resolve_source_root(tooling_root) == tooling_root.resolve()

    monkeypatch.setenv(module.SOURCE_ROOT_ENV, str(candidate_root))
    assert module.resolve_source_root(tooling_root) == candidate_root.resolve()


def test_source_root_rejects_a_missing_candidate_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setenv(module.SOURCE_ROOT_ENV, str(tmp_path / "missing"))

    with pytest.raises(ValueError, match=module.SOURCE_ROOT_ENV):
        module.resolve_source_root(tmp_path)


def test_isolated_quality_checker_uses_the_explicit_candidate_git_head(tmp_path: Path) -> None:
    """A frozen script must not default its commit check to the tooling tree."""

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=candidate, check=True)
    (candidate / "README.md").write_text("candidate\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=candidate, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=0042-R2 tooling test",
            "-c",
            "user.email=0042-r2-tooling@example.invalid",
            "commit",
            "-qm",
            "candidate",
        ],
        cwd=candidate,
        check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=candidate, capture_output=True, text=True, check=True
    ).stdout.strip()
    snapshot = tmp_path / "quality.json"
    snapshot.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": {"commit": head, "dirty": False},
                "outcome": "pass",
                "runs": [
                    {
                        "label": "branch-coverage",
                        "returncode": 0,
                        "integrity_ok": True,
                        "branch_coverage_percent": 60.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    environment = {**os.environ, "FINCORE_0042R2_SOURCE_ROOT": str(candidate)}
    checker = Path(__file__).parents[2] / "scripts" / "check_quality_snapshot.py"

    result = subprocess.run(
        [sys.executable, "-I", str(checker), "--snapshot", str(snapshot)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
