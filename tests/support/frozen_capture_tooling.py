"""Build a clean, committed tooling root for capture integration tests."""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_TOOLING_FILES = (
    "capture_capability_baseline.py",
    "check_0042_r2_complete_surface_inventory.py",
    "check_0042_r2_repository_surface_disposition.py",
)


def create_frozen_capture_tooling_root(root: Path, source_scripts: Path) -> Path:
    """Copy the static capture tools into a clean standalone Git worktree."""
    scripts_root = root / "scripts"
    scripts_root.mkdir(parents=True)
    for filename in _TOOLING_FILES:
        shutil.copy2(source_scripts / filename, scripts_root / filename)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "scripts"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=0042-R2 tooling test",
            "-c",
            "user.email=0042-r2-tooling@example.invalid",
            "commit",
            "-qm",
            "frozen capture tooling",
        ],
        cwd=root,
        check=True,
    )
    return root
