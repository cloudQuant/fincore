"""Regression tests for the public-API snapshot entry point."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_direct_snapshot_script_prefers_its_checkout_source(tmp_path: Path) -> None:
    """Runtime ``__all__`` discovery must come from the checkout being scanned."""

    repository = tmp_path / "checkout"
    script_dir = repository / "scripts"
    package_dir = repository / "fincore"
    script_dir.mkdir(parents=True)
    package_dir.mkdir()
    shadow_package_dir = tmp_path / "shadow" / "fincore"
    shadow_package_dir.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    shutil.copy2(root / "scripts" / "snapshot_public_api.py", script_dir / "snapshot_public_api.py")
    (package_dir / "__init__.py").write_text("__all__ = ['checkout_sentinel']\n", encoding="utf-8")
    (shadow_package_dir / "__init__.py").write_text(
        "__all__ = [f'shadow_{number}' for number in range(500)]\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path / "shadow"), str(repository)))

    output = repository / "snapshot.json"
    subprocess.run(
        [sys.executable, str(script_dir / "snapshot_public_api.py"), "--output", str(output)],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["surfaces"]["fincore"]["public_symbols"] == ["checkout_sentinel"]
