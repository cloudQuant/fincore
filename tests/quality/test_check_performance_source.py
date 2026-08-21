"""Regression tests for the direct performance-gate entry point."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_direct_performance_gate_prefers_its_checkout_source(tmp_path: Path) -> None:
    """The performance gate must benchmark its candidate, not an installed package."""

    repository = tmp_path / "checkout"
    script_dir = repository / "scripts"
    api_dir = repository / "fincore" / "api"
    script_dir.mkdir(parents=True)
    api_dir.mkdir(parents=True)
    shadow_api_dir = tmp_path / "shadow" / "fincore" / "api"
    shadow_api_dir.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    shutil.copy2(root / "scripts" / "check_performance.py", script_dir / "check_performance.py")
    (repository / "fincore" / "__init__.py").write_text("", encoding="utf-8")
    (api_dir / "__init__.py").write_text(
        "def build_builtin_catalog():\n    raise RuntimeError('checkout-performance-sentinel')\n",
        encoding="utf-8",
    )
    (api_dir / "invoke.py").write_text("def invoke(*args, **kwargs):\n    return None\n", encoding="utf-8")
    (shadow_api_dir.parent / "__init__.py").write_text("", encoding="utf-8")
    (shadow_api_dir / "__init__.py").write_text(
        "def build_builtin_catalog():\n    raise RuntimeError('shadow-performance-sentinel')\n",
        encoding="utf-8",
    )
    (shadow_api_dir / "invoke.py").write_text("def invoke(*args, **kwargs):\n    return None\n", encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path / "shadow"), str(repository)))

    completed = subprocess.run(
        [sys.executable, str(script_dir / "check_performance.py")],
        cwd=repository,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert "checkout-performance-sentinel" in completed.stderr
