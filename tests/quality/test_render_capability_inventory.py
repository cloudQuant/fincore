"""Regression tests for the capability-inventory generator entry point."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_direct_inventory_script_prefers_its_checkout_source(tmp_path: Path) -> None:
    """A direct script invocation must not import a different installed fincore."""

    repository = tmp_path / "checkout"
    script_dir = repository / "scripts"
    package_dir = repository / "fincore"
    script_dir.mkdir(parents=True)
    package_dir.mkdir()
    shadow_package_dir = tmp_path / "shadow" / "fincore"
    shadow_package_dir.mkdir(parents=True)
    root = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).resolve().parents[2])).resolve()
    shutil.copy2(root / "scripts" / "render_capability_inventory.py", script_dir / "render_capability_inventory.py")
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "capabilities.py").write_text(
        """
from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    id: str
    domain: str
    status: str
    public_path: str
    input_contract: str
    output_contract: str
    docs_path: str


def list_capabilities():
    return [
        Capability(
            id=\"checkout-only\",
            domain=\"test\",
            status=\"stable\",
            public_path=\"fincore.checkout_only\",
            input_contract=\"checkout source\",
            output_contract=\"sentinel\",
            docs_path=\"api/test.md\",
        )
    ]
""".lstrip(),
        encoding="utf-8",
    )
    (shadow_package_dir / "__init__.py").write_text("", encoding="utf-8")
    (shadow_package_dir / "capabilities.py").write_text(
        "def list_capabilities():\n    raise RuntimeError('shadow-capability-sentinel')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path / "shadow"), str(repository)))

    completed = subprocess.run(
        [sys.executable, str(script_dir / "render_capability_inventory.py")],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "wrote" in completed.stdout
    inventory = (repository / "docs" / "quality" / "capability-inventory.md").read_text(encoding="utf-8")
    assert "checkout-only" in inventory
    assert "checkout source" in inventory
