"""Capability-extra and isolated wheel-consumer contracts for Fincore 0.5."""

from __future__ import annotations

import ast
import json
import os
import runpy
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _pyproject_version() -> str:
    with PYPROJECT.open("rb") as fh:
        return str(tomllib.load(fh)["project"]["version"])


def _wheel_script() -> dict[str, object]:
    return runpy.run_path(str(REPO_ROOT / "scripts" / "test_installed_wheel.py"), run_name="wheel_consumer_test")


def test_core_namespace_import_does_not_load_optional_dependencies() -> None:
    optional_roots = _wheel_script()["OPTIONAL_ROOTS"]
    probe = textwrap.dedent(
        """
        import json
        import sys
        import sysconfig
        from pathlib import Path

        root = Path(sys.argv[1]).resolve()
        sys.path[:0] = [str(root), sysconfig.get_paths()["purelib"]]
        import fincore

        namespace = {}
        exec("from fincore import *", namespace)
        print(json.dumps({
            "all": fincore.__all__,
            "optional": [name for name in json.loads(sys.argv[2]) if name in sys.modules],
            "legacy": [name for name in ("Empyrical", "Pyfolio", "sharpe_ratio", "analyze") if name in namespace],
        }, sort_keys=True))
        """
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-S",
            "-E",
            "-c",
            probe,
            str(REPO_ROOT),
            json.dumps(optional_roots),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["optional"] == []
    assert payload["legacy"] == []
    assert tuple(payload["all"]) == _wheel_script()["CANONICAL_ROOT_EXPORTS"]


def test_runtime_version_matches_pyproject() -> None:
    import fincore

    assert Path(fincore.__file__).resolve().is_relative_to(REPO_ROOT)
    assert fincore.__version__ == _pyproject_version()


def test_source_version_resolution_prefers_the_checkout_over_ambient_metadata(monkeypatch) -> None:
    import importlib.metadata

    import fincore

    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "999.999.999")
    assert fincore._resolve_version() == _pyproject_version()


def test_installed_wheel_cli_accepts_direct_capability_profiles(tmp_path: Path) -> None:
    profiles = ("core", "factor-analysis", "visualization", "interactive", "report-pdf", "report-xlsx", "all")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "test_installed_wheel.py"),
            "--dist",
            str(tmp_path),
            "--profiles",
            *profiles,
            "--data-providers",
            "all",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stderr
    assert f"no wheel matching fincore-{_pyproject_version()}-" in proc.stderr


def test_isolated_wheel_consumer_bootstrap_imports_json() -> None:
    namespace = _wheel_script()
    consumer_tree = ast.parse(str(namespace["_CONSUMER"]))
    imported_modules = {
        alias.name for node in ast.walk(consumer_tree) if isinstance(node, ast.Import) for alias in node.names
    }
    assert "json" in imported_modules


def test_required_wheel_profiles_have_explicit_bounded_timeouts() -> None:
    profiles = _wheel_script()["PROFILES"]
    assert isinstance(profiles, dict)
    for name, spec in profiles.items():
        assert 0 < spec["install_timeout"] <= 900, name
        assert 0 < spec["consumer_timeout"] <= 300, name


def test_all_profile_uses_a_real_venv_for_pip_check() -> None:
    profiles = _wheel_script()["PROFILES"]
    assert profiles["all"]["install_mode"] == "venv"
    assert profiles["all"]["pip_check"] is True


def test_pip_check_in_a_venv_detects_an_unsatisfied_installed_requirement(tmp_path: Path) -> None:
    venv_dir = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True, timeout=120)
    python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    purelib = Path(
        subprocess.run(
            [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    )
    metadata = purelib / "fincore_pip_check_regression-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: fincore-pip-check-regression\n"
        "Version: 1.0\n"
        "Requires-Dist: fincore-definitely-missing-test-dependency (==1.0)\n",
        encoding="utf-8",
    )
    result = subprocess.run([python, "-m", "pip", "check"], capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "fincore-definitely-missing-test-dependency" in result.stdout + result.stderr
