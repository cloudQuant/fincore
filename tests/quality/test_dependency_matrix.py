"""Dependency-matrix and optional-import probe tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_dependency_matrix import check_installed_versions, load_matrix, probe_import


def test_constraints_cover_each_supported_python_environment() -> None:
    matrix = load_matrix(ROOT / "constraints")

    assert matrix["minimum"]["pandas"]
    assert matrix["latest"]["pandas"]
    assert matrix["minimum"]["numpy"] <= matrix["latest"]["numpy"]


def test_constraints_declare_core_runtime_floors() -> None:
    matrix = load_matrix(ROOT / "constraints")

    for name in ("numpy", "pandas", "scipy", "pytz", "packaging"):
        assert matrix["minimum"][name], f"minimum constraints missing {name}"


def test_minimum_constraints_exclude_known_broken_tls_combination() -> None:
    text = (ROOT / "constraints" / "minimum.txt").read_text(encoding="utf-8")

    assert "pyOpenSSL" in text
    assert "cryptography" in text


def test_yfinance_import_probe_succeeds_with_minimum_constraints() -> None:
    result = probe_import("yfinance", constraints=ROOT / "constraints" / "minimum.txt")

    assert result["success"], f"yfinance import failed: {result.get('error')}"


def test_import_probe_records_failure_for_missing_module() -> None:
    result = probe_import("fincore_definitely_missing_module_xyz")

    assert not result["success"]
    assert result["error"]


def test_minimum_lane_requires_exact_installed_versions() -> None:
    expected = {"numpy": "1.24.0", "pandas": "1.5.3"}

    assert not check_installed_versions(expected, expected, lane="minimum")
    assert check_installed_versions({"numpy": "1.24.1", "pandas": "1.5.3"}, expected, lane="minimum")


def test_latest_lane_accepts_newer_installed_versions_but_not_old_ones() -> None:
    expected = {"numpy": "1.24.0", "pandas": "1.5.3"}

    assert not check_installed_versions({"numpy": "2.0.0", "pandas": "1.5.3"}, expected, lane="latest")
    assert check_installed_versions({"numpy": "1.23.0", "pandas": "1.5.3"}, expected, lane="latest")
