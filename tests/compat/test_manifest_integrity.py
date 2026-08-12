from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
GENERATOR = Path(__file__).parents[2] / "scripts" / "generate_compat_manifest.py"


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _assert_portable_provenance(data: dict[str, Any]) -> None:
    serialized = json.dumps(data, sort_keys=True)
    assert "/Users/" not in serialized
    assert "\\Users\\" not in serialized
    for source in data["source_files"]:
        assert not Path(source["path"]).is_absolute()
        assert len(source["sha256"]) == 64


def _assert_compatibility_levels(entry: dict[str, Any]) -> None:
    assert set(entry["compatibility"]) == {"C0", "C1", "C2", "C3", "C4"}


def _assert_reconstructable_signature(signature: str) -> None:
    ast.parse(f"def _f{signature}:\n    pass\n")


def test_empyrical_manifest_is_pinned_and_complete() -> None:
    data = _load("empyrical-0.6.0-api.json")
    assert data["version"] == "0.6.0"
    assert data["commit"] == "74655e974ed2935563820c548c339731f1fe0621"
    assert len(data["public_symbols"]) == 54
    assert len(data["callables"]) == 49
    assert {"calmar_ratio", "beta", "perf_attrib", "DAILY"} <= set(data["public_symbols"])
    assert len(set(data["public_symbols"])) == 54
    assert all(entry["extraction"] == "static_ast" for entry in data["symbols"])
    assert all(entry["reviewed"] is False for entry in data["symbols"])
    assert all(entry["signature"].startswith("(") for entry in data["callables"])
    for entry in data["callables"]:
        _assert_reconstructable_signature(entry["signature"])
    dynamic_review = [entry for entry in data["symbols"] if entry["needs_dynamic_review"]]
    assert len(dynamic_review) == 9
    assert all(entry["factory"].startswith("_create_") for entry in dynamic_review)
    for entry in data["symbols"]:
        _assert_compatibility_levels(entry)
    _assert_portable_provenance(data)


def test_pyfolio_manifest_is_pinned() -> None:
    data = _load("pyfolio-0.9.6-api.json")
    assert data["version"] == "0.9.6"
    assert data["commit"] == "724bbd7dbed9a88bb47e1057f2ca29b3409d8e7a"
    assert "create_full_tear_sheet" in data["compatibility_profile"]
    assert "create_risk_tear_sheet" in data["compatibility_profile"]
    assert len(data["compatibility_profile"]) == 11
    for entry in data["compatibility_profile"].values():
        assert entry["public_path"].startswith("pyfolio.")
        assert isinstance(entry["parameters"], list)
        assert entry["signature"].startswith("(")
        _assert_reconstructable_signature(entry["signature"])
        _assert_compatibility_levels(entry)
    _assert_portable_provenance(data)


def test_flat_api_migration_manifest_is_explicit() -> None:
    data = _load("fincore-flat-api-migrations.json")
    assert data["fincore_version"] == "0.3.0"
    assert data["policy"] == "preserve-0.3.x-flat-api"
    assert data["entries"]
    for entry in data["entries"]:
        assert {
            "symbol",
            "current_target",
            "recommended_target",
            "deprecate_in",
            "remove_or_switch_in",
        } <= set(entry)


def test_generator_default_path_is_ast_only() -> None:
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])
    assert {"empyrical", "pyfolio", "importlib", "inspect"}.isdisjoint(imported)
    assert "sys.path" not in GENERATOR.read_text(encoding="utf-8")
    assert _load("empyrical-0.6.0-api.json")["oracle_verification"] == {
        "reviewed": False,
        "status": "not_run",
    }


def test_full_generator_is_byte_idempotent_when_pinned_roots_are_available(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).parents[4]
    empyrical_root = project_root / "empyrical"
    pyfolio_root = project_root / "pyfolio"
    if not (empyrical_root.is_dir() and pyfolio_root.is_dir()):
        pytest.skip("pinned sibling roots are not available; frozen fixtures remain CI input")
    command = [
        sys.executable,
        GENERATOR.as_posix(),
        "--empyrical-root",
        empyrical_root.as_posix(),
        "--pyfolio-root",
        pyfolio_root.as_posix(),
        "--output",
        tmp_path.as_posix(),
    ]
    subprocess.run(command, check=True)
    first = {path.name: path.read_bytes() for path in sorted(tmp_path.glob("*.json"))}
    subprocess.run(command, check=True)
    second = {path.name: path.read_bytes() for path in sorted(tmp_path.glob("*.json"))}
    assert first == second
    assert set(first) == {
        "empyrical-0.6.0-api.json",
        "fincore-flat-api-migrations.json",
        "pyfolio-0.9.6-api.json",
    }
    for name, contents in first.items():
        assert contents == (FIXTURES / name).read_bytes()
