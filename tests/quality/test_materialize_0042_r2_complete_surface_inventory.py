"""Contracts for the reviewed, complete 0042-R2 surface inventory materializer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
FIXTURES = REPOSITORY_ROOT / "tests" / "parity" / "fixtures"
SCRIPT = REPOSITORY_ROOT / "scripts" / "materialize_0042_r2_complete_surface_inventory.py"
CHECKER = REPOSITORY_ROOT / "scripts" / "check_0042_r2_complete_surface_inventory.py"


def _load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _paths() -> dict[str, Path]:
    return {
        "legacy_discovery": FIXTURES / "legacy-surface-discovery-0042-r2.json",
        "surface_union": FIXTURES / "surface-union-facts-discovery-0042-r2.json",
        "scoped_inventory": FIXTURES / "legacy-surface-inventory-0042-r2.json",
        "ledger": FIXTURES / "capability-ledger-0042-r2.json",
        "module_disposition": FIXTURES / "module-disposition-0042-r2.json",
        "repository_disposition": FIXTURES / "repository-surface-disposition-0042-r2.json",
        "complete_inventory": FIXTURES / "complete-surface-inventory-0042-r2.json",
    }


def test_materializer_reproduces_the_committed_reviewed_inventory() -> None:
    materializer = _load_module(SCRIPT, "fincore_0042_r2_complete_inventory_materializer")
    checker = _load_module(CHECKER, "fincore_0042_r2_complete_inventory_checker")
    paths = _paths()

    result = materializer.materialize_complete_inventory(
        **{key: value for key, value in paths.items() if key != "complete_inventory"}
    )
    committed = json.loads(paths["complete_inventory"].read_text(encoding="utf-8"))

    assert result == committed
    summary = checker.validate_complete_inventory(
        paths["legacy_discovery"], paths["surface_union"], paths["complete_inventory"]
    )
    assert summary["entry_count"] == len(committed["entries"])
    assert summary["unmapped_entries"] == []


def test_materializer_rejects_an_unreviewed_legacy_target(tmp_path: Path) -> None:
    materializer = _load_module(SCRIPT, "fincore_0042_r2_complete_inventory_materializer_reject")
    paths = _paths()
    scoped_inventory = json.loads(paths["scoped_inventory"].read_text(encoding="utf-8"))
    target = next(
        entry for entry in scoped_inventory["entries"] if entry["target_operation_id"] == "metrics.annual_return"
    )
    target["target_operation_id"] = "metrics.unreviewed_operation"
    scoped_copy = tmp_path / "scoped-inventory.json"
    scoped_copy.write_text(json.dumps(scoped_inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(materializer.CompleteInventoryMaterializationError, match="no ledger decision"):
        materializer.materialize_complete_inventory(
            legacy_discovery=paths["legacy_discovery"],
            surface_union=paths["surface_union"],
            scoped_inventory=scoped_copy,
            ledger=paths["ledger"],
            module_disposition=paths["module_disposition"],
            repository_disposition=paths["repository_disposition"],
        )
