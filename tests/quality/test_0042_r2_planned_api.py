"""Fail-closed contracts for the planned fincore 0.5.0 public API document.

The planned API freezes the structural cutover contract from plan sections
0.2, 2.1, and 2.2: one root shape, one namespace set, and the legacy
surfaces that must stop resolving.  Per-symbol planned snapshots are added by
the domain tranches before Task 8 compares actual against planned.
"""

from __future__ import annotations

import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
FIXTURES = REPOSITORY_ROOT / "tests" / "parity" / "fixtures"
PLANNED_API = FIXTURES / "planned-api-0.5.0.json"
MODULE_DISPOSITION = FIXTURES / "module-disposition-0042-r2.json"

_REQUIRED_NON_ASSERTIONS = frozenset({"D0", "D-TECH", "installed_wheel_behavior", "legacy_zero"})
_OWNERS = frozenset(
    {
        "attribution",
        "data",
        "extensions",
        "factor",
        "metrics",
        "optimization",
        "packaging",
        "performance",
        "portfolio",
        "report",
        "risk",
        "runtime",
        "simulation",
        "viz",
    }
)
_ENTRY_MODEL_IDS = frozenset({"domain_function", "runtime_execution", "domain_workflow"})


def _load_planned() -> dict:
    assert PLANNED_API.is_file(), "committed planned API fixture is missing"
    return json.loads(PLANNED_API.read_text(encoding="utf-8"))


def test_planned_api_header_is_scoped_and_targets_0_5_0() -> None:
    planned = _load_planned()

    assert planned["schema_version"] == 1
    assert planned["artifact_type"] == "planned_public_api"
    assert planned["target_version"] == "0.5.0"
    assert planned["decision_status"] == "scoped"
    assert planned["not_for_d0"] is True
    assert set(planned["does_not_assert"]) >= _REQUIRED_NON_ASSERTIONS


def test_entry_models_freeze_the_three_allowed_call_models() -> None:
    planned = _load_planned()
    models = planned["entry_models"]

    assert {model["model_id"] for model in models} == _ENTRY_MODEL_IDS
    for model in models:
        assert model["rule"].strip(), model["model_id"]


def test_root_shape_is_namespace_only_without_flat_callables() -> None:
    planned = _load_planned()
    root = planned["root_shape"]

    assert root["namespace_only"] is True
    assert set(root["allowed_symbols"]) == {"__version__", "errors"}
    assert root["rule"].strip()


def test_target_namespaces_are_disjoint_from_removed_surfaces() -> None:
    planned = _load_planned()
    namespaces = planned["target_namespaces"]
    removed = set(planned["removed_surfaces"])

    assert "fincore.runtime" in namespaces
    assert "fincore.exceptions" in namespaces
    for surface, spec in namespaces.items():
        assert surface.startswith("fincore."), surface
        assert surface not in removed, surface
        assert spec["owner"] in _OWNERS, surface
        assert spec["role"].strip(), surface


def test_removed_surfaces_match_the_module_disposition_decisions() -> None:
    planned = _load_planned()
    disposition = json.loads(MODULE_DISPOSITION.read_text(encoding="utf-8"))
    by_path = {entry["path"]: entry for entry in disposition["entries"]}

    for surface in planned["removed_surfaces"]:
        relative = surface.replace(".", "/")
        covered = [
            entry
            for path, entry in by_path.items()
            if path == f"{relative}.py" or path.startswith(f"{relative}/")
        ]
        assert covered, f"removed surface {surface} has no module disposition rows"
        for entry in covered:
            keeps_legacy_path = entry["disposition"] == "keep" and entry["target_path"] == entry["path"]
            assert not keeps_legacy_path, (
                f"removed surface {surface} keeps legacy path {entry['path']} in place"
            )


def test_source_contract_binds_the_plan_and_disposition_documents() -> None:
    planned = _load_planned()
    contract = planned["source_contract"]

    assert contract["module_disposition_path"] == MODULE_DISPOSITION.name
    assert contract["plan_path"] == "docs/plans/2026-08-30-fincore-0042-r2-breaking-unified-core.md"
    assert set(contract["plan_sections"]) == {"0.2", "2.1", "2.2"}
    assert (REPOSITORY_ROOT / contract["plan_path"]).is_file()
