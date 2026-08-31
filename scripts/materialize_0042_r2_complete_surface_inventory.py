#!/usr/bin/env python3
"""Materialize the reviewed 0042-R2 complete surface inventory.

This is a deterministic *review materializer*, not a discovery command and
not a D0 verdict.  It joins the two frozen raw discoveries with already
reviewed source records:

* a scoped legacy inventory supplies the individual legacy-operation decision;
* the capability ledger supplies source/wheel scenario evidence for required
  analytical operations;
* module and repository dispositions supply explicit ownership for supporting
  source, documentation, and workflow records.

The materializer is deliberately fail-closed.  It never guesses an operation
from a function name or text token.  An analytical legacy operation must have
an exact ledger record, while the small, enumerated set of retired compatibility
surfaces follows an explicit retirement policy.  Public definitions are
supporting implementation-lineage records: every required observable is bound
through its corresponding legacy raw record and ledger evidence instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA_VERSION = 1
_NON_ASSERTIONS = ["D-TECH", "D0", "installed_wheel_behavior", "legacy_zero"]
_LEGACY_RETIREMENT_SOURCES = frozenset(
    {"capability_registry", "distribution_extras", "installed_consumer_profiles", "public_api_snapshot"}
)
_COMPATIBILITY_ORACLE_PREFIXES = (
    "tests/compat/fixtures/",
    "tests/contracts/fixtures/public-api-",
)
_DOCUMENTATION_PATH_EXCEPTIONS = {"CODE_OF_CONDUCT.md": "docs"}
_SUPPORT_TARGETS = {
    "architecture": ("architecture.surface_migration", "architecture.surface_migration"),
    "attribution": ("attribution.implementation_surface", "attribution.implementation_surface"),
    "data": ("data.implementation_surface", "data.implementation_surface"),
    "docs": ("docs.retarget", "docs.retarget"),
    "extensions": ("extensions.implementation_surface", "extensions.implementation_surface"),
    "factor": ("factor.implementation_surface", "factor.implementation_surface"),
    "metrics": ("metrics.implementation_surface", "metrics.implementation_surface"),
    "optimization": ("optimization.implementation_surface", "optimization.implementation_surface"),
    "packaging": ("packaging.distribution_contract", "packaging.distribution_contract"),
    "performance": ("performance.implementation_surface", "performance.implementation_surface"),
    "portfolio": ("portfolio.implementation_surface", "portfolio.implementation_surface"),
    "quality": ("quality.surface_contract", "quality.surface_contract"),
    "release_engineering": ("release_engineering.contract", "release_engineering.contract"),
    "report": ("report.implementation_surface", "report.implementation_surface"),
    "risk": ("risk.implementation_surface", "risk.implementation_surface"),
    "runtime": ("runtime.implementation_surface", "runtime.implementation_surface"),
    "simulation": ("simulation.implementation_surface", "simulation.implementation_surface"),
    "viz": ("viz.implementation_surface", "viz.implementation_surface"),
}


class CompleteInventoryMaterializationError(ValueError):
    """Raised when frozen sources do not support an explicit full decision."""


class _DuplicateJsonKeyError(ValueError):
    """Raised when a policy-relevant JSON key appears more than once."""


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_document(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_object_without_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as exc:
        raise CompleteInventoryMaterializationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(document, dict):
        raise CompleteInventoryMaterializationError(f"{label} must be a JSON object")
    return document


def _require_entries(document: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries or not all(isinstance(entry, dict) for entry in entries):
        raise CompleteInventoryMaterializationError(f"{label} requires a non-empty object entries list")
    return entries


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompleteInventoryMaterializationError(f"{label} must be a non-empty string")
    return value


def _require_provenance(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"commit", "tree", "clean"}:
        raise CompleteInventoryMaterializationError(f"{label} must contain exact clean Git provenance")
    if (
        not isinstance(value.get("commit"), str)
        or not isinstance(value.get("tree"), str)
        or value.get("clean") is not True
    ):
        raise CompleteInventoryMaterializationError(f"{label} must contain clean Git provenance values")
    return {"commit": value["commit"], "tree": value["tree"], "clean": True}


def _index_by_key(entries: Sequence[Mapping[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        value = _require_string(entry.get(key), f"{label} entry {index} {key}")
        if value in result:
            raise CompleteInventoryMaterializationError(f"{label} has duplicate {key}: {value}")
        result[value] = dict(entry)
    return result


def _ledger_evidence(entry: Mapping[str, Any], capability_id: str) -> tuple[list[str], dict[str, str]]:
    scenarios = entry.get("scenarios")
    if (
        not isinstance(scenarios, list)
        or not scenarios
        or not all(isinstance(scenario, dict) for scenario in scenarios)
    ):
        raise CompleteInventoryMaterializationError(f"ledger capability {capability_id} has no reviewed scenarios")
    scenario_ids = sorted(
        _require_string(scenario.get("scenario_id"), f"ledger capability {capability_id} scenario")
        for scenario in scenarios
    )
    if len(set(scenario_ids)) != len(scenario_ids):
        raise CompleteInventoryMaterializationError(f"ledger capability {capability_id} repeats a scenario_id")
    for scenario in scenarios:
        golden_path = scenario.get("golden_path")
        if isinstance(golden_path, str) and golden_path:
            return scenario_ids, {"golden_path": golden_path}
        oracle_reference = scenario.get("oracle_reference")
        if isinstance(oracle_reference, str) and oracle_reference:
            return scenario_ids, {"oracle_reference": oracle_reference}
    raise CompleteInventoryMaterializationError(f"ledger capability {capability_id} has no golden or oracle evidence")


def _support_decision(
    *,
    raw_entry: Mapping[str, Any],
    source_id: str,
    owner: str,
    disposition: str,
    completion_gate: str,
    rationale: str,
    rule_id: str,
) -> dict[str, Any]:
    try:
        capability_id, target_operation_id = _SUPPORT_TARGETS[owner]
    except KeyError as exc:
        raise CompleteInventoryMaterializationError(
            f"{source_id} entry {raw_entry.get('entry_id')!r} has no explicit support target for owner {owner!r}"
        ) from exc
    return {
        "owner": owner,
        "disposition": disposition,
        "capability_ids": [capability_id],
        "target_operation_id": target_operation_id,
        "scenario_ids": [f"{owner}.surface_disposition"],
        "source_nodeids": [],
        "wheel_nodeids": [],
        "evidence": {"oracle_reference": f"reviewed surface disposition: {rule_id}"},
        "completion_gate": completion_gate,
        "rationale": rationale,
        "rule_id": rule_id,
    }


def _legacy_decision(
    *,
    raw_entry: Mapping[str, Any],
    scoped_entry: Mapping[str, Any],
    ledger_by_operation: Mapping[str, Mapping[str, Any]],
    gap_operations: set[str],
) -> dict[str, Any]:
    source_id = _require_string(raw_entry.get("source_id"), "legacy raw source_id")
    target_operation_id = _require_string(scoped_entry.get("target_operation_id"), "scoped legacy target_operation_id")
    completion_gate = _require_string(scoped_entry.get("completion_gate"), "scoped legacy completion_gate")
    ledger_entry = ledger_by_operation.get(target_operation_id)
    if ledger_entry is not None:
        capability_id = _require_string(ledger_entry.get("capability_id"), "ledger capability_id")
        owner = _require_string(ledger_entry.get("owner"), f"ledger capability {capability_id} owner")
        source_nodeids = ledger_entry.get("source_nodeids")
        wheel_nodeids = ledger_entry.get("wheel_nodeids")
        if (
            not isinstance(source_nodeids, list)
            or not source_nodeids
            or not all(isinstance(item, str) and item for item in source_nodeids)
        ):
            raise CompleteInventoryMaterializationError(f"ledger capability {capability_id} lacks source node evidence")
        if (
            not isinstance(wheel_nodeids, list)
            or not wheel_nodeids
            or not all(isinstance(item, str) and item for item in wheel_nodeids)
        ):
            raise CompleteInventoryMaterializationError(f"ledger capability {capability_id} lacks wheel node evidence")
        scenario_ids, evidence = _ledger_evidence(ledger_entry, capability_id)
        return {
            "owner": owner,
            "disposition": "required",
            "capability_ids": [capability_id],
            "target_operation_id": target_operation_id,
            "scenario_ids": scenario_ids,
            "source_nodeids": sorted(set(source_nodeids)),
            "wheel_nodeids": sorted(set(wheel_nodeids)),
            "evidence": evidence,
            "completion_gate": completion_gate,
            "rationale": "The legacy observable has one reviewed canonical capability with source and wheel evidence.",
            "rule_id": "ledger-required-capability",
        }
    if target_operation_id in gap_operations:
        raise CompleteInventoryMaterializationError(
            f"legacy target {target_operation_id} remains a ledger coverage gap and cannot enter a complete inventory"
        )
    if source_id not in _LEGACY_RETIREMENT_SOURCES:
        raise CompleteInventoryMaterializationError(
            f"legacy target {target_operation_id} has no ledger decision or explicit retirement source"
        )
    owner = _require_string(scoped_entry.get("owner"), "scoped retirement owner")
    return _support_decision(
        raw_entry=raw_entry,
        source_id=source_id,
        owner=owner,
        disposition="retire",
        completion_gate="D-CUTOVER",
        rationale="The record describes a legacy namespace, profile, or distribution alias with no independent 0.5 observable.",
        rule_id="legacy-surface-retirement",
    )


def _union_decision(
    *,
    raw_entry: Mapping[str, Any],
    module_by_path: Mapping[str, Mapping[str, Any]],
    repository_by_path: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_kind = _require_string(raw_entry.get("source_kind"), "surface-union source_kind")
    source = raw_entry.get("source")
    if not isinstance(source, dict):
        raise CompleteInventoryMaterializationError(
            f"surface-union entry {raw_entry.get('entry_id')!r} has no source object"
        )
    artifact_path = _require_string(source.get("artifact_path"), "surface-union artifact_path")
    if source_kind == "public_definition":
        module = module_by_path.get(artifact_path)
        if module is None:
            raise CompleteInventoryMaterializationError(
                f"public definition lacks an exact module disposition: {artifact_path}"
            )
        owner = _require_string(module.get("owner"), f"module disposition owner for {artifact_path}")
        return _support_decision(
            raw_entry=raw_entry,
            source_id="surface_union",
            owner=owner,
            disposition="supporting",
            completion_gate="D-DOMAIN",
            rationale="The definition is implementation lineage for a required legacy observable, not an additional public 0.5 path.",
            rule_id="implementation-lineage",
        )
    if source_kind == "registry":
        module = module_by_path.get(artifact_path)
        if module is None:
            raise CompleteInventoryMaterializationError(f"registry lacks an exact module disposition: {artifact_path}")
        owner = _require_string(module.get("owner"), f"module disposition owner for {artifact_path}")
        return _support_decision(
            raw_entry=raw_entry,
            source_id="surface_union",
            owner=owner,
            disposition="retire",
            completion_gate="D-CUTOVER",
            rationale="The registry structure is a legacy binding mechanism; analytical leaves are separately retained through ledger records.",
            rule_id="registry-retirement",
        )
    if source_kind in {"documentation", "example"}:
        repository = repository_by_path.get(artifact_path)
        if repository is None:
            owner = _DOCUMENTATION_PATH_EXCEPTIONS.get(artifact_path)
            if owner is None:
                raise CompleteInventoryMaterializationError(
                    f"{source_kind} lacks an exact repository disposition: {artifact_path}"
                )
        else:
            owner = _require_string(repository.get("owner"), f"repository disposition owner for {artifact_path}")
        return _support_decision(
            raw_entry=raw_entry,
            source_id="surface_union",
            owner=owner,
            disposition="supporting",
            completion_gate="D-CUTOVER",
            rationale="Maintained user-facing material must be retargeted to the canonical 0.5 surface before cutover.",
            rule_id=f"{source_kind}-retarget",
        )
    if source_kind == "benchmark":
        return _support_decision(
            raw_entry=raw_entry,
            source_id="surface_union",
            owner="quality",
            disposition="supporting",
            completion_gate="D0",
            rationale="The workload is retained as performance evidence and does not define a separate analytical capability.",
            rule_id="performance-workload-support",
        )
    if source_kind in {"extra", "wheel_content"}:
        return _support_decision(
            raw_entry=raw_entry,
            source_id="surface_union",
            owner="packaging",
            disposition="supporting",
            completion_gate="D-CUTOVER",
            rationale="The distribution record is governed by the reviewed 0.5 packaging contract, not by a legacy import surface.",
            rule_id="packaging-contract",
        )
    if source_kind == "manifest":
        if artifact_path.startswith(_COMPATIBILITY_ORACLE_PREFIXES):
            return _support_decision(
                raw_entry=raw_entry,
                source_id="surface_union",
                owner="quality",
                disposition="historical_provenance",
                completion_gate="D-CUTOVER",
                rationale="The compatibility fixture is retained only as frozen migration provenance and must not become a 0.5 facade.",
                rule_id="compatibility-oracle-archive",
            )
        return _support_decision(
            raw_entry=raw_entry,
            source_id="surface_union",
            owner="packaging",
            disposition="supporting",
            completion_gate="D-CUTOVER",
            rationale="The manifest participates in the reviewed 0.5 distribution and release contract.",
            rule_id="manifest-contract",
        )
    raise CompleteInventoryMaterializationError(f"surface-union entry has an unsupported source kind: {source_kind}")


def _decision_entry(
    *,
    source_id: str,
    raw_entry: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    raw_entry_id = _require_string(raw_entry.get("entry_id"), "raw entry_id")
    return {
        "inventory_entry_id": f"{source_id}:{raw_entry_id}",
        "source_id": source_id,
        "raw_entry_id": raw_entry_id,
        "raw_entry_sha256": _canonical_sha256(raw_entry),
        "owner": decision["owner"],
        "disposition": decision["disposition"],
        "capability_ids": decision["capability_ids"],
        "target_operation_id": decision["target_operation_id"],
        "scenario_ids": decision["scenario_ids"],
        "source_nodeids": decision["source_nodeids"],
        "wheel_nodeids": decision["wheel_nodeids"],
        "evidence": decision["evidence"],
        "completion_gate": decision["completion_gate"],
        "rationale": decision["rationale"],
        "rule_id": decision["rule_id"],
    }


def materialize_complete_inventory(
    *,
    legacy_discovery: Path,
    surface_union: Path,
    scoped_inventory: Path,
    ledger: Path,
    module_disposition: Path,
    repository_disposition: Path,
) -> dict[str, Any]:
    """Return a complete reviewed inventory from explicit, frozen source records."""
    legacy = _load_document(legacy_discovery, "legacy discovery")
    union = _load_document(surface_union, "surface union")
    scoped = _load_document(scoped_inventory, "scoped legacy inventory")
    ledger_document = _load_document(ledger, "capability ledger")
    modules = _load_document(module_disposition, "module disposition")
    repository = _load_document(repository_disposition, "repository disposition")

    legacy_provenance = _require_provenance(legacy.get("source"), "legacy discovery provenance")
    union_provenance = _require_provenance(union.get("source_provenance"), "surface union provenance")
    if legacy_provenance != union_provenance:
        raise CompleteInventoryMaterializationError("legacy and surface-union discovery provenance must match exactly")
    legacy_entries = _require_entries(legacy, "legacy discovery")
    union_entries = _require_entries(union, "surface union")
    scoped_by_id = _index_by_key(
        _require_entries(scoped, "scoped legacy inventory"), "legacy_entry_id", "scoped inventory"
    )
    legacy_by_id = _index_by_key(legacy_entries, "entry_id", "legacy discovery")
    if set(scoped_by_id) != set(legacy_by_id):
        raise CompleteInventoryMaterializationError(
            "scoped legacy inventory does not map the exact raw legacy entry set"
        )
    ledger_by_operation = _index_by_key(
        _require_entries(ledger_document, "capability ledger"), "target_operation_id", "capability ledger"
    )
    gap_operations = {
        _require_string(gap.get("capability_id"), "ledger coverage gap capability_id")
        for gap in ledger_document.get("coverage_gaps", [])
        if isinstance(gap, dict)
    }
    module_by_path = _index_by_key(_require_entries(modules, "module disposition"), "path", "module disposition")
    repository_by_path = _index_by_key(
        _require_entries(repository, "repository disposition"), "path", "repository disposition"
    )

    entries = [
        _decision_entry(
            source_id="legacy_surface_discovery",
            raw_entry=raw_entry,
            decision=_legacy_decision(
                raw_entry=raw_entry,
                scoped_entry=scoped_by_id[_require_string(raw_entry.get("entry_id"), "legacy raw entry_id")],
                ledger_by_operation=ledger_by_operation,
                gap_operations=gap_operations,
            ),
        )
        for raw_entry in legacy_entries
    ]
    entries.extend(
        _decision_entry(
            source_id="surface_union",
            raw_entry=raw_entry,
            decision=_union_decision(
                raw_entry=raw_entry,
                module_by_path=module_by_path,
                repository_by_path=repository_by_path,
            ),
        )
        for raw_entry in union_entries
    )
    entries.sort(key=lambda entry: entry["inventory_entry_id"])
    owners = sorted({entry["owner"] for entry in entries})
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "complete_surface_inventory",
        "scope": "complete_legacy_surface_union",
        "decision_status": "complete",
        "does_not_assert": _NON_ASSERTIONS,
        "owners": owners,
        "source_artifacts": [
            {
                "source_id": "legacy_surface_discovery",
                "path": legacy_discovery.name,
                "sha256": _sha256_file(legacy_discovery),
                "entry_count": len(legacy_entries),
                "source_provenance": legacy_provenance,
            },
            {
                "source_id": "surface_union",
                "path": surface_union.name,
                "sha256": _sha256_file(surface_union),
                "entry_count": len(union_entries),
                "source_provenance": union_provenance,
            },
        ],
        "entries": entries,
    }


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-discovery", type=Path, required=True)
    parser.add_argument("--surface-union", type=Path, required=True)
    parser.add_argument("--scoped-inventory", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--module-disposition", type=Path, required=True)
    parser.add_argument("--repository-disposition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        inventory = materialize_complete_inventory(
            legacy_discovery=arguments.legacy_discovery,
            surface_union=arguments.surface_union,
            scoped_inventory=arguments.scoped_inventory,
            ledger=arguments.ledger,
            module_disposition=arguments.module_disposition,
            repository_disposition=arguments.repository_disposition,
        )
        _write_json(arguments.output, inventory)
    except (CompleteInventoryMaterializationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"materialized {len(inventory['entries'])} complete 0042-R2 surface decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
