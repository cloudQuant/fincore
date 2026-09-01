#!/usr/bin/env python3
"""Validate a complete, reviewed 0042-R2 legacy-surface inventory.

The interim ``legacy_surface_inventory`` is intentionally scoped and cannot
enter D0 capture.  This validator is the separate full-union boundary: it
binds every raw entry from the legacy discovery and from a complete-input
surface-union discovery to exactly one review decision.  Success proves only
the inventory linkage; it is explicitly not a D0, D-TECH, installed-wheel, or
legacy-zero verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40,64}$")
_OWNER_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
_CAPABILITY_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_OPERATION_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_ALLOWED_DISPOSITIONS = frozenset({"required", "supporting", "retire", "historical_provenance"})
_ALLOWED_GATES = frozenset({"D0", "D-RUNTIME", "D-DOMAIN", "D-CUTOVER"})
_REQUIRED_NON_ASSERTIONS = frozenset({"D0", "D-TECH", "installed_wheel_behavior", "legacy_zero"})
_REQUIRED_UNION_KINDS = frozenset(
    {
        "public_definition",
        "registry",
        "manifest",
        "documentation",
        "example",
        "benchmark",
        "extra",
        "wheel_content",
    }
)
_FORBIDDEN_ASSERTION_FIELDS = frozenset(
    {
        "assertions",
        "does_assert",
        "d0",
        "d0_status",
        "d_tech",
        "d_tech_status",
        "final",
        "final_status",
        "final_verdict",
        "legacy_zero",
        "passed",
        "release_status",
        "verdict",
        "verdicts",
    }
)
_NORMALIZED_FORBIDDEN_ASSERTION_FIELDS = frozenset(
    re.sub(r"[^a-z0-9]+", "", field.casefold()) for field in _FORBIDDEN_ASSERTION_FIELDS
)
_EXPECTED_INVENTORY_FIELDS = frozenset(
    {
        "inventory_entry_id",
        "source_id",
        "raw_entry_id",
        "raw_entry_sha256",
        "owner",
        "disposition",
        "capability_ids",
        "target_operation_id",
        "scenario_ids",
        "source_nodeids",
        "wheel_nodeids",
        "evidence",
        "completion_gate",
        "rationale",
        "rule_id",
    }
)


class CompleteInventoryValidationError(ValueError):
    """Raised when a complete surface inventory cannot prove exact coverage."""


class _DuplicateJsonKeyError(ValueError):
    """Raised when JSON repeats a policy-relevant object key."""


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def canonical_sha256(value: object) -> str:
    """Hash JSON-compatible input with a deterministic, unambiguous encoding."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_regular_file(path: Path, label: str) -> bytes:
    try:
        expected_metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise CompleteInventoryValidationError(f"cannot safely inspect {label}: {exc}") from exc
    if not stat.S_ISREG(expected_metadata.st_mode):
        raise CompleteInventoryValidationError(f"{label} must be a regular file, not a symbolic link: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CompleteInventoryValidationError(f"{label} must be a regular file, not a symbolic link: {path}")
        if (metadata.st_dev, metadata.st_ino) != (expected_metadata.st_dev, expected_metadata.st_ino):
            raise CompleteInventoryValidationError(f"{label} changed identity while opening: {path}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            return stream.read()
    except OSError as exc:
        raise CompleteInventoryValidationError(f"cannot safely read {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _load_document(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular_file(path, label)
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_json_object_without_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as exc:
        raise CompleteInventoryValidationError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompleteInventoryValidationError(f"{label} must be a JSON object")
    return value, payload


def _require_string(value: object, field: str, subject: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompleteInventoryValidationError(f"{subject} requires non-empty {field}")
    return value


def _require_sha256(value: object, field: str, subject: str) -> str:
    result = _require_string(value, field, subject)
    if not _SHA256.fullmatch(result):
        raise CompleteInventoryValidationError(f"{subject} {field} must be a lowercase SHA256")
    return result


def _require_exact_int(value: object, field: str, subject: str) -> int:
    if type(value) is not int:
        raise CompleteInventoryValidationError(f"{subject} {field} must be an integer")
    return value


def _require_sorted_strings(value: object, field: str, subject: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "possibly empty " if allow_empty else "non-empty "
        raise CompleteInventoryValidationError(f"{subject} requires a {qualifier}{field} list")
    if not all(isinstance(item, str) and item for item in value):
        raise CompleteInventoryValidationError(f"{subject} {field} must contain non-empty strings")
    if value != sorted(set(value)):
        raise CompleteInventoryValidationError(f"{subject} {field} must be sorted and unique")
    return value


def _require_repository_filename(value: object, field: str, subject: str) -> str:
    result = _require_string(value, field, subject)
    pure_path = PurePosixPath(result)
    if (
        result != pure_path.name
        or pure_path.is_absolute()
        or "\\" in result
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise CompleteInventoryValidationError(f"{subject} {field} must be one safe file name")
    return result


def _normalize_field_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold()) if isinstance(value, str) else ""


def _contains_assertion_field(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_normalize_field_key(key) in _NORMALIZED_FORBIDDEN_ASSERTION_FIELDS for key in value) or any(
            _contains_assertion_field(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_assertion_field(item) for item in value)
    return False


def _validate_provenance(value: object, subject: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"commit", "tree", "clean"}:
        raise CompleteInventoryValidationError(
            f"{subject} source provenance must contain exactly commit, tree, and clean"
        )
    commit = _require_string(value.get("commit"), "commit", subject)
    tree = _require_string(value.get("tree"), "tree", subject)
    if not _GIT_OBJECT_ID.fullmatch(commit) or not _GIT_OBJECT_ID.fullmatch(tree) or value.get("clean") is not True:
        raise CompleteInventoryValidationError(f"{subject} source provenance must be a clean Git commit/tree")
    return {"commit": commit, "tree": tree, "clean": True}


def _extract_raw_entries(document: Mapping[str, Any], source_id: str, subject: str) -> dict[str, dict[str, Any]]:
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        raise CompleteInventoryValidationError(f"{subject} entries must be a non-empty list")
    records: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CompleteInventoryValidationError(f"{subject} entry {index} must be an object")
        entry_id = _require_string(entry.get("entry_id"), "entry_id", f"{subject} entry {index}")
        if entry_id in records:
            raise CompleteInventoryValidationError(f"{subject} contains duplicate entry_id: {entry_id}")
        records[entry_id] = entry
    return records


def _validate_legacy_discovery(document: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if document.get("schema_version") != SCHEMA_VERSION or document.get("artifact_type") != "legacy_surface_discovery":
        raise CompleteInventoryValidationError("legacy discovery must be a schema-v1 legacy_surface_discovery artifact")
    if document.get("discovery_status") != "partial" or document.get("not_for_d0") is not True:
        raise CompleteInventoryValidationError("legacy discovery must retain its partial not_for_d0 raw boundary")
    provenance = _validate_provenance(document.get("source"), "legacy discovery")
    return _extract_raw_entries(document, "legacy_surface_discovery", "legacy discovery"), provenance


def _validate_surface_union(document: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("artifact_type") != "surface_union_facts_discovery"
    ):
        raise CompleteInventoryValidationError(
            "surface union must be a schema-v1 surface_union_facts_discovery artifact"
        )
    if document.get("discovery_status") != "complete" or document.get("not_for_d0") is not True:
        raise CompleteInventoryValidationError("surface union must be a complete raw not_for_d0 artifact")
    if _contains_assertion_field(document):
        raise CompleteInventoryValidationError("surface union must not assert D0, final, or legacy-zero status")
    _require_sorted_strings(document.get("does_not_assert"), "does_not_assert", "surface union")
    if not set(document["does_not_assert"]) >= _REQUIRED_NON_ASSERTIONS:
        raise CompleteInventoryValidationError("surface union does_not_assert is incomplete")
    provenance = _validate_provenance(document.get("source_provenance"), "surface union")
    records = _extract_raw_entries(document, "surface_union", "surface union")
    if _require_exact_int(document.get("entry_count"), "entry_count", "surface union") != len(records):
        raise CompleteInventoryValidationError("surface union entry_count does not match entries")
    if _require_sha256(
        document.get("canonical_entries_sha256"), "canonical_entries_sha256", "surface union"
    ) != canonical_sha256(document["entries"]):
        raise CompleteInventoryValidationError("surface union canonical_entries_sha256 does not match entries")
    kinds: set[str] = set()
    for entry_id, entry in records.items():
        if set(entry) != {"entry_id", "source_kind", "source"}:
            raise CompleteInventoryValidationError(f"surface union entry {entry_id} must be raw facts only")
        source_kind = _require_string(entry.get("source_kind"), "source_kind", f"surface union entry {entry_id}")
        source = entry.get("source")
        if not isinstance(source, dict):
            raise CompleteInventoryValidationError(f"surface union entry {entry_id} source must be an object")
        _require_string(source.get("artifact_path"), "artifact_path", f"surface union entry {entry_id}")
        _require_sha256(source.get("artifact_sha256"), "artifact_sha256", f"surface union entry {entry_id}")
        _require_string(source.get("locator"), "locator", f"surface union entry {entry_id}")
        kinds.add(source_kind)
    missing_kinds = sorted(_REQUIRED_UNION_KINDS - kinds)
    if missing_kinds:
        raise CompleteInventoryValidationError(f"surface union is missing required kinds: {', '.join(missing_kinds)}")
    return records, provenance


def _validate_source_artifacts(
    inventory: Mapping[str, Any],
    *,
    legacy_path: Path,
    legacy_payload: bytes,
    union_path: Path,
    union_payload: bytes,
    legacy_records: Mapping[str, dict[str, Any]],
    union_records: Mapping[str, dict[str, Any]],
    provenance: Mapping[str, Any],
) -> None:
    artifacts = inventory.get("source_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise CompleteInventoryValidationError("complete inventory requires exactly two source_artifacts")
    expected = {
        "legacy_surface_discovery": (legacy_path.name, legacy_payload, legacy_records),
        "surface_union": (union_path.name, union_payload, union_records),
    }
    actual_ids: set[str] = set()
    for index, artifact in enumerate(artifacts):
        subject = f"source_artifacts entry {index}"
        if not isinstance(artifact, dict) or set(artifact) != {
            "source_id",
            "path",
            "sha256",
            "entry_count",
            "source_provenance",
        }:
            raise CompleteInventoryValidationError(f"{subject} must contain the complete source binding")
        source_id = _require_string(artifact.get("source_id"), "source_id", subject)
        if source_id in actual_ids or source_id not in expected:
            raise CompleteInventoryValidationError(f"{subject} source_id is duplicate or unsupported: {source_id}")
        actual_ids.add(source_id)
        expected_name, payload, records = expected[source_id]
        if _require_repository_filename(artifact.get("path"), "path", subject) != expected_name:
            raise CompleteInventoryValidationError(f"{subject} path does not bind the supplied source file")
        if _require_sha256(artifact.get("sha256"), "sha256", subject) != hashlib.sha256(payload).hexdigest():
            raise CompleteInventoryValidationError(f"{subject} sha256 does not match its source file")
        if _require_exact_int(artifact.get("entry_count"), "entry_count", subject) != len(records):
            raise CompleteInventoryValidationError(f"{subject} entry_count does not match its source file")
        if _validate_provenance(artifact.get("source_provenance"), subject) != dict(provenance):
            raise CompleteInventoryValidationError(f"{subject} source provenance does not match the complete union")
    if actual_ids != set(expected):
        raise CompleteInventoryValidationError("complete inventory source_artifacts do not bind every required source")


def _validate_inventory_entry(
    entry: object,
    index: int,
    *,
    raw_records: Mapping[tuple[str, str], dict[str, Any]],
    owners: set[str],
) -> tuple[str, tuple[str, str]]:
    subject = f"complete inventory entry {index}"
    if not isinstance(entry, dict) or set(entry) != _EXPECTED_INVENTORY_FIELDS:
        raise CompleteInventoryValidationError(f"{subject} must contain exactly the complete decision fields")
    source_id = _require_string(entry.get("source_id"), "source_id", subject)
    raw_entry_id = _require_string(entry.get("raw_entry_id"), "raw_entry_id", subject)
    raw_key = (source_id, raw_entry_id)
    raw_entry = raw_records.get(raw_key)
    if raw_entry is None:
        raise CompleteInventoryValidationError(f"{subject} references an unknown raw source entry")
    expected_inventory_id = f"{source_id}:{raw_entry_id}"
    if _require_string(entry.get("inventory_entry_id"), "inventory_entry_id", subject) != expected_inventory_id:
        raise CompleteInventoryValidationError(
            f"{subject} inventory_entry_id must bind source_id and raw_entry_id exactly"
        )
    if _require_sha256(entry.get("raw_entry_sha256"), "raw_entry_sha256", subject) != canonical_sha256(raw_entry):
        raise CompleteInventoryValidationError(f"{subject} raw_entry_sha256 does not match its raw source entry")
    owner = _require_string(entry.get("owner"), "owner", subject)
    if owner not in owners:
        raise CompleteInventoryValidationError(f"{subject} owner must be declared in inventory owners")
    disposition = _require_string(entry.get("disposition"), "disposition", subject)
    if disposition not in _ALLOWED_DISPOSITIONS:
        raise CompleteInventoryValidationError(f"{subject} disposition must be a reviewed complete-inventory decision")
    capability_ids = _require_sorted_strings(entry.get("capability_ids"), "capability_ids", subject)
    if not all(_CAPABILITY_ID.fullmatch(capability_id) for capability_id in capability_ids):
        raise CompleteInventoryValidationError(f"{subject} capability_ids must be controlled capability identifiers")
    target_operation_id = _require_string(entry.get("target_operation_id"), "target_operation_id", subject)
    if not _OPERATION_ID.fullmatch(target_operation_id):
        raise CompleteInventoryValidationError(
            f"{subject} target_operation_id must be a controlled operation identifier"
        )
    scenario_ids = _require_sorted_strings(entry.get("scenario_ids"), "scenario_ids", subject)
    if not all(_CAPABILITY_ID.fullmatch(scenario_id) for scenario_id in scenario_ids):
        raise CompleteInventoryValidationError(f"{subject} scenario_ids must be controlled scenario identifiers")
    source_nodeids = _require_sorted_strings(entry.get("source_nodeids"), "source_nodeids", subject, allow_empty=True)
    wheel_nodeids = _require_sorted_strings(entry.get("wheel_nodeids"), "wheel_nodeids", subject, allow_empty=True)
    if disposition == "required" and (not source_nodeids or not wheel_nodeids):
        raise CompleteInventoryValidationError(
            f"{subject} required disposition needs non-empty source_nodeids and wheel_nodeids"
        )
    evidence = entry.get("evidence")
    if not isinstance(evidence, dict) or not any(
        isinstance(value, str) and value.strip()
        for value in (evidence.get("golden_path"), evidence.get("oracle_reference"))
    ):
        raise CompleteInventoryValidationError(f"{subject} evidence requires golden_path or oracle_reference")
    if _require_string(entry.get("completion_gate"), "completion_gate", subject) not in _ALLOWED_GATES:
        raise CompleteInventoryValidationError(f"{subject} completion_gate must be a controlled D gate")
    _require_string(entry.get("rationale"), "rationale", subject)
    _require_string(entry.get("rule_id"), "rule_id", subject)
    return expected_inventory_id, raw_key


def validate_complete_inventory(
    legacy_discovery_path: Path,
    surface_union_path: Path,
    inventory_path: Path,
) -> dict[str, Any]:
    """Validate protected files and return a non-verdict coverage summary."""
    legacy, legacy_payload = _load_document(legacy_discovery_path, "legacy discovery")
    union, union_payload = _load_document(surface_union_path, "surface union")
    inventory, inventory_payload = _load_document(inventory_path, "complete inventory")
    legacy_records, legacy_provenance = _validate_legacy_discovery(legacy)
    union_records, union_provenance = _validate_surface_union(union)
    if legacy_provenance != union_provenance:
        raise CompleteInventoryValidationError("raw discovery provenance must match across the complete surface union")
    if _contains_assertion_field(inventory):
        raise CompleteInventoryValidationError("complete inventory must not assert D0, final, or legacy-zero status")
    if (
        inventory.get("schema_version") != SCHEMA_VERSION
        or inventory.get("artifact_type") != "complete_surface_inventory"
    ):
        raise CompleteInventoryValidationError("inventory must be a schema-v1 complete_surface_inventory artifact")
    if inventory.get("scope") != "complete_legacy_surface_union" or inventory.get("decision_status") != "complete":
        raise CompleteInventoryValidationError(
            "inventory must declare complete_legacy_surface_union with complete decision status"
        )
    if inventory.get("not_for_d0") is True:
        raise CompleteInventoryValidationError("complete inventory must not retain the interim not_for_d0 marker")
    non_assertions = _require_sorted_strings(inventory.get("does_not_assert"), "does_not_assert", "complete inventory")
    if not set(non_assertions) >= _REQUIRED_NON_ASSERTIONS:
        raise CompleteInventoryValidationError("complete inventory does_not_assert is incomplete")
    owner_list = _require_sorted_strings(inventory.get("owners"), "owners", "complete inventory")
    if not all(_OWNER_ID.fullmatch(owner) for owner in owner_list):
        raise CompleteInventoryValidationError("complete inventory owners must be controlled identifiers")
    _validate_source_artifacts(
        inventory,
        legacy_path=legacy_discovery_path,
        legacy_payload=legacy_payload,
        union_path=surface_union_path,
        union_payload=union_payload,
        legacy_records=legacy_records,
        union_records=union_records,
        provenance=legacy_provenance,
    )
    raw_records: dict[tuple[str, str], dict[str, Any]] = {
        **{("legacy_surface_discovery", entry_id): entry for entry_id, entry in legacy_records.items()},
        **{("surface_union", entry_id): entry for entry_id, entry in union_records.items()},
    }
    entries = inventory.get("entries")
    if not isinstance(entries, list) or not entries:
        raise CompleteInventoryValidationError("complete inventory entries must be a non-empty list")
    seen_inventory_ids: set[str] = set()
    seen_raw_keys: set[tuple[str, str]] = set()
    duplicate_entries: list[str] = []
    for index, entry in enumerate(entries):
        inventory_entry_id, raw_key = _validate_inventory_entry(
            entry, index, raw_records=raw_records, owners=set(owner_list)
        )
        if inventory_entry_id in seen_inventory_ids or raw_key in seen_raw_keys:
            duplicate_entries.append(inventory_entry_id)
            continue
        seen_inventory_ids.add(inventory_entry_id)
        seen_raw_keys.add(raw_key)
    if duplicate_entries:
        raise CompleteInventoryValidationError(
            f"complete inventory has duplicate entries: {', '.join(sorted(duplicate_entries))}"
        )
    unmapped_entries = sorted(
        f"{source_id}:{entry_id}" for source_id, entry_id in set(raw_records).difference(seen_raw_keys)
    )
    if unmapped_entries:
        raise CompleteInventoryValidationError(
            f"complete inventory has unmapped raw entries: {', '.join(unmapped_entries)}"
        )
    return {
        "artifact_type": "complete_surface_inventory_validation",
        "not_a_d0_verdict": True,
        "entry_count": len(raw_records),
        "unmapped_entries": [],
        "duplicate_entries": [],
        "source_provenance": legacy_provenance,
        "legacy_discovery_sha256": hashlib.sha256(legacy_payload).hexdigest(),
        "surface_union_sha256": hashlib.sha256(union_payload).hexdigest(),
        "inventory_sha256": hashlib.sha256(inventory_payload).hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-discovery", type=Path, required=True)
    parser.add_argument("--surface-union", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = validate_complete_inventory(arguments.legacy_discovery, arguments.surface_union, arguments.inventory)
    except CompleteInventoryValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
