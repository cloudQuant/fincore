#!/usr/bin/env python3
"""Validate a scoped 0042-R2 legacy-surface inventory.

The legacy-surface discovery records independent raw facts.  This checker
binds one reviewed decision to every raw row without merging overlapping
public paths or treating the scoped raw artifact as D0 evidence.  It covers
only the supplied ``legacy_surface_discovery`` input; maintained documents,
examples, benchmarks, built distributions, and test nodes remain outside this
preparatory slice.  Every row remains ``required`` here; alias and quirk
claims require a later complete checker with independent evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40,64}$")
_OWNER_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
_SOURCE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_CAPABILITY_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_OPERATION_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_ALLOWED_GATES = frozenset({"D0", "D-RUNTIME", "D-DOMAIN", "D-CUTOVER"})
_REQUIRED_NON_ASSERTIONS = frozenset({"D0", "D-TECH", "installed_wheel_behavior", "legacy_zero"})
_FORBIDDEN_RAW_DECISION_FIELDS = frozenset({"owner", "disposition", "target_operation_id", "oracle"})
_FORBIDDEN_INVENTORY_ASSERTION_FIELDS = frozenset(
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
_NORMALIZED_FORBIDDEN_RAW_DECISION_FIELDS = frozenset(
    re.sub(r"[^a-z0-9]+", "", field.casefold()) for field in _FORBIDDEN_RAW_DECISION_FIELDS
)
_NORMALIZED_FORBIDDEN_ASSERTION_FIELDS = frozenset(
    re.sub(r"[^a-z0-9]+", "", field.casefold()) for field in _FORBIDDEN_INVENTORY_ASSERTION_FIELDS
)
_RAW_DISCOVERY_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "discovery_status",
        "not_for_d0",
        "partial_reason",
        "required_source_kinds",
        "source",
        "source_artifacts",
        "entries",
        "discrepancies",
    }
)
_INVENTORY_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "scope",
        "decision_status",
        "not_for_d0",
        "does_not_assert",
        "source_contract",
        "source_discovery",
        "owners",
        "entries",
    }
)
_SOURCE_CONTRACT_FIELDS = frozenset({"raw_artifact_type", "required_raw_status", "required_raw_not_for_d0"})
_SOURCE_DISCOVERY_FIELDS = frozenset({"path", "sha256", "entry_count", "source_provenance"})
_PROVENANCE_FIELDS = frozenset({"commit", "tree", "clean"})
_ENTRY_FIELDS = frozenset(
    {
        "legacy_entry_id",
        "source",
        "owner",
        "disposition",
        "target_operation_id",
        "rationale",
        "rule_id",
        "completion_gate",
        "capability_ids",
    }
)
_ENTRY_SOURCE_FIELDS = frozenset(
    {"source_id", "source_kind", "artifact_path", "artifact_sha256", "locator", "raw_entry_sha256"}
)


class LegacySurfaceInventoryValidationError(ValueError):
    """Raised when a raw legacy fact has no safe, exact inventory decision."""


class _DuplicateJsonKeyError(ValueError):
    """Raised when a JSON object repeats a policy-relevant key."""


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def canonical_sha256(value: object) -> str:
    """Hash one JSON-compatible value with a deterministic encoding."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_regular_file(path: Path, label: str) -> bytes:
    """Read one regular file through one protected descriptor."""
    try:
        expected_metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise LegacySurfaceInventoryValidationError(f"cannot safely inspect {label}: {exc}") from exc
    if not stat.S_ISREG(expected_metadata.st_mode):
        raise LegacySurfaceInventoryValidationError(f"{label} must be a regular file, not a symbolic link: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LegacySurfaceInventoryValidationError(f"cannot safely open {label}: {exc}") from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise LegacySurfaceInventoryValidationError(f"{label} must be a regular file, not a symbolic link: {path}")
        if (status.st_dev, status.st_ino) != (expected_metadata.st_dev, expected_metadata.st_ino):
            raise LegacySurfaceInventoryValidationError(f"{label} changed identity while opening: {path}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            return stream.read()
    except OSError as exc:
        raise LegacySurfaceInventoryValidationError(f"cannot read {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def sha256_file(path: Path) -> str:
    """Return the SHA256 of one protected, regular input file."""
    return hashlib.sha256(_read_regular_file(path, str(path))).hexdigest()


def _load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_json_object_without_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as exc:
        raise LegacySurfaceInventoryValidationError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise LegacySurfaceInventoryValidationError(f"{label} must be a JSON object")
    return value


def _require_string(mapping: Mapping[str, Any], field: str, subject: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise LegacySurfaceInventoryValidationError(f"{subject} requires non-empty {field}")
    return value


def _require_sha256(mapping: Mapping[str, Any], field: str, subject: str) -> str:
    value = _require_string(mapping, field, subject)
    if not _SHA256.fullmatch(value):
        raise LegacySurfaceInventoryValidationError(f"{subject} {field} must be a lowercase SHA256")
    return value


def _require_exact_int(value: object, field: str, subject: str) -> int:
    if type(value) is not int:
        raise LegacySurfaceInventoryValidationError(f"{subject} {field} must be an integer")
    return value


def _require_repository_path(value: object, subject: str) -> str:
    if not isinstance(value, str) or not value:
        raise LegacySurfaceInventoryValidationError(f"{subject} requires a non-empty repository-relative path")
    pure_path = PurePosixPath(value)
    if (
        pure_path.is_absolute()
        or value != str(pure_path)
        or "\\" in value
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise LegacySurfaceInventoryValidationError(f"{subject} path must be repository-relative POSIX: {value!r}")
    return value


def _require_sorted_strings(value: object, field: str, subject: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "possibly empty " if allow_empty else "non-empty "
        raise LegacySurfaceInventoryValidationError(f"{subject} requires a {qualifier}{field} list")
    if not all(isinstance(item, str) and item for item in value):
        raise LegacySurfaceInventoryValidationError(f"{subject} {field} must contain non-empty strings")
    if value != sorted(set(value)):
        raise LegacySurfaceInventoryValidationError(f"{subject} {field} must be unique and sorted")
    return value


def _validate_provenance(value: object, subject: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LegacySurfaceInventoryValidationError(f"{subject} source_provenance must be an object")
    _reject_unexpected_fields(value, _PROVENANCE_FIELDS, f"{subject} source_provenance")
    commit = _require_string(value, "commit", subject)
    tree = _require_string(value, "tree", subject)
    if not _GIT_OBJECT_ID.fullmatch(commit) or not _GIT_OBJECT_ID.fullmatch(tree):
        raise LegacySurfaceInventoryValidationError(f"{subject} source_provenance must contain Git object identifiers")
    if value.get("clean") is not True:
        raise LegacySurfaceInventoryValidationError(f"{subject} source_provenance clean must be true")
    return {"commit": commit, "tree": tree, "clean": True}


def _require_source_id(value: object, field: str, subject: str) -> str:
    source_id = _require_string({field: value}, field, subject)
    if not _SOURCE_ID.fullmatch(source_id):
        raise LegacySurfaceInventoryValidationError(f"{subject} {field} must be a controlled source identifier")
    return source_id


def _reject_unexpected_fields(mapping: Mapping[str, Any], allowed: frozenset[str], subject: str) -> None:
    unexpected = sorted(set(mapping) - allowed)
    if unexpected:
        raise LegacySurfaceInventoryValidationError(f"{subject} contains unsupported fields: {', '.join(unexpected)}")


def _normalized_field_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold()) if isinstance(value, str) else ""


def _contains_raw_decision_field(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_normalized_field_key(key) in _NORMALIZED_FORBIDDEN_RAW_DECISION_FIELDS for key in value) or any(
            _contains_raw_decision_field(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_raw_decision_field(item) for item in value)
    return False


def _contains_inventory_assertion_field(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_normalized_field_key(key) in _NORMALIZED_FORBIDDEN_ASSERTION_FIELDS for key in value) or any(
            _contains_inventory_assertion_field(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_inventory_assertion_field(item) for item in value)
    return False


def _parse_raw_discovery(raw: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if _contains_inventory_assertion_field(raw):
        raise LegacySurfaceInventoryValidationError(
            "raw discovery must not contain D0, legacy-zero, or final-verdict assertion fields"
        )
    _reject_unexpected_fields(raw, _RAW_DISCOVERY_FIELDS, "raw discovery")
    if _require_exact_int(raw.get("schema_version"), "schema_version", "raw discovery") != SCHEMA_VERSION:
        raise LegacySurfaceInventoryValidationError(f"raw discovery schema_version must be {SCHEMA_VERSION}")
    if raw.get("artifact_type") != "legacy_surface_discovery":
        raise LegacySurfaceInventoryValidationError("raw discovery artifact_type must be legacy_surface_discovery")
    if raw.get("discovery_status") != "partial" or raw.get("not_for_d0") is not True:
        raise LegacySurfaceInventoryValidationError("raw discovery must remain a partial not_for_d0 artifact")
    _require_string(raw, "partial_reason", "raw discovery")
    provenance = _validate_provenance(raw.get("source"), "raw discovery")
    required_source_kinds = _require_sorted_strings(
        raw.get("required_source_kinds"), "required_source_kinds", "raw discovery"
    )
    if not all(_SOURCE_ID.fullmatch(source_kind) for source_kind in required_source_kinds):
        raise LegacySurfaceInventoryValidationError(
            "raw discovery required_source_kinds must be controlled identifiers"
        )

    source_artifacts = raw.get("source_artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        raise LegacySurfaceInventoryValidationError("raw discovery source_artifacts must be a non-empty list")
    artifacts_by_id: dict[str, dict[str, str]] = {}
    for index, raw_artifact in enumerate(source_artifacts):
        subject = f"raw discovery source_artifact {index}"
        if not isinstance(raw_artifact, dict):
            raise LegacySurfaceInventoryValidationError(f"{subject} must be an object")
        source_id = _require_source_id(raw_artifact.get("source_id"), "source_id", subject)
        if source_id in artifacts_by_id:
            raise LegacySurfaceInventoryValidationError(f"raw discovery contains duplicate source_id: {source_id}")
        source_kind = _require_source_id(raw_artifact.get("source_kind"), "source_kind", subject)
        if source_kind != source_id:
            raise LegacySurfaceInventoryValidationError(f"{subject} source_kind must match source_id")
        artifacts_by_id[source_id] = {
            "source_id": source_id,
            "source_kind": source_kind,
            "artifact_path": _require_repository_path(raw_artifact.get("path"), subject),
            "artifact_sha256": _require_sha256(raw_artifact, "sha256", subject),
        }
    if list(artifacts_by_id) != sorted(artifacts_by_id):
        raise LegacySurfaceInventoryValidationError("raw discovery source_artifacts must be sorted by source_id")
    if set(required_source_kinds) != set(artifacts_by_id):
        raise LegacySurfaceInventoryValidationError(
            "raw discovery required_source_kinds must exactly match source_artifacts"
        )

    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        raise LegacySurfaceInventoryValidationError("raw discovery entries must be a non-empty list")
    entries_by_id: dict[str, dict[str, Any]] = {}
    source_counts: Counter[str] = Counter()
    for index, raw_entry in enumerate(entries):
        subject = f"raw discovery entry {index}"
        if not isinstance(raw_entry, dict):
            raise LegacySurfaceInventoryValidationError(f"{subject} must be an object")
        if _contains_raw_decision_field(raw_entry):
            raise LegacySurfaceInventoryValidationError(f"{subject} must not contain decision fields")
        entry_id = _require_string(raw_entry, "entry_id", subject)
        if entry_id in entries_by_id:
            raise LegacySurfaceInventoryValidationError(f"raw discovery contains duplicate entry_id: {entry_id}")
        source_id = _require_source_id(raw_entry.get("source_id"), "source_id", subject)
        source_kind = _require_source_id(raw_entry.get("source_kind"), "source_kind", subject)
        artifact = artifacts_by_id.get(source_id)
        if artifact is None or source_kind != artifact["source_kind"]:
            raise LegacySurfaceInventoryValidationError(f"{subject} source identity does not match source_artifacts")
        locator = raw_entry.get("source_locator")
        if not isinstance(locator, dict):
            raise LegacySurfaceInventoryValidationError(f"{subject} source_locator must be an object")
        artifact_path = _require_repository_path(locator.get("artifact_path"), subject)
        artifact_sha256 = _require_sha256(locator, "artifact_sha256", subject)
        locator_value = _require_string(locator, "locator", subject)
        if artifact_path != artifact["artifact_path"] or artifact_sha256 != artifact["artifact_sha256"]:
            raise LegacySurfaceInventoryValidationError(f"{subject} source_locator must match its source artifact")
        entries_by_id[entry_id] = {
            "entry_id": entry_id,
            "source_id": source_id,
            "source_kind": source_kind,
            "artifact_path": artifact_path,
            "artifact_sha256": artifact_sha256,
            "locator": locator_value,
            "raw_entry_sha256": canonical_sha256(raw_entry),
        }
        source_counts[source_id] += 1
    missing_sources = sorted(source_id for source_id in artifacts_by_id if source_counts[source_id] == 0)
    if missing_sources:
        raise LegacySurfaceInventoryValidationError(
            f"raw discovery has source_artifacts without entries: {', '.join(missing_sources)}"
        )
    return entries_by_id, provenance


def _validate_source_contract(inventory: Mapping[str, Any], raw: Mapping[str, Any]) -> None:
    contract = inventory.get("source_contract")
    if not isinstance(contract, dict):
        raise LegacySurfaceInventoryValidationError("inventory source_contract must be an object")
    _reject_unexpected_fields(contract, _SOURCE_CONTRACT_FIELDS, "inventory source_contract")
    if contract.get("raw_artifact_type") != raw.get("artifact_type"):
        raise LegacySurfaceInventoryValidationError(
            "inventory source_contract raw_artifact_type does not match raw discovery"
        )
    if contract.get("required_raw_status") != raw.get("discovery_status"):
        raise LegacySurfaceInventoryValidationError(
            "inventory source_contract required_raw_status does not match raw discovery"
        )
    if contract.get("required_raw_not_for_d0") is not raw.get("not_for_d0"):
        raise LegacySurfaceInventoryValidationError(
            "inventory source_contract required_raw_not_for_d0 does not match raw discovery"
        )


def _validate_source_discovery(
    inventory: Mapping[str, Any],
    raw_filename: str,
    raw_sha256: str,
    raw_entry_count: int,
    provenance: Mapping[str, Any],
) -> None:
    source_discovery = inventory.get("source_discovery")
    if not isinstance(source_discovery, dict):
        raise LegacySurfaceInventoryValidationError("inventory source_discovery must be an object")
    _reject_unexpected_fields(source_discovery, _SOURCE_DISCOVERY_FIELDS, "inventory source_discovery")
    if _require_repository_path(source_discovery.get("path"), "inventory source_discovery") != raw_filename:
        raise LegacySurfaceInventoryValidationError(
            "inventory source_discovery path must name the supplied raw discovery"
        )
    if _require_sha256(source_discovery, "sha256", "inventory source_discovery") != raw_sha256:
        raise LegacySurfaceInventoryValidationError(
            "inventory source_discovery sha256 does not match supplied raw discovery"
        )
    if (
        _require_exact_int(source_discovery.get("entry_count"), "entry_count", "inventory source_discovery")
        != raw_entry_count
    ):
        raise LegacySurfaceInventoryValidationError(
            "inventory source_discovery entry_count does not match raw discovery"
        )
    if _validate_provenance(source_discovery.get("source_provenance"), "inventory") != dict(provenance):
        raise LegacySurfaceInventoryValidationError(
            "inventory source_discovery source_provenance does not match raw discovery"
        )


def _validate_entry(
    entry: object,
    index: int,
    raw_entry: Mapping[str, Any],
    owners: set[str],
) -> str:
    subject = f"inventory entry {index}"
    if not isinstance(entry, dict):
        raise LegacySurfaceInventoryValidationError(f"{subject} must be an object")
    entry_id = _require_string(entry, "legacy_entry_id", subject)
    if entry_id != raw_entry["entry_id"]:
        raise LegacySurfaceInventoryValidationError(f"{subject} legacy_entry_id does not match its raw entry")
    source = entry.get("source")
    if not isinstance(source, dict):
        raise LegacySurfaceInventoryValidationError(f"{subject} source must be an object")
    _reject_unexpected_fields(source, _ENTRY_SOURCE_FIELDS, f"{subject} source")
    for field in ("source_id", "source_kind", "artifact_path", "artifact_sha256", "locator", "raw_entry_sha256"):
        expected = raw_entry[field]
        if field in {"source_id", "source_kind"}:
            actual = _require_source_id(source.get(field), field, subject)
        elif field == "artifact_path":
            actual = _require_repository_path(source.get(field), subject)
        elif field in {"artifact_sha256", "raw_entry_sha256"}:
            actual = _require_sha256(source, field, subject)
        else:
            actual = _require_string(source, field, subject)
        if actual != expected:
            raise LegacySurfaceInventoryValidationError(f"{subject} source {field} does not match raw entry")
    owner = _require_string(entry, "owner", subject)
    if owner not in owners:
        raise LegacySurfaceInventoryValidationError(f"{subject} owner must appear in inventory owners")
    _require_string(entry, "rationale", subject)
    _require_string(entry, "rule_id", subject)
    completion_gate = _require_string(entry, "completion_gate", subject)
    if completion_gate not in _ALLOWED_GATES:
        choices = ", ".join(sorted(_ALLOWED_GATES))
        raise LegacySurfaceInventoryValidationError(f"{subject} completion_gate must be one of: {choices}")
    disposition = _require_string(entry, "disposition", subject)
    if disposition != "required":
        raise LegacySurfaceInventoryValidationError(
            f"{subject} preparatory raw inventory only supports required disposition; "
            "alias_only and legacy_quirk require complete independent evidence"
        )
    _reject_unexpected_fields(entry, _ENTRY_FIELDS, subject)
    capability_ids = _require_sorted_strings(entry.get("capability_ids"), "capability_ids", subject)
    if not all(_CAPABILITY_ID.fullmatch(capability_id) for capability_id in capability_ids):
        raise LegacySurfaceInventoryValidationError(f"{subject} capability_ids must be controlled identifiers")
    target_operation_id = entry.get("target_operation_id")
    if not isinstance(target_operation_id, str) or not _OPERATION_ID.fullmatch(target_operation_id):
        raise LegacySurfaceInventoryValidationError(
            f"{subject} required disposition requires a unified target_operation_id"
        )
    return entry_id


def validate_legacy_surface_inventory_payloads(
    raw_discovery_payload: bytes,
    inventory_payload: bytes,
    *,
    raw_filename: str,
) -> dict[str, Any]:
    """Validate immutable raw/inventory bytes and return a scoped non-D0 summary."""
    raw = _load_json_bytes(raw_discovery_payload, "raw discovery")
    inventory = _load_json_bytes(inventory_payload, "inventory")
    raw_entries_by_id, provenance = _parse_raw_discovery(raw)
    raw_sha256 = hashlib.sha256(raw_discovery_payload).hexdigest()
    inventory_sha256 = hashlib.sha256(inventory_payload).hexdigest()

    if _contains_inventory_assertion_field(inventory):
        raise LegacySurfaceInventoryValidationError(
            "scoped inventory must not contain D0, legacy-zero, or final-verdict assertion fields"
        )
    _reject_unexpected_fields(inventory, _INVENTORY_FIELDS, "inventory")

    if _require_exact_int(inventory.get("schema_version"), "schema_version", "inventory") != SCHEMA_VERSION:
        raise LegacySurfaceInventoryValidationError(f"inventory schema_version must be {SCHEMA_VERSION}")
    if inventory.get("artifact_type") != "legacy_surface_inventory":
        raise LegacySurfaceInventoryValidationError("inventory artifact_type must be legacy_surface_inventory")
    if inventory.get("scope") != "raw_legacy_surface_only":
        raise LegacySurfaceInventoryValidationError("inventory scope must be raw_legacy_surface_only")
    if inventory.get("decision_status") != "scoped" or inventory.get("not_for_d0") is not True:
        raise LegacySurfaceInventoryValidationError("inventory must be explicitly scoped and not_for_d0")
    non_assertions = _require_sorted_strings(inventory.get("does_not_assert"), "does_not_assert", "inventory")
    if not set(non_assertions) >= _REQUIRED_NON_ASSERTIONS:
        raise LegacySurfaceInventoryValidationError("inventory does_not_assert must retain all non-D0 boundaries")
    owner_list = _require_sorted_strings(inventory.get("owners"), "owners", "inventory")
    if not all(_OWNER_ID.fullmatch(owner) for owner in owner_list):
        raise LegacySurfaceInventoryValidationError("inventory owners must be controlled owner identifiers")
    _validate_source_contract(inventory, raw)
    _validate_source_discovery(inventory, raw_filename, raw_sha256, len(raw_entries_by_id), provenance)

    inventory_entries = inventory.get("entries")
    if not isinstance(inventory_entries, list) or not inventory_entries:
        raise LegacySurfaceInventoryValidationError("inventory entries must be a non-empty list")
    seen_entry_ids: set[str] = set()
    duplicate_entry_ids: list[str] = []
    disposition_counts: Counter[str] = Counter()
    source_kind_counts: Counter[str] = Counter()
    prior_entry_id: str | None = None
    for index, entry in enumerate(inventory_entries):
        if not isinstance(entry, dict):
            raise LegacySurfaceInventoryValidationError(f"inventory entry {index} must be an object")
        entry_id = _require_string(entry, "legacy_entry_id", f"inventory entry {index}")
        if entry_id in seen_entry_ids:
            duplicate_entry_ids.append(entry_id)
            continue
        if prior_entry_id is not None and entry_id <= prior_entry_id:
            raise LegacySurfaceInventoryValidationError("inventory entries must be sorted by legacy_entry_id")
        prior_entry_id = entry_id
        raw_entry = raw_entries_by_id.get(entry_id)
        if raw_entry is None:
            raise LegacySurfaceInventoryValidationError(
                f"inventory entry {index} maps an unknown raw entry_id: {entry_id}"
            )
        mapped_entry_id = _validate_entry(entry, index, raw_entry, set(owner_list))
        seen_entry_ids.add(mapped_entry_id)
        disposition_counts["required"] += 1
        source_kind_counts[str(raw_entry["source_kind"])] += 1
    if duplicate_entry_ids:
        raise LegacySurfaceInventoryValidationError(
            f"inventory has duplicate legacy_entry_id values: {', '.join(sorted(duplicate_entry_ids))}"
        )
    unmapped_entry_ids = sorted(set(raw_entries_by_id) - seen_entry_ids)
    if unmapped_entry_ids:
        raise LegacySurfaceInventoryValidationError(
            f"inventory has unmapped raw entry_id values: {', '.join(unmapped_entry_ids)}"
        )
    return {
        "artifact_type": "legacy_surface_inventory_validation",
        "scope": "raw_legacy_surface_only",
        "not_for_d0": True,
        "record_count": len(raw_entries_by_id),
        "source_provenance": provenance,
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "unmapped_entry_ids": [],
        "duplicate_entry_ids": [],
        "raw_discovery_sha256": raw_sha256,
        "inventory_sha256": inventory_sha256,
    }


def validate_legacy_surface_inventory(raw_discovery_path: Path, inventory_path: Path) -> dict[str, Any]:
    """Validate protected input files by delegating their frozen bytes to the core."""
    raw_discovery_payload = _read_regular_file(raw_discovery_path, "raw discovery")
    inventory_payload = _read_regular_file(inventory_path, "inventory")
    return validate_legacy_surface_inventory_payloads(
        raw_discovery_payload,
        inventory_payload,
        raw_filename=raw_discovery_path.name,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-discovery", type=Path, required=True, help="partial raw legacy-surface discovery JSON")
    parser.add_argument("--inventory", type=Path, required=True, help="scoped legacy-surface inventory JSON")
    arguments = parser.parse_args(argv)
    try:
        summary = validate_legacy_surface_inventory(arguments.raw_discovery, arguments.inventory)
    except LegacySurfaceInventoryValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
