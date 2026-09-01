#!/usr/bin/env python3
"""Validate a scoped 0042-R2 repository-surface disposition ledger.

Repository-surface discovery intentionally records raw path facts only.  This
checker binds a separately reviewed disposition document to those immutable
facts and rejects missing, duplicated, or drifted decisions.  A successful
result is still *not* D0 evidence: it proves only that this one scoped ledger
is complete for the supplied discovery artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40,64}$")
_OWNER_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
_ALLOWED_LIFECYCLES = frozenset({"maintained", "transition", "historical_provenance"})
_ALLOWED_DISPOSITIONS = frozenset({"retain", "retarget", "replace", "remove", "allowlist"})
_ALLOWED_GATES = frozenset({"D0", "D-RUNTIME", "D-DOMAIN", "D-CUTOVER"})
_ALLOWED_LEGACY_POLICIES = frozenset(
    {"none", "retarget_before_cutover", "text_only_allowlist", "artifact_provenance_allowlist"}
)
_REQUIRED_NON_ASSERTIONS = frozenset({"D0", "D-TECH", "installed_wheel_behavior", "legacy_zero"})
_HISTORICAL_TAG = "historical_provenance_candidate"
_TEXT_ONLY_HISTORICAL_SUFFIXES = frozenset(
    {".csv", ".diff", ".json", ".markdown", ".md", ".rst", ".txt", ".yaml", ".yml"}
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


class DispositionValidationError(ValueError):
    """Raised when repository facts cannot support a complete disposition."""


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
    """Hash one JSON-compatible value with an unambiguous canonical encoding."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_regular_file(path: Path, label: str) -> bytes:
    """Read one regular file through one protected descriptor."""
    try:
        expected_metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DispositionValidationError(f"cannot safely inspect {label}: {exc}") from exc
    if not stat.S_ISREG(expected_metadata.st_mode):
        raise DispositionValidationError(f"{label} must be a regular file, not a symbolic link: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DispositionValidationError(f"cannot safely open {label}: {exc}") from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise DispositionValidationError(f"{label} must be a regular file, not a symbolic link: {path}")
        if (status.st_dev, status.st_ino) != (expected_metadata.st_dev, expected_metadata.st_ino):
            raise DispositionValidationError(f"{label} changed identity while opening: {path}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            return stream.read()
    except OSError as exc:
        raise DispositionValidationError(f"cannot read {label}: {exc}") from exc
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
        raise DispositionValidationError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise DispositionValidationError(f"{label} must be a JSON object")
    return value


def _require_string(mapping: Mapping[str, Any], field: str, subject: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DispositionValidationError(f"{subject} requires non-empty {field}")
    return value


def _require_sha256(mapping: Mapping[str, Any], field: str, subject: str) -> str:
    value = _require_string(mapping, field, subject)
    if not _SHA256.fullmatch(value):
        raise DispositionValidationError(f"{subject} {field} must be a lowercase SHA256")
    return value


def _require_exact_int(value: object, field: str, subject: str) -> int:
    if type(value) is not int:
        raise DispositionValidationError(f"{subject} {field} must be an integer")
    return value


def _require_repository_path(value: object, subject: str) -> str:
    if not isinstance(value, str) or not value:
        raise DispositionValidationError(f"{subject} requires a non-empty repository-relative path")
    pure_path = PurePosixPath(value)
    if (
        pure_path.is_absolute()
        or value != str(pure_path)
        or "\\" in value
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise DispositionValidationError(f"{subject} path must be repository-relative POSIX: {value!r}")
    return value


def _require_sorted_strings(value: object, field: str, subject: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise DispositionValidationError(
            f"{subject} requires a {'possibly empty ' if allow_empty else 'non-empty '}{field} list"
        )
    if not all(isinstance(item, str) and item for item in value):
        raise DispositionValidationError(f"{subject} {field} must contain non-empty strings")
    if value != sorted(set(value)):
        raise DispositionValidationError(f"{subject} {field} must be unique and sorted")
    return value


def _normalized_field_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold()) if isinstance(value, str) else ""


def _contains_assertion_field(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_normalized_field_key(key) in _NORMALIZED_FORBIDDEN_ASSERTION_FIELDS for key in value) or any(
            _contains_assertion_field(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_assertion_field(item) for item in value)
    return False


def _validate_provenance(value: object, subject: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DispositionValidationError(f"{subject} source_provenance must be an object")
    commit = _require_string(value, "commit", subject)
    tree = _require_string(value, "tree", subject)
    if not _GIT_OBJECT_ID.fullmatch(commit) or not _GIT_OBJECT_ID.fullmatch(tree):
        raise DispositionValidationError(f"{subject} source_provenance must contain Git object identifiers")
    if value.get("clean") is not True:
        raise DispositionValidationError(f"{subject} source_provenance clean must be true")
    return {"commit": commit, "tree": tree, "clean": True}


def _parse_facts(facts: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if _contains_assertion_field(facts):
        raise DispositionValidationError("facts must not contain D0, legacy-zero, or final-verdict assertion fields")
    if _require_exact_int(facts.get("schema_version"), "schema_version", "facts") != SCHEMA_VERSION:
        raise DispositionValidationError(f"facts schema_version must be {SCHEMA_VERSION}")
    if facts.get("artifact_type") != "repository_surface_facts_discovery":
        raise DispositionValidationError("facts artifact_type must be repository_surface_facts_discovery")
    if facts.get("discovery_status") != "partial" or facts.get("not_for_d0") is not True:
        raise DispositionValidationError("facts must remain a partial not_for_d0 discovery artifact")
    if not isinstance(facts.get("boundaries"), dict):
        raise DispositionValidationError("facts boundaries must be an object")
    provenance = _validate_provenance(facts.get("source_provenance"), "facts")
    records = facts.get("records")
    if not isinstance(records, list) or not records:
        raise DispositionValidationError("facts records must be a non-empty list")
    if _require_exact_int(facts.get("record_count"), "record_count", "facts") != len(records):
        raise DispositionValidationError("facts record_count must equal records length")

    records_by_path: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(records):
        subject = f"facts record {index}"
        if not isinstance(raw_record, dict):
            raise DispositionValidationError(f"{subject} must be an object")
        path = _require_repository_path(raw_record.get("path"), subject)
        if path in records_by_path:
            raise DispositionValidationError(f"facts contains duplicate path: {path}")
        git_mode = _require_string(raw_record, "git_mode", subject)
        if git_mode not in {"100644", "100755"}:
            raise DispositionValidationError(f"{subject} git_mode must name a regular Git blob")
        token_facts = raw_record.get("token_facts")
        if not isinstance(token_facts, dict):
            raise DispositionValidationError(f"{subject} token_facts must be an object")
        content_kind = _require_string(token_facts, "content_kind", subject)
        if content_kind not in {"text", "binary"}:
            raise DispositionValidationError(f"{subject} token_facts content_kind must be text or binary")
        records_by_path[path] = {
            "path": path,
            "git_mode": git_mode,
            "blob_sha256": _require_sha256(raw_record, "blob_sha256", subject),
            "kind": _require_string(raw_record, "kind", subject),
            "category_tags": _require_sorted_strings(raw_record.get("category_tags"), "category_tags", subject),
            "content_kind": content_kind,
        }
    return records_by_path, provenance


def _validate_source_contract(disposition: Mapping[str, Any], facts: Mapping[str, Any]) -> None:
    contract = disposition.get("source_contract")
    if not isinstance(contract, dict):
        raise DispositionValidationError("disposition source_contract must be an object")
    if contract.get("raw_artifact_type") != facts.get("artifact_type"):
        raise DispositionValidationError("disposition source_contract raw_artifact_type does not match facts")
    if contract.get("required_raw_status") != facts.get("discovery_status"):
        raise DispositionValidationError("disposition source_contract required_raw_status does not match facts")
    if contract.get("required_raw_not_for_d0") is not facts.get("not_for_d0"):
        raise DispositionValidationError("disposition source_contract required_raw_not_for_d0 does not match facts")
    expected_boundaries_sha256 = canonical_sha256(facts["boundaries"])
    if _require_sha256(contract, "boundaries_sha256", "disposition source_contract") != expected_boundaries_sha256:
        raise DispositionValidationError("disposition source_contract boundaries_sha256 does not match facts")


def _validate_source_facts(
    disposition: Mapping[str, Any],
    facts_filename: str,
    facts_sha256: str,
    facts: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    source_facts = disposition.get("source_facts")
    if not isinstance(source_facts, dict):
        raise DispositionValidationError("disposition source_facts must be an object")
    if _require_repository_path(source_facts.get("path"), "disposition source_facts") != facts_filename:
        raise DispositionValidationError("disposition source_facts path must name the supplied facts file")
    if _require_sha256(source_facts, "sha256", "disposition source_facts") != facts_sha256:
        raise DispositionValidationError("disposition source_facts sha256 does not match supplied facts")
    if _require_exact_int(source_facts.get("record_count"), "record_count", "disposition source_facts") != facts.get(
        "record_count"
    ):
        raise DispositionValidationError("disposition source_facts record_count does not match facts")
    if _validate_provenance(source_facts.get("source_provenance"), "disposition") != dict(provenance):
        raise DispositionValidationError("disposition source_facts source_provenance does not match facts")


def _validate_target(value: object, subject: str) -> None:
    if not isinstance(value, dict):
        raise DispositionValidationError(f"{subject} target must be an object")
    _require_repository_path(value.get("path"), subject)
    _require_sorted_strings(value.get("contract_ids"), "target contract_ids", subject, allow_empty=True)
    _require_sorted_strings(value.get("capability_ids"), "target capability_ids", subject, allow_empty=True)


def _historical_policy(fact: Mapping[str, Any]) -> tuple[str, bool]:
    suffix = PurePosixPath(str(fact["path"])).suffix.lower()
    text_only = fact["content_kind"] == "text" and suffix in _TEXT_ONLY_HISTORICAL_SUFFIXES
    return ("text_only_allowlist", True) if text_only else ("artifact_provenance_allowlist", False)


def _validate_entry(
    entry: object, index: int, fact: Mapping[str, Any], owners: set[str]
) -> tuple[str, dict[str, Any] | None]:
    subject = f"disposition entry {index}"
    if not isinstance(entry, dict):
        raise DispositionValidationError(f"{subject} must be an object")
    path = _require_repository_path(entry.get("path"), subject)
    if path != fact["path"]:
        raise DispositionValidationError(f"{subject} path does not match its source fact")
    source = entry.get("source")
    if not isinstance(source, dict):
        raise DispositionValidationError(f"{subject} source must be an object")
    if _require_string(source, "git_mode", subject) != fact["git_mode"]:
        raise DispositionValidationError(f"{subject} source git_mode does not match source fact")
    if _require_sha256(source, "blob_sha256", subject) != fact["blob_sha256"]:
        raise DispositionValidationError(f"{subject} source blob_sha256 does not match source fact")
    if _require_string(source, "kind", subject) != fact["kind"]:
        raise DispositionValidationError(f"{subject} source kind does not match source fact")
    if _require_sorted_strings(source.get("category_tags"), "source category_tags", subject) != fact["category_tags"]:
        raise DispositionValidationError(f"{subject} source category_tags do not match source fact category_tags")
    if _require_string(source, "content_kind", subject) != fact["content_kind"]:
        raise DispositionValidationError(f"{subject} source content_kind does not match source fact")
    owner = _require_string(entry, "owner", subject)
    if owner not in owners:
        raise DispositionValidationError(f"{subject} owner must appear in disposition owners")
    _require_string(entry, "rationale", subject)
    _require_string(entry, "rule_id", subject)
    lifecycle = _require_string(entry, "lifecycle", subject)
    disposition = _require_string(entry, "disposition", subject)
    completion_gate = _require_string(entry, "completion_gate", subject)
    legacy_reference_policy = _require_string(entry, "legacy_reference_policy", subject)
    if lifecycle not in _ALLOWED_LIFECYCLES:
        raise DispositionValidationError(
            f"{subject} lifecycle must be one of: {', '.join(sorted(_ALLOWED_LIFECYCLES))}"
        )
    if disposition not in _ALLOWED_DISPOSITIONS:
        raise DispositionValidationError(
            f"{subject} disposition must be one of: {', '.join(sorted(_ALLOWED_DISPOSITIONS))}"
        )
    if completion_gate not in _ALLOWED_GATES:
        raise DispositionValidationError(
            f"{subject} completion_gate must be one of: {', '.join(sorted(_ALLOWED_GATES))}"
        )
    if legacy_reference_policy not in _ALLOWED_LEGACY_POLICIES:
        raise DispositionValidationError(
            f"{subject} legacy_reference_policy must be one of: {', '.join(sorted(_ALLOWED_LEGACY_POLICIES))}"
        )
    target = entry.get("target")

    if _HISTORICAL_TAG in fact["category_tags"] and lifecycle != "historical_provenance":
        raise DispositionValidationError(f"{subject} historical source requires historical_provenance lifecycle")
    if lifecycle == "historical_provenance":
        if _HISTORICAL_TAG not in fact["category_tags"]:
            raise DispositionValidationError(
                f"{subject} historical_provenance lifecycle requires a historical source fact"
            )
        expected_policy, text_only = _historical_policy(fact)
        if (disposition, completion_gate, legacy_reference_policy, target) != (
            "allowlist",
            "D0",
            expected_policy,
            None,
        ):
            raise DispositionValidationError(
                f"{subject} historical_provenance requires allowlist, D0, {expected_policy}, and null target"
            )
        return path, {
            "path": path,
            "blob_sha256": fact["blob_sha256"],
            "reason": entry["rationale"],
            "text_only": text_only,
        }
    if disposition == "allowlist":
        raise DispositionValidationError(f"{subject} allowlist disposition requires historical_provenance lifecycle")
    if lifecycle == "maintained" and disposition not in {"retain", "retarget", "replace"}:
        raise DispositionValidationError(f"{subject} maintained lifecycle cannot use {disposition}")
    if lifecycle == "transition" and disposition not in {"retarget", "replace", "remove"}:
        raise DispositionValidationError(f"{subject} transition lifecycle cannot use {disposition}")
    if disposition == "remove":
        if target is not None:
            raise DispositionValidationError(f"{subject} remove target must be null")
    else:
        _validate_target(target, subject)
    return path, None


def _validate_historical_allowlist(value: object, expected: list[dict[str, Any]]) -> None:
    if not isinstance(value, list):
        raise DispositionValidationError("historical_provenance_allowlist must be a list")
    actual: list[dict[str, Any]] = []
    for index, record in enumerate(value):
        subject = f"historical_provenance_allowlist entry {index}"
        if not isinstance(record, dict):
            raise DispositionValidationError(f"{subject} must be an object")
        actual.append(
            {
                "path": _require_repository_path(record.get("path"), subject),
                "blob_sha256": _require_sha256(record, "blob_sha256", subject),
                "reason": _require_string(record, "reason", subject),
                "text_only": record.get("text_only"),
            }
        )
    if not all(type(item["text_only"]) is bool for item in actual):
        raise DispositionValidationError("historical_provenance_allowlist text_only must be boolean")
    if actual != sorted(actual, key=lambda item: item["path"]):
        raise DispositionValidationError("historical_provenance_allowlist must be sorted by path")
    if actual != expected:
        raise DispositionValidationError("historical_provenance_allowlist must exactly project historical dispositions")


def validate_disposition_payloads(
    facts_payload: bytes,
    disposition_payload: bytes,
    *,
    facts_filename: str,
) -> dict[str, Any]:
    """Validate frozen repository-surface bytes and return a non-D0 summary.

    Callers that already captured immutable Git blobs should use this entry
    point so validation and reported hashes apply to the exact same bytes.
    """
    facts = _load_json_bytes(facts_payload, "facts")
    disposition = _load_json_bytes(disposition_payload, "disposition")
    records_by_path, provenance = _parse_facts(facts)
    facts_sha256 = hashlib.sha256(facts_payload).hexdigest()
    disposition_sha256 = hashlib.sha256(disposition_payload).hexdigest()

    if _contains_assertion_field(disposition):
        raise DispositionValidationError(
            "disposition must not contain D0, legacy-zero, or final-verdict assertion fields"
        )
    if _require_exact_int(disposition.get("schema_version"), "schema_version", "disposition") != SCHEMA_VERSION:
        raise DispositionValidationError(f"disposition schema_version must be {SCHEMA_VERSION}")
    if disposition.get("artifact_type") != "repository_surface_disposition":
        raise DispositionValidationError("disposition artifact_type must be repository_surface_disposition")
    if disposition.get("scope") != "classified_repository_surface_only":
        raise DispositionValidationError("disposition scope must be classified_repository_surface_only")
    if disposition.get("decision_status") != "scoped" or disposition.get("not_for_d0") is not True:
        raise DispositionValidationError("disposition must be explicitly scoped and not_for_d0")
    non_assertions = _require_sorted_strings(disposition.get("does_not_assert"), "does_not_assert", "disposition")
    if not set(non_assertions) >= _REQUIRED_NON_ASSERTIONS:
        raise DispositionValidationError("disposition does_not_assert must retain all non-D0 boundaries")
    owner_list = _require_sorted_strings(disposition.get("owners"), "owners", "disposition")
    if not all(_OWNER_ID.fullmatch(owner) for owner in owner_list):
        raise DispositionValidationError("disposition owners must be controlled owner identifiers")
    _validate_source_contract(disposition, facts)
    _validate_source_facts(
        disposition,
        facts_filename,
        facts_sha256,
        facts,
        provenance,
    )

    entries = disposition.get("entries")
    if not isinstance(entries, list) or not entries:
        raise DispositionValidationError("disposition entries must be a non-empty list")
    seen_paths: set[str] = set()
    duplicate_paths: list[str] = []
    expected_allowlist: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise DispositionValidationError(f"disposition entry {index} must be an object")
        path = _require_repository_path(entry.get("path"), f"disposition entry {index}")
        if path in seen_paths:
            duplicate_paths.append(path)
            continue
        fact = records_by_path.get(path)
        if fact is None:
            raise DispositionValidationError(f"disposition entry {index} maps an unknown source path: {path}")
        path, historical_record = _validate_entry(entry, index, fact, set(owner_list))
        seen_paths.add(path)
        if historical_record is not None:
            expected_allowlist.append(historical_record)
    if duplicate_paths:
        raise DispositionValidationError(f"disposition has duplicate paths: {', '.join(sorted(duplicate_paths))}")
    unmapped_paths = sorted(set(records_by_path) - seen_paths)
    if unmapped_paths:
        raise DispositionValidationError(f"disposition has unmapped source paths: {', '.join(unmapped_paths)}")
    _validate_historical_allowlist(
        disposition.get("historical_provenance_allowlist"), sorted(expected_allowlist, key=lambda item: item["path"])
    )
    return {
        "artifact_type": "repository_surface_disposition_validation",
        "not_for_d0": True,
        "record_count": len(records_by_path),
        "source_provenance": provenance,
        "unmapped_paths": [],
        "duplicate_paths": [],
        "facts_sha256": facts_sha256,
        "disposition_sha256": disposition_sha256,
    }


def validate_disposition(facts_path: Path, disposition_path: Path) -> dict[str, Any]:
    """Validate protected files by delegating their frozen bytes to the core."""
    facts_payload = _read_regular_file(facts_path, "facts")
    disposition_payload = _read_regular_file(disposition_path, "disposition")
    return validate_disposition_payloads(
        facts_payload,
        disposition_payload,
        facts_filename=facts_path.name,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True, help="raw repository-surface facts JSON")
    parser.add_argument("--disposition", type=Path, required=True, help="scoped repository-surface disposition JSON")
    arguments = parser.parse_args(argv)
    try:
        summary = validate_disposition(arguments.facts, arguments.disposition)
    except DispositionValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
