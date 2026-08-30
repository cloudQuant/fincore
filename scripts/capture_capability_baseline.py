#!/usr/bin/env python3
"""Capture a fail-closed 0042-R2 capability-baseline input artifact.

This command captures *inputs and provenance only*.  A successful JSON file is
a ``capability_baseline_capture`` artifact, not a D0 or D-TECH verdict.  It
requires a clean Git worktree at an exact commit/tree and never contacts the
network; callers must explicitly acknowledge that policy with ``--deny-network``.

Ledger schema version 1 is a JSON object with ``schema_version: 1`` and a
non-empty ``entries`` list.  Every entry needs a unique non-empty
``capability_id``, ``owner`` and ``disposition``.  Every entry also records a
non-empty ``target_operation_id``, ``source_nodeids``, ``wheel_nodeids`` and
``scenarios`` list.  Each scenario needs a non-empty ``scenario_id`` and either
an existing fixture-relative ``golden_path`` or an ``oracle_reference``.  A
``disposition`` is exactly one of ``required``, ``alias_only`` or
``legacy_quirk``.  A ``required`` entry additionally needs an independent
authority mapping for each scenario.  Every authority has a non-empty
``kind`` and ``reference``.  The external implementation kinds
(``independent_reference_implementation``, ``pinned_upstream_oracle`` and
``upstream_reference``) additionally require a non-empty external
``source_project`` plus a ``version`` or ``artifact_digest``.  The normalized
source project cannot identify Fincore, a candidate or current source.  The
publication kinds (``published_standard`` and ``peer_reviewed_paper``) require
one of ``publication``, ``doi``, ``version`` or ``digest``.  The
``property_invariant`` kind requires ``invariant_id``.  Reference text that
describes a current/candidate Fincore self-output is rejected as defense in
depth; it cannot replace the structured provenance fields.

The independent authority schema is fail-closed: fields that are irrelevant to
a kind do not substitute for that kind's required provenance fields.

The inventory, module-disposition and test-disposition inputs are JSON objects
with an ``entries`` list; every record must have a non-empty ``disposition``.
Repository-surface facts and their reviewed disposition are separately
validated as scoped, explicitly non-D0 inputs.  All input and fixture bytes
are read from immutable blobs in the initially
recorded clean ``HEAD`` tree, never from the mutable working tree.  Each
``golden_path`` is a portable fixture-relative POSIX path: it cannot be
absolute, contain ``.``, ``..`` or empty segments, or contain a backslash.  The
selected input and fixture entries must be regular Git blobs; symbolic links are
rejected rather than being followed through the mutable worktree.  The
initial and final Git provenance must match before ``--output`` is atomically
replaced.  This binds every recorded SHA256 to the original commit/tree even if
the worktree is changed and restored while capture is running.

The capture command itself must run from the supplied clean tooling worktree.
Its capture/checker blobs and Git provenance are recorded separately from the
candidate source, so a later review can identify the exact static checker that
validated the scoped repository-surface inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import types
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Sequence

SCHEMA_VERSION = 1
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40,64}$")
_REQUIRED_DISPOSITION = "required"
_LEDGER_DISPOSITIONS = frozenset({"required", "alias_only", "legacy_quirk"})
_EXTERNAL_IMPLEMENTATION_KINDS = frozenset(
    {"independent_reference_implementation", "pinned_upstream_oracle", "upstream_reference"}
)
_PUBLICATION_AUTHORITY_KINDS = frozenset({"published_standard", "peer_reviewed_paper"})
_PROPERTY_INVARIANT_KIND = "property_invariant"
_INDEPENDENT_AUTHORITY_KINDS = (
    _EXTERNAL_IMPLEMENTATION_KINDS | _PUBLICATION_AUTHORITY_KINDS | {_PROPERTY_INVARIANT_KIND}
)
_EXTERNAL_ARTIFACT_FIELDS = ("version", "artifact_digest")
_PUBLICATION_TRACEABILITY_FIELDS = ("publication", "doi", "version", "digest")
_RESERVED_SOURCE_PROJECT_IDENTIFIERS = ("fincore", "candidate", "current")
_CAPTURE_TOOL_RELATIVE = "scripts/capture_capability_baseline.py"
_REPOSITORY_SURFACE_CHECKER_RELATIVE = "scripts/check_0042_r2_repository_surface_disposition.py"


class CaptureValidationError(ValueError):
    """Raised when a capture input cannot serve as exact-SHA provenance."""


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureValidationError(f"cannot read {label} JSON from initial HEAD blob: {exc}") from exc
    if not isinstance(value, dict):
        raise CaptureValidationError(f"{label} must be a JSON object")
    return value


def _require_entries(document: dict[str, Any], label: str) -> list[dict[str, Any]]:
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        raise CaptureValidationError(f"{label} entries must be non-empty")
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CaptureValidationError(f"{label} entry {index} must be a JSON object")
        normalized.append(entry)
    return normalized


def _validate_disposition_document(document: dict[str, Any], label: str) -> None:
    for index, entry in enumerate(_require_entries(document, label)):
        if not _non_empty_string(entry.get("disposition")):
            raise CaptureValidationError(f"{label} disposition entry {index} must be non-empty")


def _require_string(entry: dict[str, Any], key: str, subject: str) -> str:
    value = entry.get(key)
    if not _non_empty_string(value):
        raise CaptureValidationError(f"{subject} requires non-empty {key}")
    assert isinstance(value, str)
    return value.strip()


def _require_string_list(entry: dict[str, Any], key: str, subject: str) -> list[str]:
    value = entry.get(key)
    if not isinstance(value, list) or not value or not all(_non_empty_string(item) for item in value):
        raise CaptureValidationError(f"{subject} requires a non-empty {key} list")
    return [str(item).strip() for item in value]


def _require_ledger_disposition(entry: dict[str, Any], subject: str) -> str:
    disposition = entry.get("disposition")
    if not isinstance(disposition, str) or disposition not in _LEDGER_DISPOSITIONS:
        choices = ", ".join(sorted(_LEDGER_DISPOSITIONS))
        raise CaptureValidationError(f"{subject} ledger disposition must be one of: {choices}")
    return disposition


def _normalize_authority_identifier(value: str) -> str:
    """Normalize a provenance identifier before checking its reserved source role."""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _reference_describes_fincore_self_output(reference: str) -> bool:
    normalized = _normalize_authority_identifier(reference)
    return "fincore" in normalized and ("candidate" in normalized or "current" in normalized)


def _require_any_authority_field(
    authority: dict[str, Any],
    fields: tuple[str, ...],
    subject: str,
    description: str,
) -> None:
    if not any(_non_empty_string(authority.get(field)) for field in fields):
        raise CaptureValidationError(f"{subject} requires {description}")


def _validate_required_authority(authority: dict[str, Any], subject: str) -> None:
    kind = _require_string(authority, "kind", subject)
    reference = _require_string(authority, "reference", subject)
    if kind not in _INDEPENDENT_AUTHORITY_KINDS:
        choices = ", ".join(sorted(_INDEPENDENT_AUTHORITY_KINDS))
        raise CaptureValidationError(f"{subject} requires an independent authority kind; allowed kinds: {choices}")
    if _reference_describes_fincore_self_output(reference):
        raise CaptureValidationError(f"{subject} reference must not describe a Fincore candidate/current self-output")
    if kind in _EXTERNAL_IMPLEMENTATION_KINDS:
        source_project = _require_string(authority, "source_project", subject)
        normalized_project = _normalize_authority_identifier(source_project)
        if not normalized_project or any(
            identifier in normalized_project for identifier in _RESERVED_SOURCE_PROJECT_IDENTIFIERS
        ):
            raise CaptureValidationError(f"{subject} source_project must identify an external project")
        _require_any_authority_field(
            authority,
            _EXTERNAL_ARTIFACT_FIELDS,
            subject,
            "version or artifact_digest",
        )
        return
    if kind in _PUBLICATION_AUTHORITY_KINDS:
        _require_any_authority_field(
            authority,
            _PUBLICATION_TRACEABILITY_FIELDS,
            subject,
            "publication, doi, version, or digest",
        )
        return
    if kind == _PROPERTY_INVARIANT_KIND:
        _require_string(authority, "invariant_id", subject)
        return
    raise CaptureValidationError(f"{subject} has no structured provenance policy for authority kind {kind!r}")


def _validate_scenario(scenario: dict[str, Any], subject: str, *, required: bool) -> None:
    _require_string(scenario, "scenario_id", subject)
    golden_path = scenario.get("golden_path")
    oracle_reference = scenario.get("oracle_reference")
    if golden_path is not None and not _non_empty_string(golden_path):
        raise CaptureValidationError(f"{subject} golden_path must be non-empty when supplied")
    if oracle_reference is not None and not _non_empty_string(oracle_reference):
        raise CaptureValidationError(f"{subject} oracle_reference must be non-empty when supplied")
    if not _non_empty_string(golden_path) and not _non_empty_string(oracle_reference):
        raise CaptureValidationError(f"{subject} requires golden_path or oracle_reference")
    if not required:
        return

    authority = scenario.get("authority")
    if not isinstance(authority, dict):
        raise CaptureValidationError(f"{subject} requires an independent authority")
    _validate_required_authority(authority, subject)


def validate_ledger(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the versioned ledger schema and return its normalized entries."""
    if ledger.get("schema_version") != SCHEMA_VERSION:
        raise CaptureValidationError(f"ledger schema_version must be {SCHEMA_VERSION}")
    entries = _require_entries(ledger, "ledger")

    capability_ids: set[str] = set()
    for index, entry in enumerate(entries):
        subject = f"ledger entry {index}"
        capability_id = _require_string(entry, "capability_id", subject)
        if capability_id in capability_ids:
            raise CaptureValidationError(f"duplicate capability_id {capability_id!r}")
        capability_ids.add(capability_id)
        _require_string(entry, "owner", subject)
        disposition = _require_ledger_disposition(entry, subject)
        _require_string(entry, "target_operation_id", subject)
        _require_string_list(entry, "source_nodeids", subject)
        _require_string_list(entry, "wheel_nodeids", subject)
        scenarios = entry.get("scenarios")
        if not isinstance(scenarios, list) or not scenarios:
            raise CaptureValidationError(f"{subject} requires a non-empty scenarios list")
        for scenario_index, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict):
                raise CaptureValidationError(f"{subject} scenario {scenario_index} must be a JSON object")
            _validate_scenario(
                scenario,
                f"{subject} scenario {scenario_index}",
                required=disposition == _REQUIRED_DISPOSITION,
            )
    return entries


def _git_output(source_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=source_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CaptureValidationError(f"cannot inspect source Git worktree: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise CaptureValidationError(f"cannot inspect source Git worktree: {detail or 'git command failed'}")
    return result.stdout.strip()


def _git_bytes(source_root: Path, *arguments: str) -> bytes:
    """Run a Git object query without ever reading source worktree bytes."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=source_root,
            capture_output=True,
            text=False,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CaptureValidationError(f"cannot read initial HEAD Git objects: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        raise CaptureValidationError(f"cannot read initial HEAD Git objects: {detail or 'git command failed'}")
    return result.stdout


def _source_provenance(source_root: Path) -> dict[str, str | bool]:
    reported_root = Path(_git_output(source_root, "rev-parse", "--show-toplevel")).resolve()
    if reported_root != source_root:
        raise CaptureValidationError("capture must run from the source Git worktree root")
    dirty_status = _git_output(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty_status:
        raise CaptureValidationError("capture requires a clean Git worktree")
    commit = _git_output(source_root, "rev-parse", "--verify", "HEAD")
    tree = _git_output(source_root, "rev-parse", "--verify", "HEAD^{tree}")
    if _GIT_OBJECT_ID.fullmatch(commit) is None or _GIT_OBJECT_ID.fullmatch(tree) is None:
        raise CaptureValidationError("capture requires full commit and tree object IDs")
    return {"commit": commit, "tree": tree, "clean": True}


def _relative_source_path(source_root: Path, path: Path, label: str) -> str:
    """Derive a source path lexically, without following mutable worktree links."""
    lexical_path = path.absolute()
    if ".." in lexical_path.parts:
        raise CaptureValidationError(f"{label} must not contain lexical parent-directory traversal")
    try:
        relative = lexical_path.relative_to(source_root)
    except ValueError as exc:
        raise CaptureValidationError(f"{label} must be inside the source worktree") from exc
    return relative.as_posix()


def _git_tree_blobs(source_root: Path, commit: str, relative_path: str) -> dict[str, str]:
    """Return immutable blob IDs under one literal path in ``commit``'s tree."""
    output = _git_bytes(
        source_root,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        commit,
        "--",
        f":(literal){relative_path}",
    )
    blobs: dict[str, str] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", maxsplit=1)
            mode, object_type, raw_object_id = metadata.split(b" ", maxsplit=2)
            object_id = raw_object_id.decode("ascii")
            path = os.fsdecode(raw_path)
        except (UnicodeDecodeError, ValueError) as exc:
            raise CaptureValidationError("initial HEAD tree contains an invalid Git entry") from exc
        if mode not in {b"100644", b"100755"} or object_type != b"blob":
            raise CaptureValidationError(f"initial HEAD path {path!r} must contain regular-file blobs only")
        if _GIT_OBJECT_ID.fullmatch(object_id) is None:
            raise CaptureValidationError(f"initial HEAD path {path!r} has an invalid blob object ID")
        if path in blobs:
            raise CaptureValidationError(f"initial HEAD tree repeats path {path!r}")
        blobs[path] = object_id
    return blobs


def _git_blob_bytes(source_root: Path, object_id: str) -> bytes:
    return _git_bytes(source_root, "cat-file", "blob", object_id)


def _initial_head_blob_bytes(source_root: Path, commit: str, relative_path: str, label: str) -> bytes:
    blobs = _git_tree_blobs(source_root, commit, relative_path)
    if set(blobs) != {relative_path}:
        raise CaptureValidationError(f"{label} must be a regular file in the initial HEAD tree")
    return _git_blob_bytes(source_root, blobs[relative_path])


def _frozen_tooling_identity(tooling_root: Path) -> tuple[Path, bytes, dict[str, Any]]:
    """Return the clean static tooling root, checker bytes, and immutable identity."""
    resolved_root = tooling_root.resolve()
    expected_capture_path = resolved_root / _CAPTURE_TOOL_RELATIVE
    actual_capture_path = Path(__file__).resolve()
    if expected_capture_path.resolve() != actual_capture_path:
        raise CaptureValidationError("capture must execute from the supplied frozen tooling root")
    tooling_provenance = _source_provenance(resolved_root)
    tooling_commit = tooling_provenance["commit"]
    if not isinstance(tooling_commit, str):
        raise CaptureValidationError("capture requires a string frozen tooling commit")
    capture_payload = _initial_head_blob_bytes(
        resolved_root,
        tooling_commit,
        _CAPTURE_TOOL_RELATIVE,
        "frozen capture tooling",
    )
    checker_payload = _initial_head_blob_bytes(
        resolved_root,
        tooling_commit,
        _REPOSITORY_SURFACE_CHECKER_RELATIVE,
        "frozen repository-surface disposition checker",
    )
    return (
        resolved_root,
        checker_payload,
        {
            "root": str(resolved_root),
            "source": tooling_provenance,
            "capture": {
                "path": _CAPTURE_TOOL_RELATIVE,
                "sha256": _sha256_bytes(capture_payload),
            },
            "repository_surface_disposition_checker": {
                "path": _REPOSITORY_SURFACE_CHECKER_RELATIVE,
                "sha256": _sha256_bytes(checker_payload),
            },
        },
    )


def _portable_fixture_relative_path(value: str, subject: str) -> str:
    """Return one canonical, non-escaping POSIX fixture-manifest key."""
    parts = value.split("/")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise CaptureValidationError(f"{subject} golden_path must be a portable fixture-relative POSIX path")
    normalized = "/".join(parts)
    if PurePosixPath(normalized).as_posix() != normalized:
        raise CaptureValidationError(f"{subject} golden_path must be a portable fixture-relative POSIX path")
    return normalized


def _fixture_manifest_from_initial_head(
    source_root: Path, commit: str, fixture_dir_relative: str
) -> dict[str, dict[str, str]]:
    blobs = _git_tree_blobs(source_root, commit, fixture_dir_relative)
    prefix = "" if fixture_dir_relative == "." else f"{fixture_dir_relative.rstrip('/')}/"
    manifest: dict[str, dict[str, str]] = {}
    for source_relative, object_id in blobs.items():
        if prefix and not source_relative.startswith(prefix):
            raise CaptureValidationError("initial HEAD fixture manifest escapes the requested fixture directory")
        fixture_relative = source_relative[len(prefix) :] if prefix else source_relative
        fixture_key = _portable_fixture_relative_path(fixture_relative, f"fixture {fixture_relative!r}")
        if fixture_key in manifest:
            raise CaptureValidationError(f"initial HEAD fixture manifest repeats path {fixture_key!r}")
        manifest[fixture_key] = {"sha256": _sha256_bytes(_git_blob_bytes(source_root, object_id))}
    if not manifest:
        raise CaptureValidationError("fixture directory must contain at least one file in the initial HEAD tree")
    return manifest


def _validate_ledger_fixture_paths(
    ledger_entries: list[dict[str, Any]], fixture_manifest: dict[str, dict[str, str]]
) -> None:
    for entry_index, entry in enumerate(ledger_entries):
        for scenario_index, scenario in enumerate(entry["scenarios"]):
            golden_path = scenario.get("golden_path")
            if not _non_empty_string(golden_path):
                continue
            assert isinstance(golden_path, str)
            subject = f"ledger entry {entry_index} scenario {scenario_index}"
            fixture_key = _portable_fixture_relative_path(golden_path, subject)
            if fixture_key not in fixture_manifest:
                raise CaptureValidationError(
                    f"{subject} golden_path does not exist in the initial HEAD fixture manifest: {fixture_key}"
                )


def _validate_repository_surface_inputs(
    facts_payload: bytes,
    disposition_payload: bytes,
    *,
    facts_filename: str,
    checker_payload: bytes,
    checker_filename: str,
) -> dict[str, Any]:
    """Validate inputs with the exact checker bytes recorded from frozen tooling."""
    checker = types.ModuleType("fincore_0042_r2_repository_surface_checker")
    checker.__file__ = checker_filename
    try:
        checker_code = compile(checker_payload, checker_filename, "exec")
        exec(checker_code, checker.__dict__)
    except (ImportError, SyntaxError, TypeError, ValueError) as exc:
        raise CaptureValidationError(f"cannot load static repository-surface disposition checker: {exc}") from exc
    validator = getattr(checker, "validate_disposition_payloads", None)
    if not callable(validator):
        raise CaptureValidationError("static repository-surface disposition checker lacks byte validation")
    try:
        summary = validator(
            facts_payload,
            disposition_payload,
            facts_filename=facts_filename,
        )
    except (TypeError, ValueError) as exc:
        raise CaptureValidationError(f"repository-surface disposition validation failed: {exc}") from exc
    if not isinstance(summary, dict) or summary.get("not_for_d0") is not True:
        raise CaptureValidationError("repository-surface disposition validation must remain explicitly not_for_d0")
    return summary


def _atomic_write_json(output: Path, artifact: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(artifact, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def capture(
    *,
    source_root: Path,
    tooling_root: Path,
    inventory_path: Path,
    module_disposition_path: Path,
    test_disposition_path: Path,
    ledger_path: Path,
    repository_surface_facts_path: Path,
    repository_surface_disposition_path: Path,
    fixture_dir: Path,
    output_path: Path,
    deny_network: bool,
) -> dict[str, Any]:
    """Validate inputs, then atomically persist one provenance-only capture artifact."""
    if not deny_network:
        raise CaptureValidationError("--deny-network is required for fail-closed capture")
    source_root = source_root.resolve()
    requested_tooling_root = tooling_root.resolve()
    try:
        source_root.relative_to(requested_tooling_root)
        roots_overlap = True
    except ValueError:
        try:
            requested_tooling_root.relative_to(source_root)
            roots_overlap = True
        except ValueError:
            roots_overlap = False
    if roots_overlap:
        raise CaptureValidationError("frozen tooling root must be distinct from the source worktree")
    tooling_root, checker_payload, tooling_identity = _frozen_tooling_identity(requested_tooling_root)
    output_path = output_path.resolve()
    try:
        output_path.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise CaptureValidationError("output must be outside the source worktree")
    try:
        output_path.relative_to(tooling_root)
    except ValueError:
        pass
    else:
        raise CaptureValidationError("output must be outside the frozen tooling worktree")

    provenance = _source_provenance(source_root)
    initial_commit = provenance["commit"]
    if not isinstance(initial_commit, str):
        raise CaptureValidationError("capture requires a string initial source commit")
    input_paths = {
        "inventory": inventory_path,
        "module_disposition": module_disposition_path,
        "test_disposition": test_disposition_path,
        "ledger": ledger_path,
        "repository_surface_facts": repository_surface_facts_path,
        "repository_surface_disposition": repository_surface_disposition_path,
    }
    input_labels = {
        "inventory": "inventory",
        "module_disposition": "module",
        "test_disposition": "test",
        "ledger": "ledger",
        "repository_surface_facts": "repository-surface facts",
        "repository_surface_disposition": "repository-surface disposition",
    }
    documents: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    inputs: dict[str, dict[str, str]] = {}
    for key, path in input_paths.items():
        relative = _relative_source_path(source_root, path, key.replace("_", " "))
        payload = _initial_head_blob_bytes(source_root, initial_commit, relative, key.replace("_", " "))
        payloads[key] = payload
        inputs[key] = {"path": relative, "sha256": _sha256_bytes(payload)}
    for key in ("inventory", "module_disposition", "test_disposition", "ledger"):
        documents[key] = _load_json_bytes(payloads[key], input_labels[key])

    _validate_disposition_document(documents["inventory"], input_labels["inventory"])
    _validate_disposition_document(documents["module_disposition"], input_labels["module_disposition"])
    _validate_disposition_document(documents["test_disposition"], input_labels["test_disposition"])
    ledger_entries = validate_ledger(documents["ledger"])
    repository_surface_summary = _validate_repository_surface_inputs(
        payloads["repository_surface_facts"],
        payloads["repository_surface_disposition"],
        facts_filename=PurePosixPath(inputs["repository_surface_facts"]["path"]).name,
        checker_payload=checker_payload,
        checker_filename=str(tooling_root / _REPOSITORY_SURFACE_CHECKER_RELATIVE),
    )
    if (
        repository_surface_summary.get("facts_sha256") != inputs["repository_surface_facts"]["sha256"]
        or repository_surface_summary.get("disposition_sha256") != inputs["repository_surface_disposition"]["sha256"]
    ):
        raise CaptureValidationError("repository-surface disposition checker returned mismatched input digests")
    fixture_dir_relative = _relative_source_path(source_root, fixture_dir, "fixture directory")
    fixture_manifest = _fixture_manifest_from_initial_head(source_root, initial_commit, fixture_dir_relative)
    _validate_ledger_fixture_paths(ledger_entries, fixture_manifest)
    final_provenance = _source_provenance(source_root)
    if final_provenance != provenance:
        raise CaptureValidationError("source Git provenance changed during capture")
    final_tooling_provenance = _source_provenance(tooling_root)
    if final_tooling_provenance != tooling_identity["source"]:
        raise CaptureValidationError("frozen tooling Git provenance changed during capture")

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "capability_baseline_capture",
        "capture_status": "captured",
        "does_not_assert": ["D0", "D-TECH"],
        "captured_at": datetime.now(UTC).isoformat(),
        "source": provenance,
        "tooling": tooling_identity,
        "inputs": inputs,
        "repository_surface": {
            "scope": "classified_repository_surface_only",
            "not_for_d0": True,
            "facts_sha256": inputs["repository_surface_facts"]["sha256"],
            "disposition_sha256": inputs["repository_surface_disposition"]["sha256"],
            "validation": repository_surface_summary,
        },
        "fixtures": fixture_manifest,
        "ledger_summary": {
            "entries": len(ledger_entries),
            "required_entries": sum(entry["disposition"] == _REQUIRED_DISPOSITION for entry in ledger_entries),
        },
    }
    _atomic_write_json(output_path, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, help="versioned legacy-surface inventory JSON")
    parser.add_argument("--module-disposition", required=True, help="module disposition JSON")
    parser.add_argument("--test-disposition", required=True, help="test-node disposition JSON")
    parser.add_argument("--ledger", required=True, help="versioned capability ledger JSON")
    parser.add_argument("--repository-surface-facts", required=True, help="scoped repository-surface facts JSON")
    parser.add_argument(
        "--repository-surface-disposition",
        required=True,
        help="reviewed scoped repository-surface disposition JSON",
    )
    parser.add_argument("--tooling-root", required=True, help="clean worktree containing this frozen capture tooling")
    parser.add_argument("--fixture-dir", required=True, help="directory containing golden fixtures")
    parser.add_argument("--output", required=True, help="repository-external capture artifact path")
    parser.add_argument(
        "--deny-network", action="store_true", help="acknowledge that capture must make no network calls"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        artifact = capture(
            source_root=Path.cwd(),
            tooling_root=Path(args.tooling_root),
            inventory_path=Path(args.inventory),
            module_disposition_path=Path(args.module_disposition),
            test_disposition_path=Path(args.test_disposition),
            ledger_path=Path(args.ledger),
            repository_surface_facts_path=Path(args.repository_surface_facts),
            repository_surface_disposition_path=Path(args.repository_surface_disposition),
            fixture_dir=Path(args.fixture_dir),
            output_path=Path(args.output),
            deny_network=args.deny_network,
        )
    except (CaptureValidationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"captured capability baseline inputs for {artifact['source']['commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
