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
All inputs and all files below ``--fixture-dir`` must be tracked by the clean
source tree.  ``--output`` must be outside that source tree, and is atomically
replaced only after every validation succeeds.
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = 1
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40}$")
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


class CaptureValidationError(ValueError):
    """Raised when a capture input cannot serve as exact-SHA provenance."""


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureValidationError(f"cannot read {label} JSON at {path}: {exc}") from exc
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


def _relative_tracked_path(source_root: Path, path: Path, label: str) -> str:
    try:
        relative = path.resolve().relative_to(source_root)
    except ValueError as exc:
        raise CaptureValidationError(f"{label} must be inside the source worktree") from exc
    if not path.is_file():
        raise CaptureValidationError(f"{label} must be an existing regular file")
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
            cwd=source_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CaptureValidationError(f"cannot verify that {label} is tracked: {exc}") from exc
    if result.returncode != 0:
        raise CaptureValidationError(f"{label} must be tracked by the source worktree")
    return relative.as_posix()


def _fixture_manifest(source_root: Path, fixture_dir: Path) -> dict[str, dict[str, str]]:
    try:
        fixture_dir.resolve().relative_to(source_root)
    except ValueError as exc:
        raise CaptureValidationError("fixture directory must be inside the source worktree") from exc
    if not fixture_dir.is_dir():
        raise CaptureValidationError("fixture directory must exist")
    files = [path for path in sorted(fixture_dir.rglob("*")) if path.is_file()]
    if not files:
        raise CaptureValidationError("fixture directory must contain at least one file")
    manifest: dict[str, dict[str, str]] = {}
    for path in files:
        relative = path.relative_to(fixture_dir).as_posix()
        _relative_tracked_path(source_root, path, f"fixture {relative}")
        manifest[relative] = {"sha256": _sha256(path)}
    return manifest


def _validate_ledger_fixture_paths(ledger_entries: list[dict[str, Any]], fixture_dir: Path) -> None:
    resolved_fixture_dir = fixture_dir.resolve()
    for entry_index, entry in enumerate(ledger_entries):
        for scenario_index, scenario in enumerate(entry["scenarios"]):
            golden_path = scenario.get("golden_path")
            if not _non_empty_string(golden_path):
                continue
            assert isinstance(golden_path, str)
            candidate = (resolved_fixture_dir / golden_path).resolve()
            try:
                candidate.relative_to(resolved_fixture_dir)
            except ValueError as exc:
                raise CaptureValidationError(
                    f"ledger entry {entry_index} scenario {scenario_index} golden_path escapes fixture directory"
                ) from exc
            if not candidate.is_file():
                raise CaptureValidationError(
                    f"ledger entry {entry_index} scenario {scenario_index} golden_path does not exist: {golden_path}"
                )


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
    inventory_path: Path,
    module_disposition_path: Path,
    test_disposition_path: Path,
    ledger_path: Path,
    fixture_dir: Path,
    output_path: Path,
    deny_network: bool,
) -> dict[str, Any]:
    """Validate inputs, then atomically persist one provenance-only capture artifact."""
    if not deny_network:
        raise CaptureValidationError("--deny-network is required for fail-closed capture")
    source_root = source_root.resolve()
    output_path = output_path.resolve()
    try:
        output_path.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise CaptureValidationError("output must be outside the source worktree")

    provenance = _source_provenance(source_root)
    input_paths = {
        "inventory": inventory_path.resolve(),
        "module_disposition": module_disposition_path.resolve(),
        "test_disposition": test_disposition_path.resolve(),
        "ledger": ledger_path.resolve(),
    }
    input_labels = {
        "inventory": "inventory",
        "module_disposition": "module",
        "test_disposition": "test",
        "ledger": "ledger",
    }
    documents: dict[str, dict[str, Any]] = {}
    inputs: dict[str, dict[str, str]] = {}
    for key, path in input_paths.items():
        relative = _relative_tracked_path(source_root, path, key.replace("_", " "))
        documents[key] = _load_json(path, key.replace("_", " "))
        inputs[key] = {"path": relative, "sha256": _sha256(path)}

    _validate_disposition_document(documents["inventory"], input_labels["inventory"])
    _validate_disposition_document(documents["module_disposition"], input_labels["module_disposition"])
    _validate_disposition_document(documents["test_disposition"], input_labels["test_disposition"])
    ledger_entries = validate_ledger(documents["ledger"])
    fixture_manifest = _fixture_manifest(source_root, fixture_dir.resolve())
    _validate_ledger_fixture_paths(ledger_entries, fixture_dir)
    final_provenance = _source_provenance(source_root)
    if final_provenance != provenance:
        raise CaptureValidationError("source Git provenance changed during capture")

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "capability_baseline_capture",
        "capture_status": "captured",
        "does_not_assert": ["D0", "D-TECH"],
        "captured_at": datetime.now(UTC).isoformat(),
        "source": provenance,
        "inputs": inputs,
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
            inventory_path=Path(args.inventory),
            module_disposition_path=Path(args.module_disposition),
            test_disposition_path=Path(args.test_disposition),
            ledger_path=Path(args.ledger),
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
