#!/usr/bin/env python3
"""Collect a deterministic, raw 0042-R2 legacy-surface discovery artifact.

This command is intentionally a discovery boundary, not a reconciliation
boundary.  It records source facts from a clean initial Git ``HEAD`` only and
marks the resulting JSON as partial and unusable for D0 or for capability
baseline capture.  In particular, no ledger ownership, disposition, target
operation, or oracle decision is inferred here.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40,64}$")
_FORBIDDEN_DECISION_FIELDS = frozenset({"owner", "disposition", "target_operation_id", "oracle"})


@dataclass(frozen=True)
class SourceSpec:
    """One source artifact required by this deliberately partial discovery."""

    source_id: str
    path: str


@dataclass(frozen=True)
class SourceArtifact:
    """A regular source blob read from the initial commit, never the worktree."""

    source_id: str
    path: str
    sha256: str
    payload: bytes


class DiscoveryError(RuntimeError):
    """Raised when exact-source discovery cannot be completed safely."""


_SOURCE_SPECS = (
    SourceSpec("metric_registry", "fincore/_registry.py"),
    SourceSpec("workflow_registry", "fincore/contracts/workflows.py"),
    SourceSpec("performance_operation_specs", "fincore/api/builtins.py"),
    SourceSpec("alphalens_function_specs", "fincore/contracts/factor_analysis.py"),
    SourceSpec("alphalens_workflow_specs", "fincore/contracts/factor_workflows.py"),
    SourceSpec("public_api_snapshot", "tests/contracts/fixtures/public-api-0.4.0.dev0.json"),
    SourceSpec("empyrical_compat_manifest", "tests/compat/fixtures/empyrical-0.6.0-api.json"),
    SourceSpec("pyfolio_compat_manifest", "tests/compat/fixtures/pyfolio-0.9.6-api.json"),
    SourceSpec("alphalens_compat_manifest", "tests/compat/fixtures/alphalens-0.4.0-cloudquant-api.json"),
    SourceSpec("capability_registry", "fincore/capabilities.py"),
    SourceSpec("distribution_extras", "pyproject.toml"),
    SourceSpec("installed_consumer_profiles", "scripts/test_installed_wheel.py"),
    SourceSpec("pyfolio_class_methods", "fincore/_pyfolio_impl.py"),
)
_REQUIRED_SOURCE_KINDS = tuple(sorted(spec.source_id for spec in _SOURCE_SPECS))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _controlled_git_environment() -> dict[str, str]:
    """Keep Git discovery rooted at ``cwd`` instead of inherited Git state."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
    )
    return environment


def _git_bytes(source_root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", "-c", "core.fsmonitor=false", *arguments],
            cwd=source_root,
            capture_output=True,
            check=False,
            timeout=30,
            env=_controlled_git_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DiscoveryError(f"cannot inspect source Git worktree: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        raise DiscoveryError(f"cannot inspect source Git worktree: {detail or 'git command failed'}")
    return result.stdout


def _git_text(source_root: Path, *arguments: str) -> str:
    return _git_bytes(source_root, *arguments).decode("utf-8", errors="strict").strip()


def _validate_repo_relative_path(path: str) -> None:
    pure_path = PurePosixPath(path)
    if (
        not path
        or path != str(pure_path)
        or pure_path.is_absolute()
        or "\\" in path
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise DiscoveryError(f"source artifact path must be a repository-relative POSIX path: {path!r}")


def _require_worktree_root(source_root: Path) -> Path:
    top_level = Path(_git_text(source_root, "rev-parse", "--show-toplevel")).resolve()
    if source_root.resolve() != top_level:
        raise DiscoveryError("run this command from the clean Git worktree root")
    return top_level


def _is_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path is a root or descendant of ``root``."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _git_control_paths(source_root: Path) -> tuple[Path, ...]:
    """Resolve every Git administration root that must never be an output target."""
    raw_paths = (
        source_root / ".git",
        Path(_git_text(source_root, "rev-parse", "--git-dir")),
        Path(_git_text(source_root, "rev-parse", "--git-common-dir")),
    )
    resolved_paths: list[Path] = []
    for raw_path in raw_paths:
        candidate = raw_path if raw_path.is_absolute() else source_root / raw_path
        resolved = candidate.resolve()
        if resolved not in resolved_paths:
            resolved_paths.append(resolved)
    return tuple(resolved_paths)


def _reject_git_control_output(source_root: Path, output: Path) -> None:
    """Fail before writing anywhere in the worktree's Git administration data."""
    for control_path in _git_control_paths(source_root):
        if _is_within(output, control_path):
            raise DiscoveryError("output must not target the Git control directory")


def _provenance(source_root: Path) -> dict[str, Any]:
    _require_worktree_root(source_root)
    dirty = _git_text(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise DiscoveryError("source Git worktree must be clean before raw discovery")
    commit = _git_text(source_root, "rev-parse", "HEAD")
    tree = _git_text(source_root, "rev-parse", "HEAD^{tree}")
    if not _GIT_OBJECT_ID.fullmatch(commit) or not _GIT_OBJECT_ID.fullmatch(tree):
        raise DiscoveryError("source Git HEAD and tree must resolve to object identifiers")
    return {"commit": commit, "tree": tree, "clean": True}


def _verify_provenance(source_root: Path, initial: Mapping[str, Any]) -> None:
    final = _provenance(source_root)
    if final != dict(initial):
        raise DiscoveryError("source Git provenance changed while raw discovery was running")


def _read_regular_blob(source_root: Path, commit: str, spec: SourceSpec) -> SourceArtifact:
    _validate_repo_relative_path(spec.path)
    records = [
        record for record in _git_bytes(source_root, "ls-tree", "-z", commit, "--", spec.path).split(b"\0") if record
    ]
    if len(records) != 1:
        raise DiscoveryError(f"source artifact is absent or ambiguous in initial HEAD: {spec.path}")
    try:
        metadata, raw_path = records[0].split(b"\t", 1)
        mode, object_type, _object_id = metadata.decode("ascii").split()
        listed_path = raw_path.decode("utf-8", errors="strict")
    except (UnicodeDecodeError, ValueError) as exc:
        raise DiscoveryError(f"cannot inspect initial HEAD source artifact: {spec.path}") from exc
    if listed_path != spec.path or object_type != "blob" or not mode.startswith("100"):
        raise DiscoveryError(f"source artifact must be a regular Git blob, not a link or tree: {spec.path}")
    payload = _git_bytes(source_root, "show", f"{commit}:{spec.path}")
    return SourceArtifact(spec.source_id, spec.path, _sha256_bytes(payload), payload)


def _load_json(artifact: SourceArtifact) -> dict[str, Any]:
    try:
        document = json.loads(artifact.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiscoveryError(f"cannot parse initial HEAD JSON blob: {artifact.path}") from exc
    if not isinstance(document, dict):
        raise DiscoveryError(f"JSON source must be an object: {artifact.path}")
    return document


def _source_locator(artifact: SourceArtifact, locator: str) -> dict[str, str]:
    return {
        "artifact_path": artifact.path,
        "artifact_sha256": artifact.sha256,
        "locator": locator,
    }


def _json_value(value: Any) -> Any:
    """Convert selected source facts to deterministic, JSON-safe primitives."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | str):
        return [_json_value(item) for item in value]
    return str(value)


def _entry(
    *,
    source_id: str,
    entry_key: str,
    origin: Mapping[str, Any],
    surface: Mapping[str, Any],
    concept: Mapping[str, Any],
    source_locator: Mapping[str, Any],
    facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "entry_id": f"{source_id}:{entry_key}",
        "source_id": source_id,
        "source_kind": source_id,
        "origin": _json_value(origin),
        "surface": _json_value(surface),
        "concept": _json_value(concept),
        "source_locator": _json_value(source_locator),
    }
    if facts:
        record["facts"] = _json_value({key: value for key, value in facts.items() if value is not None})
    return record


def _origin(legacy_family: str, project: str, version: str | None) -> dict[str, str]:
    result = {"legacy_family": legacy_family, "project": project}
    if version:
        result["version"] = version
    return result


def _metric_entries(
    artifact: SourceArtifact,
    metric_registry: Mapping[tuple[str, str, str], Any],
    surface_paths: Mapping[str, str],
    project_version: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for (surface_name, public_name, variant), spec in sorted(metric_registry.items(), key=lambda item: item[0]):
        module = surface_paths.get(surface_name)
        if not isinstance(module, str) or not module:
            raise DiscoveryError(f"metric registry has no discovered public module for {surface_name!r}")
        legacy_family = "empyrical" if surface_name.startswith("empyrical") else "fincore_metrics"
        entries.append(
            _entry(
                source_id=artifact.source_id,
                entry_key=f"{surface_name}:{public_name}:{variant}",
                origin=_origin(legacy_family, "fincore", project_version),
                surface={
                    "module": module,
                    "public_path": f"{module}.{public_name}",
                    "member": public_name,
                    "kind": "metric_binding",
                    "surface_name": surface_name,
                    "variant": variant,
                },
                concept={
                    "source_key": f"{surface_name}:{public_name}:{variant}",
                    "relation": "metric_registry_binding",
                },
                source_locator=_source_locator(
                    artifact, f"METRIC_REGISTRY[{surface_name!r}, {public_name!r}, {variant!r}]"
                ),
                facts={
                    "kernel_ref": spec.kernel_ref,
                    "adapter_ref": spec.adapter_ref,
                    "signature_manifest_key": spec.signature_manifest_key,
                    "binding": spec.binding,
                    "validation_profile": spec.validation_profile,
                    "result_contract_key": spec.result_contract_key,
                    "result_projection": spec.result_projection,
                    "out_policy": spec.out_policy,
                },
            )
        )
    return entries


def _workflow_entries(
    artifact: SourceArtifact,
    workflow_registry: Mapping[tuple[str, str, str], Any],
    surface_paths: Mapping[str, str],
    project_version: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for (surface_name, public_name, variant), spec in sorted(workflow_registry.items(), key=lambda item: item[0]):
        module = surface_paths.get(surface_name)
        if not isinstance(module, str) or not module:
            raise DiscoveryError(f"workflow registry has no discovered public module for {surface_name!r}")
        entries.append(
            _entry(
                source_id=artifact.source_id,
                entry_key=f"{surface_name}:{public_name}:{variant}",
                origin=_origin("pyfolio", "fincore", project_version),
                surface={
                    "module": module,
                    "public_path": f"{module}.{public_name}",
                    "member": public_name,
                    "kind": "workflow_binding",
                    "surface_name": surface_name,
                    "variant": variant,
                },
                concept={
                    "source_key": f"{surface_name}:{public_name}:{variant}",
                    "relation": "workflow_registry_binding",
                },
                source_locator=_source_locator(
                    artifact, f"WORKFLOW_REGISTRY[{surface_name!r}, {public_name!r}, {variant!r}]"
                ),
                facts={
                    "signature_manifest_key": spec.signature_manifest_key,
                    "workflow_ref": spec.workflow_ref,
                    "adapter_ref": spec.adapter_ref,
                    "validation_profile": spec.validation_profile,
                    "result_contract_key": spec.result_contract_key,
                    "result_projection": spec.result_projection,
                },
            )
        )
    return entries


def _performance_entries(
    artifact: SourceArtifact, specs: Sequence[Sequence[str]], project_version: str
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in specs:
        if not isinstance(item, tuple) or len(item) != 4 or not all(isinstance(value, str) for value in item):
            raise DiscoveryError("PERFORMANCE_OPERATION_SPECS must contain four-string source tuples")
        name, kernel_ref, signature, result_projection = item
        module_name, _, implementation_member = kernel_ref.partition(":")
        entries.append(
            _entry(
                source_id=artifact.source_id,
                entry_key=name,
                origin=_origin("fincore_performance", "fincore", project_version),
                surface={
                    "module": module_name,
                    "public_path": f"{module_name}.{implementation_member}",
                    "member": name,
                    "kind": "candidate_projection",
                },
                concept={"source_key": name, "relation": "performance_candidate_projection"},
                source_locator=_source_locator(artifact, f"PERFORMANCE_OPERATION_SPECS[{name!r}]"),
                facts={"kernel_ref": kernel_ref, "signature": signature, "result_projection": result_projection},
            )
        )
    return entries


def _factor_function_entries(artifact: SourceArtifact, specs: Mapping[tuple[str, str], Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for (module, public_name), spec in sorted(specs.items(), key=lambda item: item[0]):
        profile = str(spec.profile)
        entries.append(
            _entry(
                source_id=artifact.source_id,
                entry_key=f"{module}:{public_name}",
                origin=_origin("alphalens", "fincore", profile),
                surface={
                    "module": f"fincore.alphalens.{module}",
                    "public_path": f"fincore.alphalens.{module}.{public_name}",
                    "member": public_name,
                    "kind": "factor_function_contract",
                },
                concept={"source_key": f"{module}:{public_name}", "relation": "factor_function_contract"},
                source_locator=_source_locator(artifact, f"ALPHALENS_FUNCTION_SPECS[{module!r}, {public_name!r}]"),
                facts={
                    "source_signature": str(spec.source_signature),
                    "introspection_signature": str(spec.introspection_signature),
                    "implementation": spec.implementation,
                    "profile": profile,
                    "optional_extra": spec.optional_extra,
                    "adapter": spec.adapter,
                    "result_projection": spec.result_projection,
                },
            )
        )
    return entries


def _factor_workflow_entries(artifact: SourceArtifact, specs: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for public_name, spec in sorted(specs.items()):
        entries.append(
            _entry(
                source_id=artifact.source_id,
                entry_key=public_name,
                origin=_origin("alphalens", "fincore", "legacy_alphalens_cloudquant_0_4_0"),
                surface={
                    "module": "fincore.alphalens.tears",
                    "public_path": f"fincore.alphalens.tears.{public_name}",
                    "member": public_name,
                    "kind": "factor_workflow_contract",
                },
                concept={"source_key": public_name, "relation": "factor_workflow_contract"},
                source_locator=_source_locator(artifact, f"ALPHALENS_WORKFLOW_SPECS[{public_name!r}]"),
                facts={
                    "source_signature": str(spec.source_signature),
                    "introspection_signature": str(spec.introspection_signature),
                    "model_ref": spec.model_ref,
                    "renderer_ref": spec.renderer_ref,
                    "optional_extra": spec.optional_extra,
                    "result_projection": spec.result_projection,
                    "by_group_variants": list(spec.by_group_variants),
                },
            )
        )
    return entries


def _snapshot_entries(artifact: SourceArtifact, document: Mapping[str, Any]) -> list[dict[str, Any]]:
    surfaces = document.get("surfaces")
    if not isinstance(surfaces, Mapping) or not surfaces:
        raise DiscoveryError("public API snapshot must contain non-empty surfaces")
    baseline = document.get("baseline")
    project = document.get("project")
    if not isinstance(baseline, str) or not isinstance(project, str):
        raise DiscoveryError("public API snapshot must state its project and baseline")
    entries: list[dict[str, Any]] = []
    for module, surface_document in sorted(surfaces.items(), key=lambda item: str(item[0])):
        if not isinstance(module, str) or not isinstance(surface_document, Mapping):
            raise DiscoveryError("public API snapshot surface must be a named object")
        profile = surface_document.get("profile")
        symbols = surface_document.get("entries")
        if not isinstance(symbols, Mapping):
            raise DiscoveryError(f"public API snapshot surface has no entry map: {module}")
        for symbol, detail in sorted(symbols.items(), key=lambda item: str(item[0])):
            if not isinstance(symbol, str) or not isinstance(detail, Mapping):
                raise DiscoveryError(f"public API snapshot has an invalid entry under {module}")
            public_path = detail.get("public_path")
            if not isinstance(public_path, str) or not public_path:
                raise DiscoveryError(f"public API snapshot entry has no public_path: {module}.{symbol}")
            entries.append(
                _entry(
                    source_id=artifact.source_id,
                    entry_key=public_path,
                    origin=_origin("fincore_public_snapshot", project, baseline),
                    surface={
                        "module": module,
                        "public_path": public_path,
                        "member": symbol,
                        "kind": detail.get("kind", "unknown"),
                        "profile": profile,
                    },
                    concept={"source_key": f"{module}:{symbol}", "relation": "public_api_snapshot_symbol"},
                    source_locator=_source_locator(artifact, f"surfaces.{module}.entries.{symbol}"),
                )
            )
    return entries


def _compat_entry(
    *,
    artifact: SourceArtifact,
    entry_key: str,
    public_path: str,
    source_locator: str,
    project: str,
    version: str | None,
    kind: str,
    signature: str | None,
    source_detail: Mapping[str, Any],
) -> dict[str, Any]:
    module, separator, member = public_path.rpartition(".")
    if not separator or not module or not member:
        raise DiscoveryError(f"compatibility manifest public path is invalid: {public_path!r}")
    facts = {
        "signature": signature,
        "source_signature": source_detail.get("source_signature"),
        "introspection_signature": source_detail.get("introspection_signature"),
        "compatibility": source_detail.get("compatibility"),
        "needs_dynamic_review": source_detail.get("needs_dynamic_review"),
        "source_path": source_detail.get("source_path"),
        "source_line": source_detail.get("source_line"),
        "source_sha256": source_detail.get("source_sha256"),
    }
    return _entry(
        source_id=artifact.source_id,
        entry_key=entry_key,
        origin=_origin(project, project, version),
        surface={"module": module, "public_path": public_path, "member": member, "kind": kind},
        concept={"source_key": entry_key, "relation": "pinned_compatibility_manifest"},
        source_locator=_source_locator(artifact, source_locator),
        facts=facts,
    )


def _empyrical_manifest_entries(artifact: SourceArtifact, document: Mapping[str, Any]) -> list[dict[str, Any]]:
    project = document.get("project")
    version = document.get("version")
    callables = document.get("callables")
    if not isinstance(project, str) or not isinstance(version, str) or not isinstance(callables, list):
        raise DiscoveryError("empyrical compatibility manifest is missing its identity or callables")
    entries: list[dict[str, Any]] = []
    for index, detail in enumerate(callables):
        if not isinstance(detail, Mapping):
            raise DiscoveryError("empyrical compatibility callable must be an object")
        public_path = detail.get("public_path")
        if not isinstance(public_path, str):
            raise DiscoveryError("empyrical compatibility callable must have public_path")
        entries.append(
            _compat_entry(
                artifact=artifact,
                entry_key=public_path,
                public_path=public_path,
                source_locator=f"callables[{index}]",
                project=project,
                version=version,
                kind=str(detail.get("kind", "callable")),
                signature=detail.get("signature") if isinstance(detail.get("signature"), str) else None,
                source_detail=detail,
            )
        )
    return entries


def _pyfolio_manifest_entries(artifact: SourceArtifact, document: Mapping[str, Any]) -> list[dict[str, Any]]:
    project = document.get("project")
    version = document.get("version")
    profile = document.get("compatibility_profile")
    if not isinstance(project, str) or not isinstance(version, str) or not isinstance(profile, Mapping):
        raise DiscoveryError("pyfolio compatibility manifest is missing its identity or profile")
    entries: list[dict[str, Any]] = []
    for key, detail in sorted(profile.items(), key=lambda item: str(item[0])):
        if not isinstance(key, str) or not isinstance(detail, Mapping):
            raise DiscoveryError("pyfolio compatibility profile entry must be a named object")
        public_path = detail.get("public_path")
        if not isinstance(public_path, str):
            raise DiscoveryError("pyfolio compatibility profile entry must have public_path")
        entries.append(
            _compat_entry(
                artifact=artifact,
                entry_key=key,
                public_path=public_path,
                source_locator=f"compatibility_profile.{key}",
                project=project,
                version=version,
                kind=str(detail.get("kind", "callable")),
                signature=detail.get("signature") if isinstance(detail.get("signature"), str) else None,
                source_detail=detail,
            )
        )
    return entries


def _alphalens_manifest_entries(artifact: SourceArtifact, document: Mapping[str, Any]) -> list[dict[str, Any]]:
    project = document.get("project")
    profile = document.get("profile")
    manifest_entries = document.get("entries")
    if not isinstance(project, str) or not isinstance(profile, str) or not isinstance(manifest_entries, list):
        raise DiscoveryError("alphalens compatibility manifest is missing identity, profile, or entries")
    entries: list[dict[str, Any]] = []
    for index, detail in enumerate(manifest_entries):
        if not isinstance(detail, Mapping):
            raise DiscoveryError("alphalens compatibility manifest entry must be an object")
        module = detail.get("module")
        symbol = detail.get("symbol")
        if not isinstance(module, str) or not isinstance(symbol, str):
            raise DiscoveryError("alphalens compatibility manifest entry must have module and symbol")
        entries.append(
            _compat_entry(
                artifact=artifact,
                entry_key=f"{module}:{symbol}",
                public_path=f"alphalens.{module}.{symbol}",
                source_locator=f"entries[{index}]",
                project=project,
                version=profile,
                kind=str(detail.get("kind", "function")),
                signature=detail.get("source_signature") if isinstance(detail.get("source_signature"), str) else None,
                source_detail=detail,
            )
        )
    return entries


def _capability_entries(
    artifact: SourceArtifact, capabilities: Sequence[Any], project_version: str
) -> list[dict[str, Any]]:
    return [
        _entry(
            source_id=artifact.source_id,
            entry_key=capability.id,
            origin=_origin("fincore_capability_registry", "fincore", project_version),
            surface={
                "module": capability.public_path.rpartition(".")[0],
                "public_path": capability.public_path,
                "member": capability.public_path.rsplit(".", 1)[-1],
                "kind": "capability",
            },
            concept={"source_key": capability.id, "relation": "capability_registry_row"},
            source_locator=_source_locator(artifact, f"list_capabilities()[{capability.id!r}]"),
            facts={
                "domain": capability.domain,
                "status": capability.status,
                "input_contract": capability.input_contract,
                "output_contract": capability.output_contract,
                "docs_path": capability.docs_path,
                "rationale": capability.rationale,
            },
        )
        for capability in sorted(capabilities, key=lambda item: item.id)
    ]


def _extra_entries(artifact: SourceArtifact, document: Mapping[str, Any], project_version: str) -> list[dict[str, Any]]:
    project = document.get("project")
    if not isinstance(project, Mapping):
        raise DiscoveryError("pyproject source must contain a project table")
    extras = project.get("optional-dependencies")
    if not isinstance(extras, Mapping) or not extras:
        raise DiscoveryError("pyproject source must contain non-empty optional dependencies")
    entries: list[dict[str, Any]] = []
    for name, requirements in sorted(extras.items(), key=lambda item: str(item[0])):
        if (
            not isinstance(name, str)
            or not isinstance(requirements, list)
            or not all(isinstance(item, str) for item in requirements)
        ):
            raise DiscoveryError("optional dependency entries must be named string lists")
        entries.append(
            _entry(
                source_id=artifact.source_id,
                entry_key=name,
                origin=_origin("fincore_distribution", "fincore", project_version),
                surface={
                    "module": "distribution",
                    "public_path": f"fincore[{name}]",
                    "member": name,
                    "kind": "optional_dependency",
                },
                concept={"source_key": name, "relation": "pep621_optional_dependency"},
                source_locator=_source_locator(artifact, f"project.optional-dependencies.{name}"),
                facts={"requirements": requirements},
            )
        )
    return entries


def _profile_entries(
    artifact: SourceArtifact, profiles: Mapping[str, Any], project_version: str
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name, profile in sorted(profiles.items(), key=lambda item: str(item[0])):
        if not isinstance(name, str) or not isinstance(profile, Mapping):
            raise DiscoveryError("installed-consumer profiles must be named objects")
        entries.append(
            _entry(
                source_id=artifact.source_id,
                entry_key=name,
                origin=_origin("fincore_installed_consumer", "fincore", project_version),
                surface={
                    "module": "scripts.test_installed_wheel",
                    "public_path": f"installed-consumer-profile:{name}",
                    "member": name,
                    "kind": "installed_consumer_profile",
                },
                concept={"source_key": name, "relation": "installed_consumer_profile"},
                source_locator=_source_locator(artifact, f"PROFILES[{name!r}]"),
                facts={"profile": profile},
            )
        )
    return entries


def _pyfolio_method_entries(artifact: SourceArtifact, project_version: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(artifact.payload.decode("utf-8"), filename=artifact.path)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise DiscoveryError("cannot parse Pyfolio class source from initial HEAD blob") from exc
    pyfolio_class = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Pyfolio"),
        None,
    )
    if pyfolio_class is None:
        raise DiscoveryError("initial HEAD Pyfolio source has no Pyfolio class")
    entries: list[dict[str, Any]] = []
    for node in pyfolio_class.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        entries.append(
            _entry(
                source_id=artifact.source_id,
                entry_key=node.name,
                origin=_origin("pyfolio", "fincore", project_version),
                surface={
                    "module": "fincore.pyfolio.Pyfolio",
                    "public_path": f"fincore.pyfolio.Pyfolio.{node.name}",
                    "member": node.name,
                    "kind": "class_method",
                },
                concept={"source_key": node.name, "relation": "pyfolio_class_method"},
                source_locator=_source_locator(artifact, f"Pyfolio.{node.name}:line {node.lineno}"),
                facts={"signature": f"({ast.unparse(node.args)})", "source_line": node.lineno},
            )
        )
    return entries


@contextmanager
def _initial_head_snapshot(source_root: Path, commit: str) -> Iterator[Path]:
    """Materialize a temporary import tree exclusively from initial HEAD bytes."""
    archive = _git_bytes(source_root, "archive", "--format=tar", commit)
    with tempfile.TemporaryDirectory(prefix="fincore-0042-r2-snapshot-") as temporary_directory:
        snapshot = Path(temporary_directory)
        try:
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
                for member in tar.getmembers():
                    pure_path = PurePosixPath(member.name)
                    if pure_path.is_absolute() or any(part in {"", ".", ".."} for part in pure_path.parts):
                        raise DiscoveryError("initial HEAD archive contains an unsafe path")
                    destination = snapshot.joinpath(*pure_path.parts)
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        raise DiscoveryError("initial HEAD archive contains a non-regular import-tree member")
                    source = tar.extractfile(member)
                    if source is None:
                        raise DiscoveryError("cannot read a regular initial HEAD archive member")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(source.read())
                    destination.chmod(member.mode & 0o777)
        except tarfile.TarError as exc:
            raise DiscoveryError("cannot materialize initial HEAD import tree") from exc
        yield snapshot


@contextmanager
def _snapshot_imports(snapshot: Path) -> Iterator[None]:
    """Prefer the snapshot and restore any pre-existing fincore modules afterwards."""
    saved_modules = {
        name: module for name, module in sys.modules.items() if name == "fincore" or name.startswith("fincore.")
    }
    for name in saved_modules:
        del sys.modules[name]
    original_sys_path = list(sys.path)
    sys.path.insert(0, str(snapshot))
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path[:] = original_sys_path
        for name in tuple(sys.modules):
            if name == "fincore" or name.startswith("fincore."):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        importlib.invalidate_caches()


def _import_snapshot_module(module_name: str, snapshot: Path) -> Any:
    module = importlib.import_module(module_name)
    source_file = getattr(module, "__file__", None)
    if not isinstance(source_file, str) or not Path(source_file).resolve().is_relative_to(snapshot.resolve()):
        raise DiscoveryError(f"source module did not load from initial HEAD snapshot: {module_name}")
    return module


def _load_snapshot_script(snapshot: Path) -> Any:
    script_path = snapshot / "scripts" / "test_installed_wheel.py"
    spec = importlib.util.spec_from_file_location("_fincore_0042_r2_installed_profiles", script_path)
    if spec is None or spec.loader is None:
        raise DiscoveryError("cannot load installed-consumer profiles from initial HEAD snapshot")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contains_forbidden_decision_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(set(value) & _FORBIDDEN_DECISION_FIELDS) or any(
            _contains_forbidden_decision_field(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_decision_field(item) for item in value)
    return False


def _validate_raw_entries(entries: Sequence[Mapping[str, Any]], artifacts: Mapping[str, SourceArtifact]) -> None:
    if not entries:
        raise DiscoveryError("raw discovery produced no entries")
    seen_ids: set[str] = set()
    source_counts = dict.fromkeys(artifacts, 0)
    for entry in entries:
        if _contains_forbidden_decision_field(entry):
            raise DiscoveryError("raw discovery entries must not contain ledger or reconciliation decision fields")
        entry_id = entry.get("entry_id")
        source_id = entry.get("source_id")
        if not isinstance(entry_id, str) or not entry_id or entry_id in seen_ids:
            raise DiscoveryError("raw discovery entries must have unique, stable entry_id values")
        if not isinstance(source_id, str) or source_id not in artifacts or entry.get("source_kind") != source_id:
            raise DiscoveryError("raw discovery entry source identity is invalid")
        seen_ids.add(entry_id)
        source_counts[source_id] += 1
        for key in ("origin", "surface", "concept", "source_locator"):
            if not isinstance(entry.get(key), Mapping) or not entry[key]:
                raise DiscoveryError(f"raw discovery entry must include non-empty {key}")
        locator = entry["source_locator"]
        artifact = artifacts[source_id]
        if locator.get("artifact_path") != artifact.path or locator.get("artifact_sha256") != artifact.sha256:
            raise DiscoveryError("raw discovery entry is not bound to its initial HEAD source artifact")
    missing = sorted(source_id for source_id, count in source_counts.items() if count == 0)
    if missing:
        raise DiscoveryError(f"required source kinds produced no raw entries: {', '.join(missing)}")


def _discrepancies() -> list[dict[str, Any]]:
    return [
        {
            "discrepancy_id": "catalog_projection_not_complete_source",
            "source_kinds": ["metric_registry", "workflow_registry", "performance_operation_specs"],
            "statement": (
                "PERFORMANCE_OPERATION_SPECS is a candidate projection/intersection, not a complete source for "
                "the metric or workflow registries."
            ),
        },
        {
            "discrepancy_id": "snapshot_paths_not_equivalent_to_catalog_bindings",
            "source_kinds": ["public_api_snapshot", "metric_registry", "workflow_registry"],
            "statement": (
                "Public API snapshot paths describe exported surfaces and are not a one-to-one completeness assertion "
                "against registry or catalog bindings."
            ),
        },
        {
            "discrepancy_id": "pyfolio_class_methods_not_workflows",
            "source_kinds": ["pyfolio_class_methods", "workflow_registry"],
            "statement": "Pyfolio class methods and the eleven module workflows are distinct raw source surfaces.",
        },
        {
            "discrepancy_id": "factor_contract_manifest_not_one_to_one",
            "source_kinds": [
                "alphalens_compat_manifest",
                "alphalens_function_specs",
                "alphalens_workflow_specs",
            ],
            "statement": "Factor manifest entries, function contracts, and tear-sheet workflows do not merge one-to-one.",
        },
        {
            "discrepancy_id": "distribution_extras_not_installed_profiles",
            "source_kinds": ["distribution_extras", "installed_consumer_profiles"],
            "statement": "PEP 621 distribution extras and installed-consumer test profiles are separate source facts.",
        },
    ]


def _collect_artifact(source_root: Path) -> dict[str, Any]:
    initial = _provenance(source_root)
    artifacts = {spec.source_id: _read_regular_blob(source_root, initial["commit"], spec) for spec in _SOURCE_SPECS}
    pyproject = tomllib.loads(artifacts["distribution_extras"].payload.decode("utf-8"))
    project_table = pyproject.get("project")
    if not isinstance(project_table, Mapping) or not isinstance(project_table.get("version"), str):
        raise DiscoveryError("initial HEAD pyproject must contain a project version")
    project_version = project_table["version"]
    entries: list[dict[str, Any]] = []

    try:
        with _initial_head_snapshot(source_root, initial["commit"]) as snapshot, _snapshot_imports(snapshot):
            registry_module = _import_snapshot_module("fincore._registry", snapshot)
            workflows_module = _import_snapshot_module("fincore.contracts.workflows", snapshot)
            builtins_module = _import_snapshot_module("fincore.api.builtins", snapshot)
            factor_analysis_module = _import_snapshot_module("fincore.contracts.factor_analysis", snapshot)
            factor_workflows_module = _import_snapshot_module("fincore.contracts.factor_workflows", snapshot)
            capabilities_module = _import_snapshot_module("fincore.capabilities", snapshot)
            installed_profiles_module = _load_snapshot_script(snapshot)

            entries.extend(
                _metric_entries(
                    artifacts["metric_registry"],
                    registry_module.METRIC_REGISTRY,
                    builtins_module.SURFACE_PATH,
                    project_version,
                )
            )
            entries.extend(
                _workflow_entries(
                    artifacts["workflow_registry"],
                    workflows_module.WORKFLOW_REGISTRY,
                    builtins_module.SURFACE_PATH,
                    project_version,
                )
            )
            entries.extend(
                _performance_entries(
                    artifacts["performance_operation_specs"],
                    builtins_module.PERFORMANCE_OPERATION_SPECS,
                    project_version,
                )
            )
            entries.extend(
                _factor_function_entries(
                    artifacts["alphalens_function_specs"], factor_analysis_module.ALPHALENS_FUNCTION_SPECS
                )
            )
            entries.extend(
                _factor_workflow_entries(
                    artifacts["alphalens_workflow_specs"], factor_workflows_module.ALPHALENS_WORKFLOW_SPECS
                )
            )
            entries.extend(
                _capability_entries(
                    artifacts["capability_registry"], capabilities_module.list_capabilities(), project_version
                )
            )
            entries.extend(
                _profile_entries(
                    artifacts["installed_consumer_profiles"], installed_profiles_module.PROFILES, project_version
                )
            )
    except DiscoveryError:
        raise
    except Exception as exc:
        raise DiscoveryError(f"cannot collect initial HEAD Python source facts: {exc}") from exc

    entries.extend(_snapshot_entries(artifacts["public_api_snapshot"], _load_json(artifacts["public_api_snapshot"])))
    entries.extend(
        _empyrical_manifest_entries(
            artifacts["empyrical_compat_manifest"], _load_json(artifacts["empyrical_compat_manifest"])
        )
    )
    entries.extend(
        _pyfolio_manifest_entries(
            artifacts["pyfolio_compat_manifest"], _load_json(artifacts["pyfolio_compat_manifest"])
        )
    )
    entries.extend(
        _alphalens_manifest_entries(
            artifacts["alphalens_compat_manifest"], _load_json(artifacts["alphalens_compat_manifest"])
        )
    )
    entries.extend(_extra_entries(artifacts["distribution_extras"], pyproject, project_version))
    entries.extend(_pyfolio_method_entries(artifacts["pyfolio_class_methods"], project_version))
    entries.sort(key=lambda entry: (entry["source_kind"], entry["entry_id"]))
    _validate_raw_entries(entries, artifacts)

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "legacy_surface_discovery",
        "discovery_status": "partial",
        "not_for_d0": True,
        "partial_reason": (
            "This raw discovery intentionally does not collect maintained docs, examples, benchmarks, built wheel "
            "artifacts, or test-node collection. It cannot be used as D0 evidence or as a capability-baseline capture input."
        ),
        "required_source_kinds": list(_REQUIRED_SOURCE_KINDS),
        "source": initial,
        "source_artifacts": [
            {
                "source_id": artifact.source_id,
                "source_kind": artifact.source_id,
                "path": artifact.path,
                "sha256": artifact.sha256,
            }
            for artifact in sorted(artifacts.values(), key=lambda item: item.source_id)
        ],
        "entries": entries,
        "discrepancies": _discrepancies(),
    }
    _verify_provenance(source_root, initial)
    return artifact


def _atomic_write(output: Path, artifact: Mapping[str, Any]) -> None:
    serialized = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary_name).replace(output)
    except Exception:
        with suppress(FileNotFoundError):
            Path(temporary_name).unlink()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="path for the deterministic raw discovery JSON")
    arguments = parser.parse_args(argv)
    source_root = Path.cwd().resolve()
    output = arguments.output if arguments.output.is_absolute() else source_root / arguments.output
    output = output.resolve()

    try:
        _require_worktree_root(source_root)
        _reject_git_control_output(source_root, output)
        source_artifact_paths = {(source_root / spec.path).resolve() for spec in _SOURCE_SPECS}
        if output in source_artifact_paths:
            raise DiscoveryError("output must not overwrite a required source artifact")
        artifact = _collect_artifact(source_root)
        _atomic_write(output, artifact)
    except DiscoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"error: raw discovery failed closed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
