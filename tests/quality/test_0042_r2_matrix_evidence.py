"""Fail-closed contracts for the frozen 0042-R2 gate manifest and matrix schema.

The gate manifest freezes the required gate set and the fail-closed runner
contract.  The matrix schema freezes the per-cell evidence shape that
matrix-aggregate consumes.  Neither artifact asserts D0, D-TECH, or release.
"""

from __future__ import annotations

import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
GATE_MANIFEST = REPOSITORY_ROOT / "tests" / "parity" / "fixtures" / "0042-r2-gate-manifest.json"
MATRIX_SCHEMA = REPOSITORY_ROOT / "tests" / "parity" / "fixtures" / "0042-r2-matrix-evidence.schema.json"

_REQUIRED_GATES = frozenset(
    {
        "tests",
        "static",
        "package",
        "quality",
        "parity",
        "architecture",
        "performance",
        "report",
        "installed",
        "matrix-cell",
        "matrix-aggregate",
        "final",
        "evidence-child",
    }
)

_FINAL_REQUIRES = frozenset(
    {
        "tests",
        "static",
        "package",
        "quality",
        "parity",
        "architecture",
        "performance",
        "report",
        "installed",
        "matrix-aggregate",
    }
)

_MATRIX_REQUIRED_FIELDS = frozenset(
    {
        "matrix_contract_version",
        "candidate_commit",
        "candidate_tree",
        "wheel_sha256",
        "d0_tooling_digest",
        "d0_bundle_digest",
        "os",
        "runner_image",
        "python_full_version",
        "dependency_lane",
        "dependency_profile",
        "argv_digest",
        "output_digest",
        "evidence_time",
        "verdict",
    }
)


def _load(path: Path) -> dict:
    assert path.is_file(), f"committed fixture is missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_gate_manifest_freezes_the_required_gate_set() -> None:
    manifest = _load(GATE_MANIFEST)

    assert manifest["schema_version"] == 1
    assert manifest["artifact_type"] == "gate_manifest"
    assert manifest["not_for_d0"] is True
    assert set(manifest["required_gates"]) == _REQUIRED_GATES
    assert set(manifest["gates"]) == _REQUIRED_GATES


def test_every_gate_is_fail_closed_and_declares_its_evidence_kind() -> None:
    manifest = _load(GATE_MANIFEST)

    for gate_id, spec in manifest["gates"].items():
        assert spec["fail_closed"] is True, gate_id
        assert spec["evidence_kind"].strip(), gate_id
        assert isinstance(spec["thresholds"], list), gate_id
        assert isinstance(spec["requires_candidate_root"], bool), gate_id


def test_final_gate_requires_every_technical_gate() -> None:
    manifest = _load(GATE_MANIFEST)

    assert set(manifest["final_requires_gates"]) == _FINAL_REQUIRES
    assert manifest["gates"]["final"]["consumes_d0_bundle"] is True
    assert manifest["gates"]["matrix-cell"]["consumes_d0_bundle"] is True
    assert manifest["gates"]["evidence-child"]["consumes_d0_bundle"] is False


def test_runner_contract_forbids_candidate_supplied_expected_values() -> None:
    manifest = _load(GATE_MANIFEST)
    contract = manifest["runner_contract"]

    assert contract["candidate_provides_only_actual"] is True
    assert contract["expected_values_come_from_tooling_sha"] is True
    assert contract["verifies_own_blob_before_execution"] is True


def test_evidence_child_policy_freezes_the_two_document_allowlist() -> None:
    manifest = _load(GATE_MANIFEST)
    policy = manifest["evidence_child"]

    assert policy["allow_paths"] == [
        "docs/quality/0042-r2-acceptance.md",
        "docs/quality/0042-r2-evidence-digests.json",
    ]
    assert policy["requires_tested_parent"] is True
    assert policy["single_parent_only"] is True


def test_matrix_schema_freezes_the_cell_contract() -> None:
    schema = _load(MATRIX_SCHEMA)

    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["title"] == "fincore-0042-r2-matrix-cell-evidence"
    assert schema["additionalProperties"] is False
    assert frozenset(schema["required"]) == _MATRIX_REQUIRED_FIELDS
    assert schema["properties"]["matrix_contract_version"]["const"] == 1
    assert set(schema["properties"]["os"]["enum"]) == {"linux", "macos", "windows"}
    assert set(schema["properties"]["verdict"]["enum"]) == {"PASS", "FAIL", "BLOCKED"}
    assert schema["properties"]["dependency_lane"]["enum"] == ["minimum", "latest", "pinned"]


def test_matrix_schema_digest_patterns_are_strict() -> None:
    schema = _load(MATRIX_SCHEMA)
    definitions = schema["definitions"]

    assert definitions["hex_sha256"]["pattern"] == "^[0-9a-f]{64}$"
    assert definitions["git_object_id"]["pattern"] == "^[0-9a-f]{40,64}$"
    for field in ("wheel_sha256", "d0_tooling_digest", "d0_bundle_digest", "argv_digest", "output_digest"):
        assert schema["properties"][field]["$ref"] == "#/definitions/hex_sha256", field
    for field in ("candidate_commit", "candidate_tree"):
        assert schema["properties"][field]["$ref"] == "#/definitions/git_object_id", field
