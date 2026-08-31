"""Fail-closed contracts for the scoped 0042-R2 test-node disposition.

Every collected non-online functional test node carries exactly one reviewed
migrate/replace/retire/retain decision.  The artifact is preparatory: it does
not assert D0, D-TECH, installed-wheel behavior, or legacy-zero conclusions.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPOSITORY_ROOT = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).parents[2])).resolve()
NODE_FACTS = REPOSITORY_ROOT / "tests" / "parity" / "fixtures" / "test-node-facts-discovery-0042-r2.json"
DISPOSITION = REPOSITORY_ROOT / "tests" / "parity" / "fixtures" / "test-node-disposition-0042-r2.json"

_DISPOSITIONS = frozenset({"migrate", "replace", "retire", "retain"})
_RETIREMENT_BASES = frozenset({"alias_only", "legacy_quirk"})
_REQUIRED_NON_ASSERTIONS = frozenset({"D0", "D-TECH", "installed_wheel_behavior", "legacy_zero"})
_ENTRY_FIELDS = frozenset({"nodeid", "disposition", "target", "rule_id", "rationale", "retirement_basis"})

_LEGACY_COMPAT_PREFIXES = (
    "tests/compat/",
    "tests/test_empyrical/",
    "tests/test_pyfolio/",
    "tests/test_dispatch_branches.py::",
    "tests/test_smoke_import.py::",
    "tests/test_import_time.py::",
)


def _load() -> tuple[dict, dict]:
    assert NODE_FACTS.is_file(), "committed test-node facts discovery fixture is missing"
    assert DISPOSITION.is_file(), "committed test-node disposition fixture is missing"
    facts = json.loads(NODE_FACTS.read_text(encoding="utf-8"))
    disposition = json.loads(DISPOSITION.read_text(encoding="utf-8"))
    return facts, disposition


def test_disposition_document_header_is_scoped_and_fail_closed() -> None:
    _, disposition = _load()

    assert disposition["schema_version"] == 1
    assert disposition["artifact_type"] == "test_node_disposition"
    assert disposition["scope"] == "non_online_functional_collection_only"
    assert disposition["decision_status"] == "scoped"
    assert disposition["not_for_d0"] is True
    assert set(disposition["does_not_assert"]) >= _REQUIRED_NON_ASSERTIONS


def test_disposition_source_facts_bind_the_committed_node_facts_bytes() -> None:
    facts, disposition = _load()
    source_facts = disposition["source_facts"]

    assert source_facts["path"] == NODE_FACTS.name
    assert source_facts["sha256"] == hashlib.sha256(NODE_FACTS.read_bytes()).hexdigest()
    assert source_facts["node_count"] == len(facts["nodes"])
    assert source_facts["source_provenance"] == facts["source_provenance"]


def test_disposition_maps_each_collected_node_exactly_once() -> None:
    facts, disposition = _load()
    nodeids = sorted(node["nodeid"] for node in facts["nodes"])

    assert [entry["nodeid"] for entry in disposition["entries"]] == nodeids


def test_disposition_entries_follow_the_controlled_vocabulary() -> None:
    _, disposition = _load()

    for entry in disposition["entries"]:
        assert set(entry) <= _ENTRY_FIELDS, entry["nodeid"]
        assert entry["disposition"] in _DISPOSITIONS, entry["nodeid"]
        assert isinstance(entry["target"], str) and entry["target"].strip(), entry["nodeid"]
        assert entry["rule_id"].strip(), entry["nodeid"]
        assert entry["rationale"].strip(), entry["nodeid"]
        if entry["disposition"] == "retire":
            assert entry["retirement_basis"] in _RETIREMENT_BASES, entry["nodeid"]
            assert "capability-ledger-0042-r2.json" in entry["target"], entry["nodeid"]
        else:
            assert "retirement_basis" not in entry, entry["nodeid"]
        if entry["disposition"] == "retain":
            assert entry["target"] == entry["nodeid"], entry["nodeid"]


def test_legacy_compat_nodes_never_claim_retention() -> None:
    _, disposition = _load()

    for entry in disposition["entries"]:
        if entry["nodeid"].startswith(_LEGACY_COMPAT_PREFIXES):
            assert entry["disposition"] in {"migrate", "replace", "retire"}, entry["nodeid"]


def test_every_entry_carries_a_non_empty_disposition_for_capture_input() -> None:
    _, disposition = _load()

    for index, entry in enumerate(disposition["entries"]):
        disposition_value = entry.get("disposition")
        assert isinstance(disposition_value, str) and disposition_value.strip(), f"entry {index}"
