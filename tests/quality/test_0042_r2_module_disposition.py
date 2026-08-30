"""Fail-closed contracts for the scoped 0042-R2 module disposition.

The disposition binds one reviewed keep/move/delete decision to every raw
module fact.  It is a preparatory decision artifact: it does not assert D0,
D-TECH, installed-wheel behavior, or legacy-zero conclusions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
MODULE_FACTS = REPOSITORY_ROOT / "tests" / "parity" / "fixtures" / "module-facts-discovery-0042-r2.json"
DISPOSITION = REPOSITORY_ROOT / "tests" / "parity" / "fixtures" / "module-disposition-0042-r2.json"

_DISPOSITIONS = frozenset({"keep", "move", "delete"})
_GATES = frozenset({"D-RUNTIME", "D-DOMAIN", "D-CUTOVER"})
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
_ENTRY_FIELDS = frozenset({"path", "disposition", "target_path", "owner", "completion_gate", "rule_id", "rationale"})
_REQUIRED_NON_ASSERTIONS = frozenset({"D0", "D-TECH", "installed_wheel_behavior", "legacy_zero"})

# Plan section 2.1: modules that must not exist after the cutover and carry no
# migrated file content of their own.
_PURE_DELETIONS = frozenset(
    {
        "fincore/_compat/__init__.py",
        "fincore/_dispatch.py",
        "fincore/_empyrical_legacy.py",
        "fincore/_pyfolio_impl.py",
        "fincore/_registry.py",
        "fincore/alphalens/__init__.py",
        "fincore/alphalens/performance.py",
        "fincore/alphalens/plotting.py",
        "fincore/alphalens/tears.py",
        "fincore/alphalens/utils.py",
        "fincore/api/adapters.py",
        "fincore/capabilities.py",
        "fincore/core/__init__.py",
        "fincore/contracts/__init__.py",
        "fincore/contracts/profiles.py",
        "fincore/contracts/workflows.py",
        "fincore/empyrical.py",
        "fincore/pyfolio.py",
        "fincore/results/__init__.py",
        "fincore/backends/__init__.py",
        "fincore/constants/__init__.py",
        "fincore/utils/__init__.py",
        "fincore/utils/deprecate.py",
    }
)

_MOVE_TARGET_ROOTS = (
    "fincore/runtime/",
    "fincore/metrics/",
    "fincore/performance/",
    "fincore/portfolio/",
    "fincore/factor_analysis/",
    "fincore/attribution/",
    "fincore/report/",
    "fincore/viz/",
    "fincore/extensions/",
    "fincore/data/",
)


def _load() -> tuple[dict, dict]:
    assert MODULE_FACTS.is_file(), "committed module facts discovery fixture is missing"
    assert DISPOSITION.is_file(), "committed module disposition fixture is missing"
    facts = json.loads(MODULE_FACTS.read_text(encoding="utf-8"))
    disposition = json.loads(DISPOSITION.read_text(encoding="utf-8"))
    return facts, disposition


def test_disposition_document_header_is_scoped_and_fail_closed() -> None:
    _, disposition = _load()

    assert disposition["schema_version"] == 1
    assert disposition["artifact_type"] == "module_disposition"
    assert disposition["scope"] == "fincore_source_tree_only"
    assert disposition["decision_status"] == "scoped"
    assert disposition["not_for_d0"] is True
    assert set(disposition["does_not_assert"]) >= _REQUIRED_NON_ASSERTIONS


def test_disposition_source_facts_bind_the_committed_module_facts_bytes() -> None:
    facts, disposition = _load()
    source_facts = disposition["source_facts"]

    assert source_facts["path"] == MODULE_FACTS.name
    assert source_facts["sha256"] == hashlib.sha256(MODULE_FACTS.read_bytes()).hexdigest()
    assert source_facts["module_count"] == len(facts["modules"])
    assert source_facts["source_provenance"] == facts["source_provenance"]


def test_disposition_maps_each_module_fact_exactly_once() -> None:
    facts, disposition = _load()
    fact_paths = [module["path"] for module in facts["modules"]]
    entries = disposition["entries"]

    assert [entry["path"] for entry in entries] == sorted(fact_paths)
    assert len(entries) == len(fact_paths)


def test_disposition_entries_follow_the_controlled_vocabulary() -> None:
    _, disposition = _load()

    for entry in disposition["entries"]:
        assert set(entry) == _ENTRY_FIELDS, entry["path"]
        assert entry["disposition"] in _DISPOSITIONS, entry["path"]
        assert entry["owner"] in _OWNERS, entry["path"]
        assert entry["completion_gate"] in _GATES, entry["path"]
        assert entry["rule_id"].strip(), entry["path"]
        assert entry["rationale"].strip(), entry["path"]
        if entry["disposition"] == "move":
            target = entry["target_path"]
            assert isinstance(target, str) and target, entry["path"]
            assert target != entry["path"], entry["path"]
            assert target.startswith(_MOVE_TARGET_ROOTS), entry["path"]
        elif entry["disposition"] == "keep":
            assert entry["target_path"] == entry["path"], entry["path"]
        else:
            assert entry["target_path"] is None, entry["path"]


def test_pure_legacy_deletions_are_marked_delete() -> None:
    _, disposition = _load()
    by_path = {entry["path"]: entry for entry in disposition["entries"]}

    for path in sorted(_PURE_DELETIONS):
        assert by_path[path]["disposition"] == "delete", path
        assert by_path[path]["target_path"] is None, path


def test_runtime_targets_only_bind_the_runtime_owner_before_d_runtime() -> None:
    _, disposition = _load()

    for entry in disposition["entries"]:
        if entry["disposition"] == "move" and entry["target_path"].startswith("fincore/runtime/"):
            assert entry["owner"] == "runtime", entry["path"]
            assert entry["completion_gate"] == "D-RUNTIME", entry["path"]


def test_every_entry_carries_a_non_empty_disposition_for_capture_input() -> None:
    _, disposition = _load()

    for index, entry in enumerate(disposition["entries"]):
        disposition_value = entry.get("disposition")
        assert isinstance(disposition_value, str) and disposition_value.strip(), f"entry {index}"
