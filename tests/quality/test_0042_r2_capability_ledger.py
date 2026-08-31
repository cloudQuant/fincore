"""Fail-closed contracts for the scoped 0042-R2 capability ledger.

The scoped ledger covers the metrics and performance families only.  It binds
each capability to real collected source nodeids, planned wheel nodeids, and
one independent-authority scenario.  It is explicitly not D0 evidence: the
capture tool rejects it while it remains scoped.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
FIXTURES = REPOSITORY_ROOT / "tests" / "parity" / "fixtures"
LEDGER = FIXTURES / "capability-ledger-0042-r2.json"
INVENTORY = FIXTURES / "legacy-surface-inventory-0042-r2.json"
NODE_FACTS = FIXTURES / "test-node-facts-discovery-0042-r2.json"
UPSTREAM_MANIFEST = REPOSITORY_ROOT / "tests" / "compat" / "fixtures" / "empyrical-0.6.0-api.json"
ALPHALENS_MANIFEST = REPOSITORY_ROOT / "tests" / "compat" / "fixtures" / "alphalens-0.4.0-cloudquant-api.json"
CAPTURE_SCRIPT = REPOSITORY_ROOT / "scripts" / "capture_capability_baseline.py"

# Packaging inventory rows are distribution configuration, not analytical
# capabilities; they remain outside the ledger by design and are gated by the
# package/installed lanes instead.
_COVERED_OWNERS = (
    "metrics",
    "performance",
    "factor",
    "risk",
    "attribution",
    "simulation",
    "optimization",
    "portfolio",
    "report",
    "viz",
    "data",
    "extensions",
    "runtime",
)
_REQUIRED_NON_ASSERTIONS = frozenset({"D0", "D-TECH", "installed_wheel_behavior", "legacy_zero"})


def _load_capture_module():
    specification = importlib.util.spec_from_file_location("capture_capability_baseline_ledger_test", CAPTURE_SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    original = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = original
    return module


def _load() -> dict:
    assert LEDGER.is_file(), "committed capability ledger fixture is missing"
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def _inventory_targets() -> dict[str, str]:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    targets: dict[str, str] = {}
    for entry in inventory["entries"]:
        target = entry["target_operation_id"]
        if entry["owner"] in _COVERED_OWNERS and ".surface." not in target:
            targets[target] = entry["owner"]
    return targets


def test_ledger_header_is_scoped_and_fail_closed() -> None:
    ledger = _load()

    assert ledger["schema_version"] == 1
    assert ledger["artifact_type"] == "capability_ledger"
    assert ledger["scope"] == "all_capability_families_except_packaging"
    assert ledger["covered_families"] == sorted(_COVERED_OWNERS)
    assert ledger["decision_status"] == "scoped"
    assert ledger["not_for_d0"] is True
    assert set(ledger["does_not_assert"]) >= _REQUIRED_NON_ASSERTIONS


def test_ledger_passes_the_frozen_capture_schema() -> None:
    capture = _load_capture_module()
    ledger = _load()

    entries = capture.validate_ledger(ledger)

    assert len(entries) == len(ledger["entries"])
    assert entries


def test_ledger_source_nodeids_are_real_collected_nodes() -> None:
    ledger = _load()
    facts = json.loads(NODE_FACTS.read_text(encoding="utf-8"))
    collected = {node["nodeid"] for node in facts["nodes"]}

    for entry in ledger["entries"]:
        unknown = [nodeid for nodeid in entry["source_nodeids"] if nodeid not in collected]
        assert unknown == [], entry["capability_id"]


def test_ledger_covers_every_covered_inventory_target_exactly_once() -> None:
    ledger = _load()
    targets = _inventory_targets()

    ledger_ids = {entry["capability_id"] for entry in ledger["entries"]}
    gap_ids = {gap["capability_id"] for gap in ledger["coverage_gaps"]}

    assert ledger_ids.isdisjoint(gap_ids)
    assert ledger_ids | gap_ids == set(targets)
    for entry in ledger["entries"]:
        assert targets[entry["capability_id"]] == entry["owner"]


def test_ledger_source_contract_binds_the_committed_input_bytes() -> None:
    ledger = _load()
    contract = ledger["source_contract"]

    assert contract["inventory_sha256"] == hashlib.sha256(INVENTORY.read_bytes()).hexdigest()
    assert contract["node_facts_sha256"] == hashlib.sha256(NODE_FACTS.read_bytes()).hexdigest()
    assert contract["upstream_manifest_sha256"] == hashlib.sha256(UPSTREAM_MANIFEST.read_bytes()).hexdigest()
    assert contract["alphalens_manifest_sha256"] == hashlib.sha256(ALPHALENS_MANIFEST.read_bytes()).hexdigest()


def test_upstream_capabilities_use_the_pinned_empyrical_oracle() -> None:
    ledger = _load()
    manifest = json.loads(UPSTREAM_MANIFEST.read_text(encoding="utf-8"))
    pinned_commit = manifest["commit"]
    upstream_symbols = {item["symbol"] for item in manifest["callables"]}

    for entry in ledger["entries"]:
        scenario = entry["scenarios"][0]
        authority = scenario["authority"]
        if authority["kind"] == "pinned_upstream_oracle" and authority["source_project"] == "empyrical":
            reference_symbol = authority["reference"].split(".", 1)[1]
            assert reference_symbol in upstream_symbols, entry["capability_id"]
            assert authority["artifact_digest"] == f"git-commit:{pinned_commit}", entry["capability_id"]
            assert authority["version"] == "0.6.0", entry["capability_id"]


def test_factor_capabilities_use_the_pinned_alphalens_oracle() -> None:
    ledger = _load()
    manifest = json.loads(ALPHALENS_MANIFEST.read_text(encoding="utf-8"))
    pinned_commit = manifest["identity"]["value"]
    upstream_symbols = {item["symbol"] for item in manifest["entries"] if item.get("kind") in {"function", "class"}}

    alphalens_references = 0
    for entry in ledger["entries"]:
        if entry["owner"] != "factor":
            continue
        scenario = entry["scenarios"][0]
        authority = scenario["authority"]
        if authority["kind"] == "pinned_upstream_oracle" and authority["source_project"] == "alphalens":
            reference_symbol = authority["reference"].rsplit(".", 1)[1]
            assert reference_symbol in upstream_symbols, entry["capability_id"]
            assert authority["artifact_digest"] == f"git-commit:{pinned_commit}", entry["capability_id"]
            assert authority["version"] == "0.4.0", entry["capability_id"]
            alphalens_references += 1

    assert alphalens_references > 0, "factor tranche must bind alphalens-derived capabilities to the pinned oracle"


def test_coverage_gaps_declare_only_missing_source_evidence() -> None:
    ledger = _load()

    assert ledger["coverage_gaps"], "a scoped ledger must declare its remaining coverage gaps"
    for gap in ledger["coverage_gaps"]:
        assert set(gap) == {"capability_id", "reason"}
        assert gap["reason"] == "no_source_nodeid"


def test_metrics_drawdown_and_leverage_coverage_is_backed_by_real_scenarios() -> None:
    """Prevent these implemented metrics from regressing to unsupported gaps.

    The three functions have stable, domain-native oracle cases.  They must
    therefore be represented by executable source and installed-wheel
    scenarios instead of indefinitely remaining in the generic coverage-gap
    queue.
    """
    ledger = _load()
    expected = {
        "metrics.gross_leverage": {
            "tests/parity/test_metrics.py::test_gross_leverage",
        },
        "metrics.second_max_drawdown": {
            "tests/parity/test_metrics.py::test_second_max_drawdown",
        },
        "metrics.third_max_drawdown": {
            "tests/parity/test_metrics.py::test_third_max_drawdown",
        },
    }

    gap_ids = {gap["capability_id"] for gap in ledger["coverage_gaps"]}
    entries = {entry["capability_id"]: entry for entry in ledger["entries"]}

    assert not expected.keys() & gap_ids
    for capability_id, wheel_nodeids in expected.items():
        entry = entries[capability_id]
        assert entry["owner"] == "metrics"
        assert entry["disposition"] == "required"
        assert entry["source_nodeids"]
        assert set(entry["wheel_nodeids"]) == wheel_nodeids


def test_optimization_and_simulation_gaps_are_backed_by_canonical_scenarios() -> None:
    """Keep direct domain behaviors out of the generic coverage-gap queue."""
    ledger = _load()
    expected = {
        "optimization.optimization_error": {
            "tests/parity/test_optimization.py::test_optimization_error",
        },
        "optimization.optimize": {
            "tests/parity/test_optimization.py::test_optimize",
        },
        "optimization.risk_parity": {
            "tests/parity/test_optimization.py::test_risk_parity",
        },
        "simulation.monte_carlo": {
            "tests/parity/test_simulation.py::test_monte_carlo",
        },
    }

    gap_ids = {gap["capability_id"] for gap in ledger["coverage_gaps"]}
    entries = {entry["capability_id"]: entry for entry in ledger["entries"]}

    assert not expected.keys() & gap_ids
    for capability_id, wheel_nodeids in expected.items():
        entry = entries[capability_id]
        assert entry["disposition"] == "required"
        assert entry["source_nodeids"]
        assert set(entry["wheel_nodeids"]) == wheel_nodeids
