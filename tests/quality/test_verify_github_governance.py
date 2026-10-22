"""Offline governance comparator contract tests (iteration 0041 / M2).

The comparator ``scripts/ci/verify_github_governance.py`` must be a
network-free verifier: it compares the repository's *expected* Ruleset
manifests against a saved (sanitized) GitHub API snapshot and emits a
machine-readable PASS/DRIFT report.

Fixtures live in ``tests/quality/fixtures/github_governance/``:
- ``empty.json``    -- nothing configured on GitHub.
- ``matching.json`` -- exactly matches both branch manifests (checks pending).
- ``drift.json``    -- seeds six independent drift classes.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARATOR_PATH = REPO_ROOT / "scripts" / "ci" / "verify_github_governance.py"
FIXTURES = REPO_ROOT / "tests" / "quality" / "fixtures" / "github_governance"
EXPECTED_DIR = REPO_ROOT / ".github" / "governance" / "rulesets"
LABELS_FILE = REPO_ROOT / ".github" / "governance" / "labels.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_github_governance", COMPARATOR_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise RuntimeError(f"cannot import {COMPARATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_manifests() -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(EXPECTED_DIR.glob("*.json"))]


# ---------------------------------------------------------------------------
# empty live state: nothing configured must produce actionable DRIFT output
# ---------------------------------------------------------------------------


def test_empty_live_state_reports_missing_policies_and_labels() -> None:
    mod = _load_module()
    report = mod.compare(_expected_manifests(), _load_json(FIXTURES / "empty.json"))

    assert report["verdict"] == "DRIFT"
    for branch in ("master", "dev"):
        entry = report["branches"][branch]
        assert entry["verdict"] == "DRIFT"
        kinds = {d["kind"] for d in entry["drifts"]}
        assert "missing_branch_policy" in kinds, f"{branch} must flag absent rules"
    # every governance label absent
    expected_label_names = {lbl["name"] for lbl in _load_json(LABELS_FILE)["labels"]}
    assert set(report["labels"]["missing"]) == expected_label_names


def test_empty_live_state_exit_code_is_one() -> None:
    mod = _load_module()
    out_file = FIXTURES.parent / "_tmp_out_empty.json"
    try:
        rc = mod.main(
            [
                "--expected",
                str(EXPECTED_DIR),
                "--labels",
                str(LABELS_FILE),
                "--actual",
                str(FIXTURES / "empty.json"),
                "--output",
                str(out_file),
            ]
        )
        assert rc == 1
        saved = _load_json(out_file)
        assert saved["verdict"] == "DRIFT"
    finally:
        out_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# matching live state: PASS while required checks remain pending-evidence
# ---------------------------------------------------------------------------


def test_matching_live_state_passes_with_pending_checks() -> None:
    mod = _load_module()
    report = mod.compare(_expected_manifests(), _load_json(FIXTURES / "matching.json"))

    assert report["verdict"] == "PASS"
    assert not report["codeowners_errors"]
    assert report["labels"]["missing"] == []
    for branch in ("master", "dev"):
        entry = report["branches"][branch]
        assert entry["verdict"] == "PASS", entry
        assert entry["drifts"] == []
        pending_names = {p["name"] for p in entry["pendings"]}
        assert pending_names == {"pr_governance_policy", "ci_build"}


def test_matching_live_state_exit_code_is_zero() -> None:
    mod = _load_module()
    out_file = FIXTURES.parent / "_tmp_out_matching.json"
    try:
        rc = mod.main(
            [
                "--expected",
                str(EXPECTED_DIR),
                "--labels",
                str(LABELS_FILE),
                "--actual",
                str(FIXTURES / "matching.json"),
                "--output",
                str(out_file),
            ]
        )
        assert rc == 0
    finally:
        out_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# drift fixture: each seeded problem must be reported precisely
# ---------------------------------------------------------------------------


def test_drift_fixture_lists_each_seeded_problem() -> None:
    mod = _load_module()
    report = mod.compare(_expected_manifests(), _load_json(FIXTURES / "drift.json"))

    assert report["verdict"] == "DRIFT"

    master = report["branches"]["master"]
    master_kinds = {d["kind"] for d in master["drifts"]}
    assert "force_push_allowed" in master_kinds
    assert "unexpected_required_check" in master_kinds

    # dev policy was attached to a wrongly named branch ("develop")
    dev = report["branches"]["dev"]
    dev_kinds = {d["kind"] for d in dev["drifts"]}
    assert "missing_branch_policy" in dev_kinds

    assert report["codeowners_errors"], "codeowners errors must surface"

    assert "risk:r3" in report["labels"]["missing"]

    bypass_drifts = [
        d
        for branch_entry in report["branches"].values()
        for d in branch_entry["drifts"]
        if d["kind"] == "unknown_bypass_actor"
    ]
    assert bypass_drifts, "undeclared bypass actor must be reported"


# ---------------------------------------------------------------------------
# manifest mutations compared against a healthy live state
# ---------------------------------------------------------------------------


def _write_mutated_expected(tmp_path: Path, mutate) -> Path:
    target = tmp_path / "rulesets"
    target.mkdir()
    for p in sorted(EXPECTED_DIR.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        mutate(data)
        (target / p.name).write_text(json.dumps(data), encoding="utf-8")
    return target


def test_wrong_required_check_context_is_drift(tmp_path: Path) -> None:
    mod = _load_module()

    def make_concrete(data):
        if data["branch"] == "master":
            for chk in data["required_checks"]:
                if chk["name"] == "ci_build":
                    chk["context"] = "CI / build"
                    chk["app_id"] = 999999  # synthetic value for mismatch detection

    mutated = _write_mutated_expected(tmp_path, make_concrete)

    # live state has NO required_status_checks at all -> concrete expectation unmet
    report = mod.compare(
        [json.loads(p.read_text()) for p in sorted(mutated.glob("*.json"))],
        _load_json(FIXTURES / "matching.json"),
    )
    master = report["branches"]["master"]
    assert master["verdict"] == "DRIFT"
    assert any(d["kind"] == "missing_required_check" for d in master["drifts"])

    # live state has a check but with a WRONG context/app id -> both sides flagged
    report2 = mod.compare(
        [json.loads(p.read_text()) for p in sorted(mutated.glob("*.json"))],
        _load_json(FIXTURES / "drift.json"),
    )
    master2 = report2["branches"]["master"]
    kinds2 = {d["kind"] for d in master2["drifts"]}
    assert {"missing_required_check", "extra_required_check"} <= kinds2


def test_d3_marker_missing_is_drift(tmp_path: Path) -> None:
    mod = _load_module()

    def drop_d3(data):
        data.pop("d3_exception", None)
        # second_reviewer_pending stays True -> inconsistency

    mutated = _write_mutated_expected(tmp_path, drop_d3)
    report = mod.compare(
        [json.loads(p.read_text()) for p in sorted(mutated.glob("*.json"))],
        _load_json(FIXTURES / "matching.json"),
    )
    drifted = [b for b, entry in report["branches"].items() if entry["verdict"] == "DRIFT"]
    assert drifted, "removing d3_exception while second reviewer pending must drift"
    for branch in ("master", "dev"):
        kinds = {d["kind"] for d in report["branches"][branch]["drifts"]}
        assert "d3_marker_missing" in kinds, f"{branch}: {kinds}"


def test_second_reviewer_claim_without_reality_is_drift(tmp_path: Path) -> None:
    mod = _load_module()

    def claim_two(data):
        data["review_policy"]["min_approvals"] = 2
        data["second_reviewer_pending"] = False

    mutated = _write_mutated_expected(tmp_path, claim_two)
    report = mod.compare(
        [json.loads(p.read_text()) for p in sorted(mutated.glob("*.json"))],
        _load_json(FIXTURES / "matching.json"),
    )
    for branch in ("master", "dev"):
        kinds = {d["kind"] for d in report["branches"][branch]["drifts"]}
        assert "approval_count_below_expectation" in kinds, f"{branch}: {kinds}"


# ---------------------------------------------------------------------------
# CLI misuse
# ---------------------------------------------------------------------------


def test_cli_bad_input_exits_two() -> None:
    mod = _load_module()
    rc = mod.main(
        [
            "--expected",
            str(EXPECTED_DIR),
            "--labels",
            str(LABELS_FILE),
            "--actual",
            str(FIXTURES / "does-not-exist.json"),
        ]
    )
    assert rc == 2
