#!/usr/bin/env python
"""Offline GitHub governance comparator for fincore (iteration 0041 / M2).

Compares the repository's *expected* Ruleset manifests
(``.github/governance/rulesets/*.json``) against a saved, sanitized snapshot of
GitHub API responses (Rulesets, simplified branch rules, CODEOWNERS errors,
labels). The tool performs **no network access** and reads **no tokens**;
raw API payloads stay outside the repository.

Verdict semantics
-----------------
- ``PASS``  -- every expected field matches; unmet ``pending-evidence`` items
  are listed separately and do not fail the comparison by themselves.
- ``DRIFT`` -- at least one concrete mismatch (missing/extra/wrong value,
  undeclared bypass actor, label drift, CODEOWNERS errors).

Exit codes: ``0`` PASS, ``1`` DRIFT, ``2`` usage/input error.

Live-state document shape (see tests/quality/fixtures/github_governance/)::

    {
      "captured_at_utc": "...",
      "repo": "cloudQuant/fincore",
      "default_branch": "master",
      "branch_rules": {"<branch>": [{"type": "...", "parameters": {...}}, ...]},
      "rulesets": [{"conditions": {"ref_name": {"include": [...]}},
                    "bypass": [...], ...}],
      "codeowners_errors": [],
      "labels": [{"name": "...", "color": "...", "description": "..."}]
    }

Decision provenance: ``.github/governance/0041/decisions.md`` (D0-D4).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
GOVERNANCE_LABELS = REPO_ROOT / ".github" / "governance" / "labels.json"

PENDING = "pending-evidence"
EXIT_PASS = 0
EXIT_DRIFT = 1
EXIT_USAGE = 2


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_expected_manifests(path: Path) -> list[dict[str, Any]]:
    """Load one manifest file or every ``*.json`` inside a directory."""
    if path.is_dir():
        files = sorted(path.glob("*.json"))
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(f"expected-manifests path not found: {path}")
    manifests: list[dict[str, Any]] = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "branch" not in data:
            raise ValueError(f"{f} is not a branch manifest (missing 'branch')")
        if data.get("schema_version") != 1:
            raise ValueError(f"{f} has unsupported schema_version")
        manifests.append(data)
    if not manifests:
        raise ValueError(f"no manifests found under {path}")
    return manifests


def _load_json_file(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _rule_of(rules: list[dict[str, Any]], rule_type: str) -> dict[str, Any] | None:
    return next((r for r in rules if r.get("type") == rule_type), None)


def _observed_policy(rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize the simplified branch-rules view into canonical fields."""
    observed: dict[str, Any] = {
        "requires_pull_request": False,
        "min_approvals": 0,
        "require_code_owner": False,
        "require_last_push_approval": False,
        "dismiss_stale_reviews": False,
        "require_conversation_resolution": False,
        "allow_force_push": True,
        "allow_deletion": True,
        "required_checks": [],
    }
    pr_rule = _rule_of(rules, "pull_request")
    if pr_rule is not None:
        params = pr_rule.get("parameters") or {}
        observed["requires_pull_request"] = True
        observed["min_approvals"] = int(params.get("required_approving_review_count") or 0)
        observed["require_code_owner"] = bool(params.get("require_code_owner_review"))
        observed["require_last_push_approval"] = bool(params.get("require_last_push_approval"))
        observed["dismiss_stale_reviews"] = bool(params.get("dismiss_stale_reviews_on_new_push"))
        observed["require_conversation_resolution"] = bool(params.get("required_review_thread_resolution"))
    observed["allow_force_push"] = _rule_of(rules, "non_fast_forward") is None
    observed["allow_deletion"] = _rule_of(rules, "deletion") is None
    status_rule = _rule_of(rules, "required_status_checks")
    if status_rule is not None:
        params = status_rule.get("parameters") or []
        if isinstance(params, dict):
            params = [params]
        observed["required_checks"] = [
            {"context": str(p.get("context")), "app_id": p.get("integration_id")} for p in params
        ]
    return observed


def _bypass_actors_for_branch(live: dict[str, Any], branch: str) -> list[dict[str, Any]]:
    actors: list[dict[str, Any]] = []
    for ruleset in live.get("rulesets") or []:
        conditions = ruleset.get("conditions") or {}
        ref_name = conditions.get("ref_name") or {}
        includes = ref_name.get("include") or []
        targets_branch = f"refs/heads/{branch}" in includes or "~DEFAULT_BRANCH" in includes
        if not targets_branch:
            continue
        actors.extend(ruleset.get("bypass") or [])
    return actors


def _d3_marker_valid(d3: Any) -> bool:
    if not isinstance(d3, dict) or not d3.get("enabled"):
        return False
    label = d3.get("label")
    if not isinstance(label, str) or not label.strip():
        return False
    try:
        cooling_hours = int(d3.get("cooling_hours", 0))
        max_age_days = int(d3.get("max_age_days", 0))
    except (TypeError, ValueError):
        return False
    return cooling_hours > 0 and 1 <= max_age_days <= 30


def _check_required_checks(
    manifest: dict[str, Any], observed_checks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    drifts: list[dict[str, Any]] = []
    pendings: list[dict[str, Any]] = []
    expected = manifest.get("required_checks") or []
    expected_concrete = [c for c in expected if c.get("context") != PENDING]
    has_pending = len(expected_concrete) < len(expected)

    if expected_concrete:
        for chk in expected_concrete:
            matched = any(
                chk.get("context") == obs.get("context") and str(chk.get("app_id")) == str(obs.get("app_id"))
                for obs in observed_checks
            )
            if not matched:
                drifts.append(
                    {
                        "kind": "missing_required_check",
                        "field": "required_checks",
                        "name": chk.get("name"),
                        "expected": {
                            "context": chk.get("context"),
                            "app_id": str(chk.get("app_id")),
                        },
                        "actual": observed_checks,
                    }
                )
        for obs in observed_checks:
            matched = any(
                chk.get("context") == obs.get("context") and str(chk.get("app_id")) == str(obs.get("app_id"))
                for chk in expected_concrete
            )
            if not matched:
                drifts.append(
                    {
                        "kind": "extra_required_check",
                        "field": "required_checks",
                        "expected": [
                            {"context": c.get("context"), "app_id": str(c.get("app_id"))} for c in expected_concrete
                        ],
                        "actual": obs,
                    }
                )

    if has_pending and observed_checks:
        drifts.append(
            {
                "kind": "unexpected_required_check",
                "field": "required_checks",
                "expected": "no configured checks while expectations are pending-evidence",
                "actual": observed_checks,
                "detail": (
                    "D4: contexts/app ids are pending-evidence until M4 drill PRs "
                    "collect real check-run values; observed configuration cannot be verified"
                ),
            }
        )

    pendings.extend(
        {
            "name": chk.get("name"),
            "reason": "context/app_id remain pending-evidence until M4 drill evidence",
        }
        for chk in expected
        if chk.get("context") == PENDING
    )
    return drifts, pendings


def compare_branch(manifest: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    drifts: list[dict[str, Any]] = []
    pendings: list[dict[str, Any]] = []

    branch_rules = (live.get("branch_rules") or {}).get(manifest["branch"])
    observed: dict[str, Any] | None = None
    observed_checks: list[dict[str, Any]] = []
    if branch_rules:
        observed = _observed_policy(branch_rules)
        observed_checks = observed["required_checks"]
    else:
        drifts.append(
            {
                "kind": "missing_branch_policy",
                "field": f"branch_rules.{manifest['branch']}",
                "expected": "simplified branch rules present",
                "actual": None,
                "detail": ("no rules target this branch; it is unprotected (or attached to a wrongly named branch)"),
            }
        )

    review = manifest.get("review_policy") or {}
    if observed is not None:
        if manifest.get("requires_pull_request") and not observed["requires_pull_request"]:
            drifts.append(
                {
                    "kind": "pull_request_not_required",
                    "field": "requires_pull_request",
                    "expected": True,
                    "actual": False,
                }
            )
        if observed["min_approvals"] < int(review.get("min_approvals", 0)):
            drifts.append(
                {
                    "kind": "approval_count_below_expectation",
                    "field": "review_policy.min_approvals",
                    "expected_at_least": review.get("min_approvals"),
                    "actual": observed["min_approvals"],
                }
            )
        boolean_fields = (
            ("require_code_owner", "review_policy.require_code_owner", "require_code_owner"),
            (
                "require_last_push_approval",
                "review_policy.require_last_push_approval",
                "require_last_push_approval",
            ),
            ("dismiss_stale_reviews", "review_policy.dismiss_stale_reviews", "dismiss_stale_reviews"),
            (
                "require_conversation_resolution",
                "review_policy.require_conversation_resolution",
                "require_conversation_resolution",
            ),
        )
        for obs_key, exp_key, pretty in boolean_fields:
            if bool(review.get(exp_key.split(".", 1)[1])) and not observed[obs_key]:
                drifts.append(
                    {
                        "kind": f"{pretty}_not_enforced",
                        "field": exp_key,
                        "expected": True,
                        "actual": observed[obs_key],
                    }
                )
        if not manifest.get("allow_force_push", True) and observed["allow_force_push"]:
            drifts.append(
                {
                    "kind": "force_push_allowed",
                    "field": "allow_force_push",
                    "expected": False,
                    "actual": True,
                    "detail": "no non_fast_forward rule observed",
                }
            )
        if not manifest.get("allow_deletion", True) and observed["allow_deletion"]:
            drifts.append(
                {
                    "kind": "deletion_allowed",
                    "field": "allow_deletion",
                    "expected": False,
                    "actual": True,
                    "detail": "no deletion rule observed",
                }
            )

    check_drifts, check_pendings = _check_required_checks(manifest, observed_checks)
    drifts.extend(check_drifts)
    pendings.extend(check_pendings)

    if manifest.get("second_reviewer_pending") and not _d3_marker_valid(manifest.get("d3_exception")):
        drifts.append(
            {
                "kind": "d3_marker_missing",
                "field": "d3_exception",
                "expected": "valid single-maintainer exception marker",
                "actual": manifest.get("d3_exception"),
                "detail": ("second_reviewer_pending=true requires a well-formed D3 exception record (decisions.md D3)"),
            }
        )

    declared_bypass = manifest.get("bypass_actors") or []
    for actor in _bypass_actors_for_branch(live, manifest["branch"]):
        known = any(
            actor.get("slug") == d.get("slug") and actor.get("actor_type") == d.get("actor_type")
            for d in declared_bypass
        )
        if not known:
            drifts.append(
                {
                    "kind": "unknown_bypass_actor",
                    "field": "bypass_actors",
                    "expected": declared_bypass or "no bypass actors declared",
                    "actual": actor,
                }
            )

    verdict = "DRIFT" if drifts else "PASS"
    return {"branch": manifest["branch"], "verdict": verdict, "drifts": drifts, "pendings": pendings}


def compare_labels(labels_doc: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    actual = {lbl.get("name"): lbl for lbl in live.get("labels") or []}
    missing: list[str] = []
    mismatched: list[dict[str, Any]] = []
    expected_names: set[str] = set()
    for lbl in labels_doc.get("labels") or []:
        name = lbl.get("name")
        expected_names.add(name)
        if name not in actual:
            missing.append(name)
            continue
        act = actual[name]
        for field in ("description", "color"):
            expected_value = lbl.get(field)
            actual_value = act.get(field)
            if expected_value is not None and expected_value != actual_value:
                mismatched.append({"label": name, "field": field, "expected": expected_value, "actual": actual_value})
    extra_info = sorted(set(actual) - expected_names)
    return {"missing": missing, "mismatched": mismatched, "extra_info": info_note(extra_info)}


def info_note(extras: list[str]) -> list[str]:
    """Extra labels are informational only (GitHub defaults are legitimate)."""
    return extras


def compare(
    manifests: list[dict[str, Any]],
    live: dict[str, Any],
    labels_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if labels_doc is None:
        labels_doc = _load_json_file(GOVERNANCE_LABELS)

    branch_reports: dict[str, Any] = {}
    for manifest in manifests:
        entry = compare_branch(manifest, live)
        branch_reports[manifest["branch"]] = entry

    label_report = compare_labels(labels_doc, live)
    codeowners_errors = live.get("codeowners_errors") or []

    drifted = any(entry["verdict"] == "DRIFT" for entry in branch_reports.values())
    drifted = drifted or bool(label_report["missing"]) or bool(label_report["mismatched"])
    drifted = drifted or bool(codeowners_errors)

    return {
        "verdict": "DRIFT" if drifted else "PASS",
        "generated_at_utc": _now_utc(),
        "branches": branch_reports,
        "labels": label_report,
        "codeowners_errors": codeowners_errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected",
        required=True,
        help="directory (or single file) containing expected branch manifests",
    )
    parser.add_argument("--labels", required=True, help="governance labels.json path")
    parser.add_argument("--actual", required=True, help="saved sanitized GitHub API snapshot (live-state JSON)")
    parser.add_argument("--output", help="also write the JSON report to this path")
    args = parser.parse_args(argv)

    try:
        manifests = load_expected_manifests(Path(args.expected))
        labels_doc = _load_json_file(Path(args.labels))
        live = _load_json_file(Path(args.actual))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return EXIT_USAGE

    report = compare(manifests, live, labels_doc)
    rendered = json.dumps(report, indent=2, sort_keys=False)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")

    return EXIT_PASS if report["verdict"] == "PASS" else EXIT_DRIFT


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
