#!/usr/bin/env python3
"""Development-time 0042-R2 feature parity checker.

Compares one D0 capability baseline capture against the capability ledger for
the requested families and reports missing capabilities or unresolved
observable differences.  This script is a candidate-checkout diagnostic: any
formal tranche, D-DOMAIN, D-CUTOVER, or D-TECH conclusion must instead be
signed by the detached ``D0_TOOLING_SHA`` acceptance runner.

Exit codes:
    0  parity verdict PASS for every requested family
    1  parity verdict FAIL (missing capability or unresolved difference)
    2  usage or identity error
    3  parity verdict BLOCKED (missing, pending, or incomplete evidence)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_BLOCKED = 3

_VERDICT_EXIT = {"PASS": EXIT_PASS, "FAIL": EXIT_FAIL, "BLOCKED": EXIT_BLOCKED}
_KNOWN_FAMILIES = frozenset(
    {
        "runtime",
        "metrics",
        "performance",
        "portfolio",
        "factor",
        "risk",
        "attribution",
        "simulation",
        "optimization",
        "report",
        "viz",
        "data",
        "extensions",
    }
)
_OK_CAPABILITY_STATUSES = frozenset({"ok"})
_PENDING_CAPABILITY_STATUSES = frozenset({"pending"})


class ParityUsageError(ValueError):
    """Raised when the checker cannot establish its input identity."""


class ParityBlockedError(ValueError):
    """Raised when parity cannot be decided from fail-closed inputs."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParityUsageError(f"cannot read {label} JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ParityUsageError(f"{label} must be a JSON object: {path}")
    return value


def _require_existing_path(value: str | None, label: str) -> Path:
    if not value:
        raise ParityUsageError(f"{label} was not supplied")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise ParityUsageError(f"{label} does not exist: {path}")
    return path


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_baseline(baseline: dict, baseline_path: Path, ledger_sha256: str) -> None:
    if baseline.get("artifact_type") != "capability_baseline_capture":
        raise ParityUsageError(f"baseline must be a capability_baseline_capture artifact: {baseline_path}")
    status = baseline.get("capture_status")
    if status == "pending":
        raise ParityBlockedError(f"baseline capture is pending: {baseline_path}")
    if status != "captured":
        raise ParityBlockedError(f"baseline capture_status must be captured, got {status!r}")
    capabilities = baseline.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ParityBlockedError("baseline does not carry a capability parity section")
    if baseline.get("evaluation_status") != "evaluated_source":
        raise ParityBlockedError("baseline capability section was not produced by the frozen source scenario evaluator")
    evaluation = baseline.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("artifact_type") != "capability_scenario_evaluation":
        raise ParityBlockedError("baseline lacks capability-scenario evaluation identity")
    if evaluation.get("mode") != "baseline_source":
        raise ParityBlockedError("baseline capability evaluation is not a frozen source evaluation")
    if not _is_sha256(evaluation.get("input_capture_sha256")):
        raise ParityBlockedError("baseline evaluation lacks an input-capture SHA256")
    ledger = evaluation.get("ledger")
    if not isinstance(ledger, dict) or ledger.get("sha256") != ledger_sha256:
        raise ParityBlockedError("baseline evaluation ledger SHA256 does not match the supplied ledger")
    source = baseline.get("source")
    if not isinstance(source, dict) or not all(source.get(key) for key in ("commit", "tree")):
        raise ParityBlockedError("baseline lacks captured source identity")
    executions = baseline.get("executions")
    if not isinstance(executions, dict) or not executions:
        raise ParityBlockedError("baseline evaluation lacks source execution evidence")


def _ledger_family_scope(ledger: dict) -> frozenset[str]:
    scope = ledger.get("scope")
    if not isinstance(scope, str) or not scope.strip():
        raise ParityUsageError("ledger must declare a scope")
    if scope == "complete":
        return frozenset(_KNOWN_FAMILIES)
    owners = {entry.get("owner") for entry in ledger.get("entries", [])}
    return frozenset(owner for owner in owners if owner in _KNOWN_FAMILIES)


def _select_ledger_capabilities(ledger: dict, families: frozenset[str]) -> list[dict]:
    return [
        entry
        for entry in ledger.get("entries", [])
        if entry.get("owner") in families and entry.get("disposition") == "required"
    ]


def _check_ledger_completeness(ledger: dict, families: frozenset[str], families_arg: list[str]) -> None:
    scope = _ledger_family_scope(ledger)
    missing_scope = sorted(families - scope)
    if missing_scope:
        raise ParityBlockedError(f"ledger scope does not cover requested families: {', '.join(missing_scope)}")
    if ledger.get("decision_status") == "scoped" or ledger.get("not_for_d0") is True:
        gap_families = set()
        for gap in ledger.get("coverage_gaps", []):
            capability_id = gap.get("capability_id", "")
            owner = capability_id.split(".", 1)[0]
            if owner in families:
                gap_families.add(owner)
        if gap_families:
            raise ParityBlockedError(
                "ledger declares unresolved coverage gaps for requested families: "
                f"{', '.join(sorted(gap_families))} (requested: {', '.join(families_arg)})"
            )
        if ledger.get("decision_status") == "scoped":
            raise ParityBlockedError("ledger decision_status scoped cannot sign parity until it is complete")


def _valid_ok_record(record: dict, baseline: dict) -> bool:
    evaluation = baseline["evaluation"]
    if record.get("input_capture_sha256") != evaluation.get("input_capture_sha256"):
        return False
    ledger = evaluation.get("ledger")
    if not isinstance(ledger, dict) or record.get("ledger_sha256") != ledger.get("sha256"):
        return False
    if record.get("source_identity") != baseline.get("source"):
        return False
    if not isinstance(record.get("wheel_nodeids"), list) or not record["wheel_nodeids"]:
        return False
    scenarios = record.get("scenarios")
    executions = baseline.get("executions")
    if not isinstance(scenarios, list) or not scenarios or not isinstance(executions, dict):
        return False
    for scenario in scenarios:
        if not isinstance(scenario, dict) or scenario.get("status") != "ok":
            return False
        if not _is_sha256(scenario.get("authority_sha256")) or not _is_sha256(scenario.get("output_sha256")):
            return False
        if not (_is_sha256(scenario.get("golden_sha256")) or _is_sha256(scenario.get("oracle_sha256"))):
            return False
        execution_ids = scenario.get("execution_ids")
        if not isinstance(execution_ids, list) or not execution_ids:
            return False
        for execution_id in execution_ids:
            execution = executions.get(execution_id)
            if not isinstance(execution, dict) or execution.get("exit_code") != 0:
                return False
            if not isinstance(execution.get("argv"), list) or not execution["argv"]:
                return False
            if not _is_sha256(execution.get("output_sha256")):
                return False
    return True


def _evaluate_family(family: str, entries: list[dict], baseline: dict) -> dict:
    capabilities = baseline["capabilities"]
    missing: list[str] = []
    divergent: list[str] = []
    pending: list[str] = []
    untrusted: list[str] = []
    verified: list[str] = []
    for entry in entries:
        capability_id = entry["capability_id"]
        record = capabilities.get(capability_id)
        if not isinstance(record, dict):
            missing.append(capability_id)
            continue
        status = record.get("status")
        if status in _OK_CAPABILITY_STATUSES:
            if _valid_ok_record(record, baseline):
                verified.append(capability_id)
            else:
                untrusted.append(capability_id)
        elif status in _PENDING_CAPABILITY_STATUSES:
            pending.append(capability_id)
        else:
            divergent.append(capability_id)
    verdict = "BLOCKED" if missing or pending or untrusted else "FAIL" if divergent else "PASS"
    return {
        "family": family,
        "verdict": verdict,
        "required_capabilities": len(entries),
        "verified_capabilities": len(verified),
        "missing_capabilities": sorted(missing),
        "pending_capabilities": sorted(pending),
        "unresolved_differences": sorted(divergent),
        "untrusted_capabilities": sorted(untrusted),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="0042-R2 development-time feature parity checker.")
    parser.add_argument("--baseline", required=True, help="D0 capability baseline capture JSON")
    parser.add_argument("--ledger", required=True, help="capability ledger JSON")
    parser.add_argument("--families", nargs="+", required=True, help="family identifiers or 'all'")
    parser.add_argument("--dist", help="optional wheel dist directory for wheel-side evidence")
    parser.add_argument("--output", help="optional evidence output path")
    args = parser.parse_args(argv)

    if args.families == ["all"]:
        families = frozenset(_KNOWN_FAMILIES)
    else:
        unknown = sorted(set(args.families) - _KNOWN_FAMILIES)
        if unknown:
            print(
                f"error: unknown families {', '.join(unknown)}; known: {', '.join(sorted(_KNOWN_FAMILIES))}",
                file=sys.stderr,
            )
            return EXIT_USAGE
        families = frozenset(args.families)

    verdict = "BLOCKED"
    reasons: list[str] = []
    family_results: list[dict] = []
    exit_override: int | None = None

    try:
        baseline_path = _require_existing_path(args.baseline, "--baseline")
        ledger_path = _require_existing_path(args.ledger, "--ledger")
        if args.dist:
            dist_path = Path(args.dist).expanduser().resolve()
            if not dist_path.is_dir():
                raise ParityUsageError(f"--dist directory does not exist: {dist_path}")
            wheels = sorted(dist_path.glob("*.whl"))
            if not wheels:
                raise ParityBlockedError(f"--dist contains no wheel artifact: {dist_path}")
        ledger = _load_json(ledger_path, "ledger")
        if ledger.get("artifact_type") != "capability_ledger":
            raise ParityUsageError(f"ledger must be a capability_ledger artifact: {ledger_path}")
        _check_ledger_completeness(ledger, families, sorted(families))
        baseline = _load_json(baseline_path, "baseline")
        _validate_baseline(baseline, baseline_path, _sha256_file(ledger_path))

        for family in sorted(families):
            entries = _select_ledger_capabilities(ledger, frozenset({family}))
            if not entries:
                raise ParityBlockedError(f"ledger declares no capabilities for family {family!r}")
            family_results.append(_evaluate_family(family, entries, baseline))

        if any(result["verdict"] == "BLOCKED" for result in family_results):
            verdict = "BLOCKED"
        elif any(result["verdict"] == "FAIL" for result in family_results):
            verdict = "FAIL"
        else:
            verdict = "PASS"
    except ParityUsageError as exc:
        exit_override = EXIT_USAGE
        reasons.append(f"usage: {exc}")
        print(f"error: {exc}", file=sys.stderr)
    except ParityBlockedError as exc:
        reasons.append(f"blocked: {exc}")
        print(f"blocked: {exc}", file=sys.stderr)

    evidence = {
        "artifact_type": "check_feature_parity_evidence",
        "baseline": str(Path(args.baseline).expanduser().resolve()) if args.baseline else None,
        "baseline_sha256": _sha256_file(Path(args.baseline).expanduser().resolve())
        if args.baseline and Path(args.baseline).expanduser().resolve().exists()
        else None,
        "families": sorted(families),
        "ledger": str(Path(args.ledger).expanduser().resolve()) if args.ledger else None,
        "reasons": reasons,
        "results": family_results,
        "schema_version": 1,
        "scope": "development_diagnostic_only",
        "verdict": verdict,
    }
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot write evidence: {exc}", file=sys.stderr)
            return EXIT_USAGE
    if exit_override is not None:
        return exit_override
    return _VERDICT_EXIT[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
