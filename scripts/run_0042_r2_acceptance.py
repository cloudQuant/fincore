#!/usr/bin/env python3
"""Independent 0042-R2 acceptance runner (frozen-tooling slice).

This runner is the only authority allowed to sign tranche/final gate
conclusions.  It never imports candidate checkers, never reads expected
values from the candidate tree, and fails closed whenever identity or
evidence inputs are missing.

Exit codes:
    0  gate verdict PASS
    1  gate verdict FAIL
    2  usage or identity error
    3  gate verdict BLOCKED (missing or incomplete evidence)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_BLOCKED = 3

_VERDICT_EXIT = {"PASS": EXIT_PASS, "FAIL": EXIT_FAIL, "BLOCKED": EXIT_BLOCKED}
_GATE_MANIFEST_RELATIVE = Path("tests") / "parity" / "fixtures" / "0042-r2-gate-manifest.json"


class RunnerUsageError(ValueError):
    """Raised when the runner cannot establish its own execution identity."""


class RunnerBlockedError(ValueError):
    """Raised when a gate cannot reach a verdict from fail-closed inputs."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _tooling_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _runner_identity() -> dict[str, str]:
    script = Path(__file__).resolve()
    return {
        "runner_path": str(script),
        "runner_blob_sha256": _sha256_file(script),
    }


def _load_gate_manifest() -> dict:
    manifest_path = _tooling_root() / _GATE_MANIFEST_RELATIVE
    if not manifest_path.is_file():
        raise RunnerUsageError(f"gate manifest is missing from the tooling root: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunnerUsageError(f"gate manifest is not valid JSON: {exc}") from exc
    if manifest.get("artifact_type") != "gate_manifest" or manifest.get("schema_version") != 1:
        raise RunnerUsageError("gate manifest does not declare the frozen 0042-R2 gate contract")
    gates = manifest.get("gates")
    if not isinstance(gates, dict) or not gates:
        raise RunnerUsageError("gate manifest declares no gates")
    return manifest


def _git(candidate_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=candidate_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RunnerBlockedError(f"git {' '.join(arguments)} failed: {detail or 'unknown error'}")
    return result.stdout.strip()


def _require_existing_path(value: str | None, label: str) -> Path:
    if not value:
        raise RunnerUsageError(f"{label} was not supplied")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise RunnerUsageError(f"{label} does not exist: {path}")
    return path


def _require_bundle_outside_candidate(bundle: Path, candidate_root: Path | None) -> None:
    if candidate_root is None:
        return
    try:
        bundle.relative_to(candidate_root)
    except ValueError:
        return
    raise RunnerUsageError(
        "the expected D0 bundle must live outside the candidate tree; candidates provide actuals only"
    )


def _write_evidence(output_dir: Path | None, evidence: dict) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "evidence.json"
    target.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _run_evidence_child(args: argparse.Namespace, manifest: dict) -> dict:
    policy = manifest.get("evidence_child", {})
    allow_paths = list(args.allow_path or [])
    manifest_allow = policy.get("allow_paths", [])
    if not allow_paths:
        allow_paths = list(manifest_allow)
    if not args.tested_parent or not args.evidence_head:
        raise RunnerUsageError("evidence-child requires --tested-parent and --evidence-head")
    if not allow_paths:
        raise RunnerUsageError("evidence-child requires at least one --allow-path or manifest allow list")
    candidate_root = _require_existing_path(args.candidate_root, "--candidate-root")

    tested_parent = _git(candidate_root, "rev-parse", "--verify", f"{args.tested_parent}^{{commit}}")
    evidence_head = _git(candidate_root, "rev-parse", "--verify", f"{args.evidence_head}^{{commit}}")
    if tested_parent == evidence_head:
        raise RunnerBlockedError("evidence head must be a child of the tested parent, not the same commit")

    parents = _git(candidate_root, "rev-list", "--parents", "-n", "1", evidence_head).split()
    if len(parents) != 2:
        raise RunnerBlockedError(f"evidence head must have exactly one parent; found {max(len(parents) - 1, 0)}")
    if parents[1] != tested_parent:
        raise RunnerBlockedError("evidence head parent is not the tested candidate commit")

    changed = _git(candidate_root, "diff", "--name-only", f"{tested_parent}..{evidence_head}").splitlines()
    allowed = set(allow_paths)
    violations = [path for path in changed if path not in allowed]
    if violations:
        raise RunnerBlockedError(f"evidence child changes paths outside the allowlist: {', '.join(sorted(violations))}")

    return {
        "gate": "evidence-child",
        "verdict": "PASS",
        "tested_parent": tested_parent,
        "evidence_head": evidence_head,
        "allow_paths": sorted(allowed),
        "changed_paths": sorted(changed),
    }


def _run_bundle_gate(args: argparse.Namespace, gate: str, manifest: dict) -> dict:
    gate_spec = manifest["gates"][gate]
    candidate_root = _require_existing_path(args.candidate_root, "--candidate-root")
    bundle = _require_existing_path(args.expected_bundle, "--expected-bundle")
    _require_bundle_outside_candidate(bundle, candidate_root)
    if not gate_spec.get("consumes_d0_bundle"):
        raise RunnerUsageError(f"gate {gate} does not consume a D0 bundle")
    raise RunnerBlockedError(
        f"gate {gate} requires the frozen 0042-R2 D0 bundle contents and the detached D0_TOOLING_SHA "
        "execution environment, which are not frozen yet in this development slice"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent 0042-R2 acceptance runner.")
    parser.add_argument("--gate", required=True, help="gate identifier from the frozen gate manifest")
    parser.add_argument("--candidate-root")
    parser.add_argument("--candidate-head")
    parser.add_argument("--candidate-wheel")
    parser.add_argument("--candidate-dist")
    parser.add_argument("--expected-bundle")
    parser.add_argument("--families", nargs="*", default=[])
    parser.add_argument("--output-dir")
    parser.add_argument("--tested-parent")
    parser.add_argument("--evidence-head")
    parser.add_argument("--allow-path", action="append", default=[])
    parser.add_argument("--evidence-dir")
    parser.add_argument("--matrix-evidence-dir")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--include-serial", action="store_true")
    parser.add_argument("--include-offline-integration", action="store_true")
    parser.add_argument("--benchmarks-covered-by")
    parser.add_argument("--require-source-wheel-equal", action="store_true")
    parser.add_argument("--require-legacy-zero", action="store_true")
    parser.add_argument("--require-no-cycles", action="store_true")
    parser.add_argument("--require-one-sdist", action="store_true")
    parser.add_argument("--require-sdist-source-equivalence", action="store_true")
    parser.add_argument("--require-fresh-coverage", action="store_true")
    parser.add_argument("--require-changed-lines", type=float)
    parser.add_argument("--require-critical-branches", type=float)
    parser.add_argument("--require-loc-reduction", type=float)
    parser.add_argument("--require-duplicate-reduction", type=float)
    parser.add_argument("--require-os", nargs="*", default=[])
    parser.add_argument("--require-support-window-from-bundle", action="store_true")
    parser.add_argument("--real-browser")
    parser.add_argument("--real-html", action="store_true")
    parser.add_argument("--real-pdf", action="store_true")
    parser.add_argument("--real-xlsx", action="store_true")
    parser.add_argument("--interactive-backends", nargs="*", default=[])
    parser.add_argument("--profiles", nargs="*", default=[])
    parser.add_argument("--data-providers", nargs="*", default=[])
    parser.add_argument("--dependency-lanes", nargs="*", default=[])
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    identity = _runner_identity()
    exit_override: int | None = None

    try:
        manifest = _load_gate_manifest()
        gate = args.gate
        if gate not in manifest["gates"]:
            raise RunnerUsageError(f"unknown gate {gate!r}; required gates: {', '.join(manifest['required_gates'])}")

        if gate == "evidence-child":
            details = _run_evidence_child(args, manifest)
        else:
            details = _run_bundle_gate(args, gate, manifest)
        verdict = details["verdict"]
        reasons: list[str] = []
    except RunnerUsageError as exc:
        verdict = "BLOCKED"
        details = {"gate": args.gate}
        reasons = [f"usage: {exc}"]
        exit_override = EXIT_USAGE
        print(f"error: {exc}", file=sys.stderr)
    except RunnerBlockedError as exc:
        verdict = "BLOCKED"
        details = {"gate": args.gate}
        reasons = [f"blocked: {exc}"]
        print(f"blocked: {exc}", file=sys.stderr)

    evidence = {
        "artifact_type": "run_0042_r2_acceptance_evidence",
        "candidate_head": args.candidate_head,
        "candidate_root": args.candidate_root,
        "details": details,
        "gate": args.gate,
        "recorded_at": _utc_now(),
        "reasons": reasons,
        "runner": identity,
        "schema_version": 1,
        "verdict": verdict,
    }
    try:
        _write_evidence(output_dir, evidence)
    except OSError as exc:
        print(f"error: cannot write evidence: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if exit_override is not None:
        return exit_override
    return _VERDICT_EXIT[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
