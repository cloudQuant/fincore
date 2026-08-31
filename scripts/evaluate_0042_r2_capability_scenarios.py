#!/usr/bin/env python3
"""Evaluate frozen 0042-R2 capability scenarios against a clean D0 source.

``capture_capability_baseline.py`` intentionally records only immutable D0
inputs.  This companion command consumes that input-only capture, executes
the ledger's *source* nodeids in the exact captured source checkout, and
materializes a separate evaluated capability-baseline artifact.  It is not a
candidate or release verdict: candidate source/wheel parity is evaluated by
the frozen acceptance runner after the breaking cutover.

The command is fail closed.  A capability is ``ok`` only when every selected
scenario has a structured authority, an immutable golden/oracle digest, and
successful source-test execution evidence.  Missing prerequisites are
``pending`` (exit 3); executed test failures are ``failed`` (exit 1).  This
distinction prevents a ledger row, a candidate assertion, or an unexecuted
wheel nodeid from being mistaken for D0 evidence.

The evaluator runs from a separate clean tooling worktree.  It reads the
ledger from the source commit named by the input capture rather than from the
mutable source working tree, records both source and tooling identities, and
writes only outside both worktrees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


SCHEMA_VERSION = 1
_ARTIFACT_TYPE = "capability_baseline_capture"
_EVALUATION_STATUS = "evaluated_source"
_SCRIPT_RELATIVE = Path("scripts") / "evaluate_0042_r2_capability_scenarios.py"
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
_SHA256_LENGTH = 64

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_BLOCKED = 3


class EvaluationUsageError(ValueError):
    """Raised when an input cannot establish immutable evaluation identity."""


class EvaluationBlockedError(ValueError):
    """Raised when a scenario lacks enough evidence for an honest verdict."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value.casefold())
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationUsageError(f"cannot read {label} JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationUsageError(f"{label} must be a JSON object: {path}")
    return value


def _git_text(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise EvaluationUsageError(f"git {' '.join(arguments)} failed: {detail or 'unknown error'}")
    return result.stdout.strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=False,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", "replace").strip()
        raise EvaluationUsageError(f"git {' '.join(arguments)} failed: {detail or 'unknown error'}")
    return result.stdout


def _worktree_identity(root: Path, label: str) -> dict[str, str | bool]:
    root = root.resolve()
    if not root.is_dir():
        raise EvaluationUsageError(f"{label} does not exist or is not a directory: {root}")
    reported_root = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
    if reported_root != root:
        raise EvaluationUsageError(f"{label} must be the Git worktree root: {root}")
    if _git_text(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise EvaluationUsageError(f"{label} must be a clean Git worktree: {root}")
    return {
        "commit": _git_text(root, "rev-parse", "--verify", "HEAD"),
        "tree": _git_text(root, "rev-parse", "--verify", "HEAD^{tree}"),
        "clean": True,
    }


def _portable_relative_path(value: object, label: str) -> str:
    if not _non_empty_string(value):
        raise EvaluationUsageError(f"{label} must be a non-empty portable relative path")
    assert isinstance(value, str)
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or path.as_posix() != value
    ):
        raise EvaluationUsageError(f"{label} must be a non-escaping POSIX relative path")
    return value


def _git_blob(root: Path, commit: str, relative_path: str, label: str) -> bytes:
    _portable_relative_path(relative_path, label)
    spec = f"{commit}:{relative_path}"
    try:
        return _git_bytes(root, "show", spec)
    except EvaluationUsageError as exc:
        raise EvaluationUsageError(
            f"{label} is not a readable blob in the captured source commit: {relative_path}"
        ) from exc


def _tooling_identity(tooling_root: Path) -> dict[str, Any]:
    tooling_root = tooling_root.resolve()
    expected_script = tooling_root / _SCRIPT_RELATIVE
    actual_script = Path(__file__).resolve()
    if actual_script != expected_script.resolve():
        raise EvaluationUsageError("evaluator must execute from the supplied frozen tooling root")
    identity = _worktree_identity(tooling_root, "tooling root")
    commit = identity["commit"]
    assert isinstance(commit, str)
    frozen_script = _git_blob(tooling_root, commit, _SCRIPT_RELATIVE.as_posix(), "frozen evaluator")
    actual_sha256 = _sha256_bytes(actual_script.read_bytes())
    frozen_sha256 = _sha256_bytes(frozen_script)
    if actual_sha256 != frozen_sha256:
        raise EvaluationUsageError("evaluator bytes do not match the frozen tooling Git blob")
    return {
        "root": str(tooling_root),
        "source": identity,
        "evaluator": {"path": _SCRIPT_RELATIVE.as_posix(), "sha256": frozen_sha256},
    }


def _require_external_output(output: Path, source_root: Path, tooling_root: Path) -> Path:
    output = output.expanduser().resolve()
    for root, label in ((source_root, "source"), (tooling_root, "tooling")):
        try:
            output.relative_to(root)
        except ValueError:
            continue
        raise EvaluationUsageError(f"output must be outside the {label} worktree")
    return output


def _require_capture_source(capture: dict[str, Any], source_identity: dict[str, str | bool]) -> None:
    if capture.get("artifact_type") != _ARTIFACT_TYPE:
        raise EvaluationUsageError("input capture must be a capability_baseline_capture artifact")
    if capture.get("capture_status") != "captured":
        raise EvaluationBlockedError("input capture_status must be captured")
    if "capabilities" in capture or capture.get("evaluation_status"):
        raise EvaluationUsageError("input capture must be the immutable input-only artifact, not an evaluated baseline")
    captured_source = capture.get("source")
    if not isinstance(captured_source, dict):
        raise EvaluationUsageError("input capture is missing source identity")
    if any(captured_source.get(key) != source_identity.get(key) for key in ("commit", "tree", "clean")):
        raise EvaluationUsageError("source worktree identity does not match the input capture")


def _load_frozen_ledger(
    capture: dict[str, Any], source_root: Path, source_identity: dict[str, str | bool]
) -> tuple[dict[str, Any], dict[str, str]]:
    inputs = capture.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get("ledger"), dict):
        raise EvaluationUsageError("input capture lacks immutable ledger input metadata")
    ledger_input = inputs["ledger"]
    ledger_path = _portable_relative_path(ledger_input.get("path"), "input capture ledger path")
    expected_sha256 = ledger_input.get("sha256")
    if not _is_sha256(expected_sha256):
        raise EvaluationUsageError("input capture ledger SHA256 is invalid")
    commit = source_identity["commit"]
    assert isinstance(commit, str)
    payload = _git_blob(source_root, commit, ledger_path, "captured ledger")
    actual_sha256 = _sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise EvaluationUsageError("captured ledger blob SHA256 does not match input capture")
    try:
        ledger = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationUsageError(f"captured ledger is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(ledger, dict) or ledger.get("artifact_type") != "capability_ledger":
        raise EvaluationUsageError("captured ledger must be a capability_ledger artifact")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or not entries:
        raise EvaluationUsageError("captured ledger must declare non-empty entries")
    return ledger, {"path": ledger_path, "sha256": actual_sha256}


def _verify_fixture_manifest(
    capture: dict[str, Any], source_root: Path, source_identity: dict[str, str | bool]
) -> dict[str, Any]:
    fixture_root = _portable_relative_path(capture.get("fixture_root"), "input capture fixture_root")
    fixtures = capture.get("fixtures")
    if not isinstance(fixtures, dict) or not fixtures:
        raise EvaluationUsageError("input capture lacks a non-empty fixture manifest")
    commit = source_identity["commit"]
    assert isinstance(commit, str)
    for fixture_path, fixture in fixtures.items():
        fixture_path = _portable_relative_path(fixture_path, "input capture fixture path")
        if not isinstance(fixture, dict) or not _is_sha256(fixture.get("sha256")):
            raise EvaluationUsageError(f"input capture fixture digest is invalid: {fixture_path}")
        payload = _git_blob(source_root, commit, f"{fixture_root}/{fixture_path}", "captured fixture")
        if _sha256_bytes(payload) != fixture["sha256"]:
            raise EvaluationUsageError(f"captured fixture blob SHA256 does not match input capture: {fixture_path}")
    return fixtures


def _selected_families(values: list[str]) -> frozenset[str]:
    if values == ["all"]:
        return _KNOWN_FAMILIES
    unknown = sorted(set(values) - _KNOWN_FAMILIES)
    if unknown:
        raise EvaluationUsageError(
            f"unknown families {', '.join(unknown)}; known: {', '.join(sorted(_KNOWN_FAMILIES))}"
        )
    return frozenset(values)


def _entry_scenarios(entry: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = entry.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios or not all(isinstance(item, dict) for item in scenarios):
        return []
    return scenarios


def _entry_source_nodeids(entry: dict[str, Any]) -> list[str]:
    nodeids = entry.get("source_nodeids")
    if not isinstance(nodeids, list) or not nodeids or not all(_non_empty_string(value) for value in nodeids):
        return []
    return [str(value).strip() for value in nodeids]


def _entry_wheel_nodeids(entry: dict[str, Any]) -> list[str]:
    nodeids = entry.get("wheel_nodeids")
    if not isinstance(nodeids, list) or not nodeids or not all(_non_empty_string(value) for value in nodeids):
        return []
    return [str(value).strip() for value in nodeids]


def _scenario_prerequisites(
    scenario: dict[str, Any], fixtures: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    scenario_id = scenario.get("scenario_id")
    if not _non_empty_string(scenario_id):
        return None, "scenario_id is missing"
    authority = scenario.get("authority")
    if (
        not isinstance(authority, dict)
        or not _non_empty_string(authority.get("kind"))
        or not _non_empty_string(authority.get("reference"))
    ):
        return None, "structured authority is missing"
    golden_path = scenario.get("golden_path")
    oracle_reference = scenario.get("oracle_reference")
    if _non_empty_string(golden_path):
        try:
            fixture_path = _portable_relative_path(golden_path, f"scenario {scenario_id!r} golden_path")
        except EvaluationUsageError as exc:
            return None, str(exc)
        fixture = fixtures.get(fixture_path)
        if not isinstance(fixture, dict) or not _is_sha256(fixture.get("sha256")):
            return None, f"golden fixture is absent from input capture: {fixture_path}"
        expected: dict[str, Any] = {"golden_path": fixture_path, "golden_sha256": fixture["sha256"]}
    elif _non_empty_string(oracle_reference):
        expected = {
            "oracle_reference": str(oracle_reference).strip(),
            "oracle_sha256": _sha256_bytes(str(oracle_reference).strip().encode("utf-8")),
        }
    else:
        return None, "golden_path or oracle_reference is missing"
    expected["authority"] = authority
    expected["authority_sha256"] = _canonical_sha256(authority)
    return expected, None


def _chunked(values: Iterable[str], size: int) -> list[list[str]]:
    values = list(values)
    return [values[index : index + size] for index in range(0, len(values), size)]


def _execution_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "FINCORE_0042_R2_DENY_NETWORK": "1",
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _run_source_batches(
    *,
    source_root: Path,
    python_executable: str,
    nodeids: list[str],
    batch_size: int,
    timeout_seconds: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    executions: dict[str, dict[str, Any]] = {}
    node_execution_ids: dict[str, str] = {}
    for index, batch in enumerate(_chunked(nodeids, batch_size), start=1):
        execution_id = f"source-{index:04d}"
        argv = [
            python_executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "-q",
            "--tb=short",
            "--maxfail=0",
            *batch,
        ]
        try:
            completed = subprocess.run(
                argv,
                cwd=source_root,
                env=_execution_environment(),
                capture_output=True,
                text=False,
                check=False,
                timeout=timeout_seconds,
            )
            output = completed.stdout + b"\n--- STDERR ---\n" + completed.stderr
            exit_code = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, bytes) else (exc.stdout or "").encode("utf-8")
            stderr = exc.stderr if isinstance(exc.stderr, bytes) else (exc.stderr or "").encode("utf-8")
            output = stdout + b"\n--- STDERR ---\n" + stderr
            exit_code = 124
            timed_out = True
        executions[execution_id] = {
            "argv": argv,
            "exit_code": exit_code,
            "nodeids": batch,
            "output_sha256": _sha256_bytes(output),
            "timed_out": timed_out,
        }
        for nodeid in batch:
            node_execution_ids[nodeid] = execution_id
    return executions, node_execution_ids


def _scenario_output_sha256(execution_ids: list[str], executions: dict[str, dict[str, Any]]) -> str:
    return _canonical_sha256(
        [
            {
                "execution_id": execution_id,
                "output_sha256": executions[execution_id]["output_sha256"],
                "exit_code": executions[execution_id]["exit_code"],
            }
            for execution_id in execution_ids
        ]
    )


def _evaluate_entries(
    *,
    entries: list[dict[str, Any]],
    capture_sha256: str,
    ledger_sha256: str,
    fixtures: dict[str, Any],
    source_identity: dict[str, str | bool],
    executions: dict[str, dict[str, Any]],
    node_execution_ids: dict[str, str],
) -> dict[str, dict[str, Any]]:
    capabilities: dict[str, dict[str, Any]] = {}
    for entry in entries:
        capability_id = entry.get("capability_id")
        if not _non_empty_string(capability_id):
            raise EvaluationUsageError("captured ledger entry has no capability_id")
        capability_id = str(capability_id).strip()
        if capability_id in capabilities:
            raise EvaluationUsageError(f"captured ledger repeats capability_id {capability_id!r}")
        source_nodeids = _entry_source_nodeids(entry)
        wheel_nodeids = _entry_wheel_nodeids(entry)
        scenarios = _entry_scenarios(entry)
        pending_reasons: list[str] = []
        if not source_nodeids:
            pending_reasons.append("source_nodeids are missing")
        if not wheel_nodeids:
            pending_reasons.append("wheel_nodeids are missing")
        if not scenarios:
            pending_reasons.append("scenarios are missing")

        scenario_records: list[dict[str, Any]] = []
        for scenario in scenarios:
            expected, reason = _scenario_prerequisites(scenario, fixtures)
            scenario_id = scenario.get("scenario_id") if isinstance(scenario.get("scenario_id"), str) else "<unknown>"
            if reason is not None:
                pending_reasons.append(f"scenario {scenario_id}: {reason}")
                scenario_records.append({"scenario_id": scenario_id, "status": "pending", "reason": reason})
                continue
            assert expected is not None
            execution_ids = sorted(
                {node_execution_ids[nodeid] for nodeid in source_nodeids if nodeid in node_execution_ids}
            )
            if len(execution_ids) != len({node_execution_ids.get(nodeid) for nodeid in source_nodeids}):
                pending_reasons.append(f"scenario {scenario_id}: source nodeids were not executed")
                scenario_records.append(
                    {
                        **expected,
                        "scenario_id": scenario_id,
                        "status": "pending",
                        "reason": "source nodeids were not executed",
                    }
                )
                continue
            failed_execution_ids = [
                execution_id for execution_id in execution_ids if executions[execution_id]["exit_code"] != 0
            ]
            status = "failed" if failed_execution_ids else "ok"
            scenario_records.append(
                {
                    **expected,
                    "scenario_id": scenario_id,
                    "status": status,
                    "execution_ids": execution_ids,
                    "failed_execution_ids": failed_execution_ids,
                    "output_sha256": _scenario_output_sha256(execution_ids, executions),
                }
            )

        statuses = {record["status"] for record in scenario_records}
        status = "pending" if pending_reasons or "pending" in statuses else "failed" if "failed" in statuses else "ok"
        capabilities[capability_id] = {
            "status": status,
            "input_capture_sha256": capture_sha256,
            "ledger_sha256": ledger_sha256,
            "target_operation_id": entry.get("target_operation_id"),
            "source_identity": source_identity,
            "source_nodeids": source_nodeids,
            "wheel_nodeids": wheel_nodeids,
            "wheel_evaluation": "deferred_to_candidate_wheel_gate",
            "scenarios": scenario_records,
            "pending_reasons": pending_reasons,
        }
    return capabilities


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


def evaluate(
    *,
    input_capture_path: Path,
    source_root: Path,
    tooling_root: Path,
    output_path: Path,
    families: list[str],
    python_executable: str,
    batch_size: int,
    timeout_seconds: int,
    deny_network: bool,
) -> tuple[dict[str, Any], int]:
    """Run source scenarios and return an evaluated artifact plus its exit code."""
    if not deny_network:
        raise EvaluationUsageError("--deny-network is required for scenario evaluation")
    if batch_size <= 0:
        raise EvaluationUsageError("--batch-size must be positive")
    if timeout_seconds <= 0:
        raise EvaluationUsageError("--timeout-seconds must be positive")
    source_root = source_root.resolve()
    tooling_root = tooling_root.resolve()
    output_path = _require_external_output(output_path, source_root, tooling_root)
    tooling_identity = _tooling_identity(tooling_root)
    source_identity = _worktree_identity(source_root, "source root")
    capture_path = input_capture_path.expanduser().resolve()
    if not capture_path.is_file():
        raise EvaluationUsageError(f"input capture does not exist: {capture_path}")
    try:
        capture_path.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise EvaluationUsageError("input capture must be outside the source worktree")
    try:
        capture_path.relative_to(tooling_root)
    except ValueError:
        pass
    else:
        raise EvaluationUsageError("input capture must be outside the tooling worktree")
    input_capture = _load_json(capture_path, "input capture")
    input_capture_sha256 = _sha256_bytes(capture_path.read_bytes())
    _require_capture_source(input_capture, source_identity)
    ledger, ledger_identity = _load_frozen_ledger(input_capture, source_root, source_identity)
    selected_families = _selected_families(families)
    all_entries = ledger["entries"]
    assert isinstance(all_entries, list)
    entries = [
        entry
        for entry in all_entries
        if isinstance(entry, dict)
        and entry.get("disposition") == "required"
        and entry.get("owner") in selected_families
    ]
    if not entries:
        raise EvaluationBlockedError(
            f"captured ledger has no required capabilities for: {', '.join(sorted(selected_families))}"
        )
    fixtures = _verify_fixture_manifest(input_capture, source_root, source_identity)

    executable_entries = [entry for entry in entries if _entry_source_nodeids(entry)]
    unique_nodeids = sorted({nodeid for entry in executable_entries for nodeid in _entry_source_nodeids(entry)})
    executions, node_execution_ids = _run_source_batches(
        source_root=source_root,
        python_executable=python_executable,
        nodeids=unique_nodeids,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
    )
    capabilities = _evaluate_entries(
        entries=entries,
        capture_sha256=input_capture_sha256,
        ledger_sha256=ledger_identity["sha256"],
        fixtures=fixtures,
        source_identity=source_identity,
        executions=executions,
        node_execution_ids=node_execution_ids,
    )
    final_source_identity = _worktree_identity(source_root, "source root")
    final_tooling_identity = _worktree_identity(tooling_root, "tooling root")
    if final_source_identity != source_identity:
        raise EvaluationUsageError("source worktree identity changed during scenario evaluation")
    if final_tooling_identity != tooling_identity["source"]:
        raise EvaluationUsageError("tooling worktree identity changed during scenario evaluation")

    statuses = {record["status"] for record in capabilities.values()}
    verdict = "BLOCKED" if "pending" in statuses else "FAIL" if "failed" in statuses else "PASS"
    exit_code = {"PASS": EXIT_PASS, "FAIL": EXIT_FAIL, "BLOCKED": EXIT_BLOCKED}[verdict]
    artifact = {
        **input_capture,
        "capabilities": capabilities,
        "evaluation": {
            "artifact_type": "capability_scenario_evaluation",
            "batch_size": batch_size,
            "input_capture_sha256": input_capture_sha256,
            "ledger": ledger_identity,
            "mode": "baseline_source",
            "python_executable": python_executable,
            "recorded_at": datetime.now(UTC).isoformat(),
            "source": source_identity,
            "tooling": tooling_identity,
            "selected_families": sorted(selected_families),
            "timeout_seconds": timeout_seconds,
            "verdict": verdict,
        },
        "evaluation_status": _EVALUATION_STATUS,
        "executions": executions,
    }
    _atomic_write_json(output_path, artifact)
    return artifact, exit_code


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-capture", required=True, help="external input-only capability capture JSON")
    parser.add_argument("--source-root", required=True, help="clean D0 source worktree root")
    parser.add_argument("--tooling-root", required=True, help="clean worktree containing this frozen evaluator")
    parser.add_argument("--output", required=True, help="external evaluated capability baseline JSON")
    parser.add_argument("--families", nargs="+", default=["all"], help="capability families or 'all'")
    parser.add_argument(
        "--python-executable", default=sys.executable, help="Python executable used for source scenarios"
    )
    parser.add_argument("--batch-size", type=int, default=50, help="source nodeids per pytest invocation")
    parser.add_argument("--timeout-seconds", type=int, default=600, help="timeout for each pytest invocation")
    parser.add_argument(
        "--deny-network", action="store_true", help="acknowledge that scenario execution must not use network"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        artifact, exit_code = evaluate(
            input_capture_path=Path(args.input_capture),
            source_root=Path(args.source_root),
            tooling_root=Path(args.tooling_root),
            output_path=Path(args.output),
            families=list(args.families),
            python_executable=args.python_executable,
            batch_size=args.batch_size,
            timeout_seconds=args.timeout_seconds,
            deny_network=args.deny_network,
        )
    except EvaluationUsageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except EvaluationBlockedError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    print(
        f"evaluated {len(artifact['capabilities'])} capability scenarios for "
        f"{artifact['evaluation']['source']['commit']}: {artifact['evaluation']['verdict']}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
