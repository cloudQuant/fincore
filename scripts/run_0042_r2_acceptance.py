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
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_BLOCKED = 3

_VERDICT_EXIT = {"PASS": EXIT_PASS, "FAIL": EXIT_FAIL, "BLOCKED": EXIT_BLOCKED}
_GATE_MANIFEST_RELATIVE = Path("tests") / "parity" / "fixtures" / "0042-r2-gate-manifest.json"
_D0_BUNDLE_MANIFEST_NAME = "d0-bundle-manifest.json"
_D0_BUNDLE_ARTIFACT_TYPE = "fincore_0042_r2_d0_bundle"
_REQUIRED_D0_ARTIFACTS = frozenset(
    {"capability_baseline", "architecture_baseline", "performance_baseline", "quality_baseline"}
)
_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PYTHON_FULL_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_MATRIX_CELL_FIELDS = frozenset(
    {
        "argv_digest",
        "candidate_commit",
        "candidate_tree",
        "d0_bundle_digest",
        "d0_tooling_digest",
        "dependency_lane",
        "dependency_profile",
        "evidence_time",
        "matrix_contract_version",
        "os",
        "output_digest",
        "python_full_version",
        "runner_image",
        "verdict",
        "wheel_sha256",
    }
)
_LEGACY_MODULES = (
    "fincore.empyrical",
    "fincore.pyfolio",
    "fincore.alphalens",
    "fincore._registry",
    "fincore._dispatch",
    "fincore._compat",
    "fincore.api",
    "fincore.backends",
    "fincore.capabilities",
    "fincore.constants",
    "fincore.contracts",
    "fincore.core",
    "fincore.hooks",
    "fincore.plugin",
    "fincore.results",
    "fincore.tearsheets",
    "fincore.utils",
    "fincore.validation",
    "fincore._types",
    "fincore.report.artifacts",
    "fincore.report.compute",
    "fincore.report.format",
    "fincore.report.model",
    "fincore.report.provenance",
    "fincore.report.render_html",
    "fincore.report.render_pdf",
)
_LEGACY_ROOT_EXPORTS = (
    "Empyrical",
    "Pyfolio",
    "alphalens",
    "analyze",
    "create_strategy_report",
    "sharpe_ratio",
    "cum_returns",
    "max_drawdown",
)


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
    tooling_root = _tooling_root()
    try:
        commit = _git(tooling_root, "rev-parse", "--verify", "HEAD")
        tree = _git(tooling_root, "rev-parse", "--verify", "HEAD^{tree}")
    except RunnerBlockedError as exc:
        raise RunnerUsageError(f"runner must execute from a Git tooling worktree: {exc}") from exc
    return {
        "commit": commit,
        "runner_path": str(script),
        "runner_blob_sha256": _sha256_file(script),
        "tree": tree,
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


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerUsageError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunnerUsageError(f"{label} must be a JSON object: {path}")
    return value


def _require_mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise RunnerUsageError(f"{label} must be an object")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunnerUsageError(f"{label} must be a non-empty string")
    return value


def _require_git_object(value: object, label: str) -> str:
    result = _require_string(value, label)
    if not _GIT_OBJECT_ID.fullmatch(result):
        raise RunnerUsageError(f"{label} must be a Git object identifier")
    return result


def _require_sha256(value: object, label: str) -> str:
    result = _require_string(value, label)
    if not _SHA256.fullmatch(result):
        raise RunnerUsageError(f"{label} must be a lowercase SHA256")
    return result


def _bundle_file(bundle: Path, relative: object, label: str) -> Path:
    raw = _require_string(relative, label)
    path = Path(raw)
    if path.is_absolute() or "\\" in raw or any(part in {"", ".", ".."} for part in path.parts):
        raise RunnerUsageError(f"{label} must be a non-escaping POSIX relative path")
    candidate = (bundle / path).resolve()
    try:
        candidate.relative_to(bundle)
    except ValueError as exc:
        raise RunnerUsageError(f"{label} escapes the D0 bundle") from exc
    if not candidate.is_file():
        raise RunnerUsageError(f"{label} is missing or not a regular file: {candidate}")
    return candidate


def _load_d0_bundle(bundle: Path, tooling_identity: dict[str, str]) -> dict:
    """Validate the immutable D0 inputs before any candidate command starts."""

    if not bundle.is_dir():
        raise RunnerUsageError(f"--expected-bundle must be a directory: {bundle}")
    manifest_path = bundle / _D0_BUNDLE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise RunnerUsageError(f"D0 bundle manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path, "D0 bundle manifest")
    if manifest.get("artifact_type") != _D0_BUNDLE_ARTIFACT_TYPE or manifest.get("schema_version") != 1:
        raise RunnerUsageError("D0 bundle manifest does not declare the frozen 0042-R2 bundle schema")

    tooling = _require_mapping(manifest.get("tooling"), "D0 bundle tooling")
    if _require_git_object(tooling.get("commit"), "D0 bundle tooling.commit") != tooling_identity.get("commit"):
        raise RunnerBlockedError("D0 bundle tooling commit does not match the executing runner")
    if _require_git_object(tooling.get("tree"), "D0 bundle tooling.tree") != tooling_identity.get("tree"):
        raise RunnerBlockedError("D0 bundle tooling tree does not match the executing runner")
    files = _require_mapping(tooling.get("files"), "D0 bundle tooling.files")
    runner_relative = "scripts/run_0042_r2_acceptance.py"
    expected_runner_blob = _require_sha256(files.get(runner_relative), f"D0 bundle tooling.files[{runner_relative!r}]")
    if expected_runner_blob != tooling_identity.get("runner_blob_sha256"):
        raise RunnerBlockedError("D0 bundle runner blob does not match the executing runner bytes")

    baseline_source = _require_mapping(manifest.get("baseline_source"), "D0 bundle baseline_source")
    _require_git_object(baseline_source.get("commit"), "D0 bundle baseline_source.commit")
    _require_git_object(baseline_source.get("tree"), "D0 bundle baseline_source.tree")
    provisioning = _require_mapping(baseline_source.get("provisioning"), "D0 bundle baseline_source.provisioning")
    git_bundle = _require_mapping(provisioning.get("git_bundle"), "D0 bundle baseline_source.provisioning.git_bundle")
    baseline_source_bundle = _bundle_file(bundle, git_bundle.get("path"), "D0 source git bundle path")
    if _sha256_file(baseline_source_bundle) != _require_sha256(git_bundle.get("sha256"), "D0 source git bundle SHA256"):
        raise RunnerBlockedError("D0 source git bundle digest does not match its manifest")

    manifest_artifacts = _require_mapping(manifest.get("artifacts"), "D0 bundle artifacts")
    if set(manifest_artifacts) != _REQUIRED_D0_ARTIFACTS:
        raise RunnerUsageError("D0 bundle artifacts must contain exactly " + ", ".join(sorted(_REQUIRED_D0_ARTIFACTS)))
    artifacts: dict[str, Path] = {}
    for name in sorted(_REQUIRED_D0_ARTIFACTS):
        specification = _require_mapping(manifest_artifacts[name], f"D0 bundle artifact {name}")
        artifact_path = _bundle_file(bundle, specification.get("path"), f"D0 bundle artifact {name}.path")
        if _sha256_file(artifact_path) != _require_sha256(
            specification.get("sha256"), f"D0 bundle artifact {name}.sha256"
        ):
            raise RunnerBlockedError(f"D0 bundle artifact digest does not match for {name}")
        artifacts[name] = artifact_path

    python_support_window = manifest.get("python_support_window")
    if (
        not isinstance(python_support_window, list)
        or not python_support_window
        or not all(
            isinstance(version, str) and _PYTHON_FULL_VERSION.fullmatch(version) for version in python_support_window
        )
        or python_support_window != sorted(set(python_support_window))
    ):
        raise RunnerUsageError(
            "D0 bundle python_support_window must be a non-empty sorted list of full Python versions"
        )

    return {
        "artifacts": artifacts,
        "baseline_source": baseline_source,
        "baseline_source_bundle": baseline_source_bundle,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "python_support_window": tuple(python_support_window),
    }


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


def _require_external_directory(value: str | None, label: str, *protected_roots: Path) -> Path:
    if not value:
        raise RunnerUsageError(f"{label} was not supplied")
    directory = Path(value).expanduser().resolve()
    for protected in protected_roots:
        try:
            directory.relative_to(protected)
        except ValueError:
            continue
        raise RunnerUsageError(f"{label} must live outside the {protected.name or 'protected'} worktree: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _candidate_identity(candidate_root: Path, expected_head: str | None) -> dict[str, str]:
    if not candidate_root.is_dir():
        raise RunnerUsageError(f"--candidate-root must be a directory: {candidate_root}")
    top_level = Path(_git(candidate_root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != candidate_root:
        raise RunnerUsageError("--candidate-root must identify the Git worktree root")
    if _git(candidate_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RunnerBlockedError("candidate worktree must be clean before formal acceptance")
    commit = _git(candidate_root, "rev-parse", "--verify", "HEAD")
    tree = _git(candidate_root, "rev-parse", "--verify", "HEAD^{tree}")
    if expected_head:
        requested = _git(candidate_root, "rev-parse", "--verify", expected_head)
        if requested != commit:
            raise RunnerBlockedError("--candidate-head does not match the clean candidate HEAD")
    return {"commit": commit, "root": str(candidate_root), "tree": tree}


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _frozen_tool(bundle: dict, relative: str) -> Path:
    specification = _require_mapping(bundle["manifest"]["tooling"].get("files"), "D0 bundle tooling.files")
    expected = _require_sha256(specification.get(relative), f"D0 bundle tooling.files[{relative!r}]")
    script = (_tooling_root() / relative).resolve()
    try:
        script.relative_to(_tooling_root())
    except ValueError as exc:
        raise RunnerUsageError(f"frozen tool path escapes the tooling worktree: {relative}") from exc
    if not script.is_file():
        raise RunnerBlockedError(f"frozen tool is missing: {relative}")
    if _sha256_file(script) != expected:
        raise RunnerBlockedError(f"frozen tool bytes do not match the D0 tooling manifest: {relative}")
    return script


def _validate_frozen_tooling(bundle: dict, tooling_identity: dict[str, str]) -> None:
    """Require an immutable detached tooling checkout before technical gates."""

    tooling_root = _tooling_root()
    if _git(tooling_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RunnerBlockedError("the acceptance tooling worktree must be clean")
    if bundle["manifest"]["tooling"].get("commit") != tooling_identity["commit"]:
        raise RunnerBlockedError("the acceptance tooling commit is not the D0-frozen commit")
    if bundle["manifest"]["tooling"].get("tree") != tooling_identity["tree"]:
        raise RunnerBlockedError("the acceptance tooling tree is not the D0-frozen tree")
    runner_relative = "scripts/run_0042_r2_acceptance.py"
    frozen_runner = subprocess.run(
        ["git", "show", f"{tooling_identity['commit']}:{runner_relative}"],
        cwd=tooling_root,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if frozen_runner.returncode != 0:
        raise RunnerBlockedError("cannot load the acceptance runner blob from the frozen tooling commit")
    if _sha256_bytes(frozen_runner.stdout) != tooling_identity["runner_blob_sha256"]:
        raise RunnerBlockedError("executing runner bytes do not match the frozen tooling Git blob")


def _run_frozen_pytest(
    *,
    candidate_root: Path,
    pytest_arguments: list[str],
    timeout_seconds: int,
    package_root: Path | None = None,
) -> dict:
    """Run frozen test files while forcing imports to resolve to candidate source."""

    tooling_root = _tooling_root()
    imported_root = package_root or candidate_root
    launcher = (
        "import sys; from pathlib import Path; "
        "candidate=Path(sys.argv[1]).resolve(); tooling=Path(sys.argv[2]).resolve(); "
        "sys.path[:]=[str(candidate),*[entry for entry in sys.path if Path(entry or '.').resolve()!=tooling]]; "
        "import pytest; raise SystemExit(pytest.main(sys.argv[3:]))"
    )
    command = [
        sys.executable,
        "-I",
        "-c",
        launcher,
        str(imported_root),
        str(tooling_root),
        "--import-mode=importlib",
        "-c",
        str(tooling_root / "pyproject.toml"),
        *pytest_arguments,
    ]
    return _run_frozen_command(command=command, candidate_root=candidate_root, timeout_seconds=timeout_seconds)


def _require_requested_flag(value: bool, name: str, gate: str) -> None:
    if not value:
        raise RunnerBlockedError(f"{gate} requires --{name}")


def _run_tests_gate(args: argparse.Namespace, candidate_root: Path) -> dict:
    _require_requested_flag(args.include_slow, "include-slow", "tests")
    _require_requested_flag(args.include_serial, "include-serial", "tests")
    _require_requested_flag(args.include_offline_integration, "include-offline-integration", "tests")
    if args.benchmarks_covered_by != "performance":
        raise RunnerBlockedError("tests requires --benchmarks-covered-by performance")
    record = _run_frozen_pytest(
        candidate_root=candidate_root,
        pytest_arguments=[
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:rerunfailures",
            "--tb=short",
            "--maxfail=0",
            "-m",
            "not integration_online and not benchmark",
            str(_tooling_root() / "tests"),
        ],
        timeout_seconds=1800,
    )
    return {"command": record, "verdict": "PASS" if record["exit_code"] == 0 else "FAIL"}


def _run_static_command(command: list[str], candidate_root: Path) -> dict:
    return _run_frozen_command(command=command, candidate_root=candidate_root, timeout_seconds=900)


def _run_static_gate(candidate_root: Path, output_dir: Path) -> dict:
    site_dir = output_dir / "mkdocs-site"
    commands = [
        [sys.executable, "-I", "-m", "ruff", "format", "--check", str(candidate_root)],
        [sys.executable, "-I", "-m", "ruff", "check", str(candidate_root)],
        [sys.executable, "-I", "-m", "mypy", str(candidate_root / "fincore")],
        [sys.executable, "-I", "-m", "mkdocs", "build", "--strict", "--site-dir", str(site_dir)],
    ]
    records = [_run_static_command(command, candidate_root) for command in commands]
    return {
        "commands": records,
        "verdict": "PASS" if all(record["exit_code"] == 0 for record in records) else "FAIL",
    }


def _require_candidate_distribution(args: argparse.Namespace, candidate_root: Path) -> tuple[Path, Path, Path]:
    wheel = _require_existing_path(args.candidate_wheel, "--candidate-wheel")
    distribution = _require_existing_path(args.candidate_dist, "--candidate-dist")
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise RunnerUsageError("--candidate-wheel must identify one wheel file")
    if not distribution.is_dir():
        raise RunnerUsageError("--candidate-dist must identify a distribution directory")
    for path, label in ((wheel, "--candidate-wheel"), (distribution, "--candidate-dist")):
        try:
            path.relative_to(candidate_root)
        except ValueError:
            continue
        raise RunnerUsageError(f"{label} must live outside the candidate worktree")
    wheels = sorted(distribution.glob("fincore-*.whl"))
    if wheels != [wheel]:
        raise RunnerBlockedError("candidate distribution must contain exactly the supplied one Fincore wheel")
    sdists = sorted(distribution.glob("fincore-*.tar.gz"))
    if len(sdists) != 1:
        raise RunnerBlockedError("candidate distribution must contain exactly one Fincore source distribution")
    return wheel, distribution, sdists[0]


def _normalized_extra(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _wheel_legacy_zero(wheel: Path) -> list[str]:
    violations: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        forbidden_prefixes = ("fincore/alphalens/", "fincore/empyrical", "fincore/pyfolio", "fincore/tearsheets/")
        offending = sorted(name for name in names if name.startswith(forbidden_prefixes))
        if offending:
            violations.append("wheel retains legacy runtime paths: " + ", ".join(offending))
        metadata_name = next((name for name in names if name.endswith(".dist-info/METADATA")), None)
        if metadata_name is None:
            violations.append("wheel has no dist-info METADATA")
        else:
            metadata = archive.read(metadata_name).decode("utf-8", "replace")
            extras = {
                _normalized_extra(line.split(":", 1)[1].strip())
                for line in metadata.splitlines()
                if line.casefold().startswith("provides-extra:")
            }
            prohibited = {"alphalens", "empyrical", "pyfolio", "viz", "datareader"} & extras
            if prohibited:
                violations.append("wheel exposes retired extras: " + ", ".join(sorted(prohibited)))
    return violations


def _sdist_matches_candidate_source(sdist: Path, candidate_root: Path) -> list[str]:
    tracked = _git(candidate_root, "ls-files", "-z").split("\0")
    required = [
        path
        for path in tracked
        if path.startswith("fincore/") or path in {"pyproject.toml", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"}
    ]
    violations: list[str] = []
    with tarfile.open(sdist) as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        roots = {name.split("/", 1)[0] for name in members if "/" in name}
        if len(roots) != 1:
            return ["sdist must contain one package root"]
        root = next(iter(roots))
        for relative in required:
            member = members.get(f"{root}/{relative}")
            if member is None:
                violations.append(f"sdist omits tracked source file {relative}")
                continue
            stream = archive.extractfile(member)
            if stream is None or stream.read() != (candidate_root / relative).read_bytes():
                violations.append(f"sdist content differs for {relative}")
    return violations


def _run_package_gate(args: argparse.Namespace, bundle: dict, candidate_root: Path) -> dict:
    _require_requested_flag(args.require_one_sdist, "require-one-sdist", "package")
    _require_requested_flag(args.require_sdist_source_equivalence, "require-sdist-source-equivalence", "package")
    _require_requested_flag(args.require_legacy_zero, "require-legacy-zero", "package")
    wheel, distribution, sdist = _require_candidate_distribution(args, candidate_root)
    script = _frozen_tool(bundle, "scripts/check_release_consistency.py")
    record = _run_frozen_command(
        command=[sys.executable, "-I", str(script), "--dist", str(distribution)],
        candidate_root=candidate_root,
        timeout_seconds=900,
    )
    legacy_violations = _wheel_legacy_zero(wheel)
    sdist_violations = _sdist_matches_candidate_source(sdist, candidate_root)
    return {
        "command": record,
        "sdist_sha256": _sha256_file(sdist),
        "sdist_source_violations": sdist_violations,
        "verdict": ("PASS" if record["exit_code"] == 0 and not legacy_violations and not sdist_violations else "FAIL"),
        "wheel_legacy_violations": legacy_violations,
        "wheel_sha256": _sha256_file(wheel),
    }


def _coverage_percent(coverage: dict) -> float:
    totals = _require_mapping(coverage.get("totals"), "coverage totals")
    if "percent_covered" in totals:
        return float(totals["percent_covered"])
    covered = int(totals.get("covered_lines", 0)) + int(totals.get("covered_branches", 0))
    total = int(totals.get("num_statements", 0)) + int(totals.get("num_branches", 0))
    if not total:
        raise RunnerBlockedError("coverage report has no measured statements or branches")
    return 100.0 * covered / total


def _quality_contract(bundle: dict) -> tuple[float, dict[str, float]]:
    contract = _require_mapping(bundle["manifest"].get("quality_contract"), "D0 bundle quality_contract")
    changed_lines_min = contract.get("changed_lines_min")
    critical = _require_mapping(contract.get("critical_modules"), "D0 bundle quality_contract.critical_modules")
    if not isinstance(changed_lines_min, (int, float)) or isinstance(changed_lines_min, bool):
        raise RunnerBlockedError("D0 quality contract changed_lines_min is invalid")
    normalized: dict[str, float] = {}
    for path, minimum in critical.items():
        if (
            not isinstance(path, str)
            or not path.startswith("fincore/")
            or not isinstance(minimum, (int, float))
            or isinstance(minimum, bool)
            or not 0.0 <= float(minimum) <= 100.0
        ):
            raise RunnerBlockedError("D0 quality contract critical_modules is invalid")
        normalized[path] = float(minimum)
    if not normalized:
        raise RunnerBlockedError("D0 quality contract has no critical modules")
    return float(changed_lines_min), normalized


def _critical_coverage_violations(
    coverage: dict,
    candidate_root: Path,
    critical_modules: dict[str, float],
) -> list[str]:
    files = _require_mapping(coverage.get("files"), "coverage files")
    normalized_files: dict[str, dict] = {}
    for raw_path, data in files.items():
        if not isinstance(raw_path, str) or not isinstance(data, dict):
            continue
        path = Path(raw_path)
        try:
            relative = path.resolve().relative_to(candidate_root).as_posix()
        except ValueError:
            relative = raw_path
        normalized_files[relative] = data
    violations: list[str] = []
    for path, minimum in sorted(critical_modules.items()):
        file_data = normalized_files.get(path)
        if file_data is None:
            violations.append(f"critical module is absent from the coverage report: {path}")
            continue
        summary = _require_mapping(file_data.get("summary"), f"coverage summary for {path}")
        actual = float(summary.get("percent_covered", -1.0))
        if actual + 1e-9 < minimum:
            violations.append(f"critical module {path} coverage {actual:.2f}% is below {minimum:.2f}%")
    return violations


def _run_quality_gate(args: argparse.Namespace, bundle: dict, candidate_root: Path, output_dir: Path) -> dict:
    _require_requested_flag(args.require_fresh_coverage, "require-fresh-coverage", "quality")
    if args.require_changed_lines is None or args.require_changed_lines < 95.0:
        raise RunnerBlockedError("quality requires --require-changed-lines of at least 95")
    if args.require_critical_branches is None or args.require_critical_branches < 90.0:
        raise RunnerBlockedError("quality requires --require-critical-branches of at least 90")
    baseline = _artifact_json(bundle, "quality_baseline")
    source = _require_mapping(baseline.get("source"), "D0 quality source")
    if source.get("commit") != bundle["baseline_source"].get("commit") or source.get("dirty") is not False:
        raise RunnerBlockedError("D0 quality source provenance does not match the clean provisioned baseline")
    baseline_coverage = None
    for run in baseline.get("runs", []):
        if isinstance(run, dict) and run.get("label") == "branch-coverage":
            baseline_coverage = run.get("branch_coverage_percent")
            break
    if not isinstance(baseline_coverage, (int, float)):
        raise RunnerBlockedError("D0 quality baseline has no branch coverage measurement")
    changed_lines_min, critical_modules = _quality_contract(bundle)
    coverage_json = output_dir / "coverage.json"
    if coverage_json.exists():
        raise RunnerUsageError(f"quality coverage output already exists: {coverage_json}")
    pytest_record = _run_frozen_pytest(
        candidate_root=candidate_root,
        pytest_arguments=[
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:rerunfailures",
            "--tb=short",
            "--maxfail=0",
            "--cov=fincore",
            "--cov-branch",
            f"--cov-report=json:{coverage_json}",
            "-m",
            "not integration_online and not benchmark",
            str(_tooling_root() / "tests"),
        ],
        timeout_seconds=1800,
    )
    if pytest_record["exit_code"] != 0 or not coverage_json.is_file():
        return {"pytest": pytest_record, "verdict": "FAIL"}
    coverage = _read_json(coverage_json, "fresh candidate coverage")
    overall = _coverage_percent(coverage)
    threshold = max(float(baseline_coverage), 60.0)
    coverage_script = _frozen_tool(bundle, "scripts/check_coverage_baseline.py")
    coverage_record = _run_frozen_command(
        command=[
            sys.executable,
            "-I",
            str(coverage_script),
            "--coverage-json",
            str(coverage_json),
            "--baseline",
            str(bundle["artifacts"]["quality_baseline"]),
            "--changed-base",
            str(bundle["baseline_source"]["commit"]),
            "--changed-lines-min",
            str(max(changed_lines_min, float(args.require_changed_lines))),
        ],
        candidate_root=candidate_root,
        timeout_seconds=300,
    )
    critical_violations = _critical_coverage_violations(coverage, candidate_root, critical_modules)
    return {
        "baseline_branch_coverage": float(baseline_coverage),
        "coverage_json_sha256": _sha256_file(coverage_json),
        "coverage_gate": coverage_record,
        "critical_violations": critical_violations,
        "overall_branch_coverage": overall,
        "pytest": pytest_record,
        "required_branch_coverage": threshold,
        "verdict": (
            "PASS"
            if coverage_record["exit_code"] == 0 and overall + 1e-9 >= threshold and not critical_violations
            else "FAIL"
        ),
    }


def _profile_cases(payload: dict, label: str) -> dict[tuple[str, str], dict]:
    if payload.get("schema") != "fincore-workload-profiles-v2":
        raise RunnerBlockedError(f"{label} has an unexpected workload-profile schema")
    measurement = _require_mapping(payload.get("measurement"), f"{label} measurement")
    if measurement != {
        "warmups": 2,
        "repeats": 5,
        "require_output_digest": True,
        "timing_unit": "seconds",
        "percentile_method": "linear",
    }:
        raise RunnerBlockedError(f"{label} does not use the frozen 2-warmup/5-repeat measurement contract")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RunnerBlockedError(f"{label} has no workload cases")
    indexed: dict[tuple[str, str], dict] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise RunnerBlockedError(f"{label} contains a malformed workload case")
        workload = _require_mapping(case.get("workload"), f"{label} workload")
        key = (str(case.get("kind")), str(workload.get("size")))
        if key in indexed:
            raise RunnerBlockedError(f"{label} repeats workload case {key[0]}/{key[1]}")
        for field in ("execution_input_digest", "output_digest"):
            _require_sha256(case.get(field), f"{label} {key[0]}/{key[1]} {field}")
        _require_sha256(workload.get("input_digest"), f"{label} {key[0]}/{key[1]} workload input digest")
        indexed[key] = case
    return indexed


def _as_positive_float(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
        raise RunnerBlockedError(f"{label} must be a positive number")
    return float(value)


def _run_performance_gate(args: argparse.Namespace, bundle: dict, candidate_root: Path, output_dir: Path) -> dict:
    families = tuple(args.families)
    expected_families = ("metrics", "rolling", "transactions", "factor", "risk", "report")
    if set(families) != set(expected_families) or len(families) != len(expected_families):
        raise RunnerBlockedError("performance requires all six frozen workload families exactly once")
    if args.warmups != 2 or args.repeats != 5:
        raise RunnerBlockedError("performance requires exactly --warmups 2 --repeats 5")
    baseline_payload = _artifact_json(bundle, "performance_baseline")
    baseline_cases = _profile_cases(baseline_payload, "D0 performance baseline")
    output = output_dir / "candidate-performance.json"
    if output.exists():
        raise RunnerUsageError(f"candidate performance output already exists: {output}")
    profiler = _frozen_tool(bundle, "scripts/profile_workloads.py")
    _frozen_tool(bundle, "scripts/profile_hotspots.py")
    _frozen_tool(bundle, "scripts/_0042_r2_tooling.py")
    profile_record = _run_frozen_command(
        command=[
            sys.executable,
            "-I",
            str(profiler),
            "--sizes",
            "small",
            "medium",
            "large",
            "--kinds",
            *expected_families,
            "--warmups",
            "2",
            "--repeats",
            "5",
            "--require-output-digest",
            "--output",
            str(output),
        ],
        candidate_root=candidate_root,
        timeout_seconds=1800,
    )
    if profile_record["exit_code"] != 0 or not output.is_file():
        return {"profile": profile_record, "verdict": "FAIL"}
    candidate_payload = _read_json(output, "candidate performance profile")
    candidate_cases = _profile_cases(candidate_payload, "candidate performance profile")
    differences: list[str] = []
    improvements: list[str] = []
    rss_improvements: list[str] = []
    if set(candidate_cases) != set(baseline_cases):
        differences.append("candidate workload keys differ from D0")
    for key in sorted(set(candidate_cases) & set(baseline_cases)):
        baseline_case = baseline_cases[key]
        candidate_case = candidate_cases[key]
        label = f"{key[0]}/{key[1]}"
        differences.extend(
            f"{label} {field} differs from D0"
            for field in ("execution_input_digest", "output_digest")
            if candidate_case.get(field) != baseline_case.get(field)
        )
        candidate_workload = _require_mapping(candidate_case.get("workload"), f"candidate {label} workload")
        baseline_workload = _require_mapping(baseline_case.get("workload"), f"D0 {label} workload")
        if candidate_workload.get("input_digest") != baseline_workload.get("input_digest"):
            differences.append(f"{label} workload input digest differs from D0")
        candidate_timing = _require_mapping(candidate_case.get("timing"), f"candidate {label} timing")
        baseline_timing = _require_mapping(baseline_case.get("timing"), f"D0 {label} timing")
        candidate_median = _as_positive_float(candidate_timing.get("median_seconds"), f"candidate {label} median")
        baseline_median = _as_positive_float(baseline_timing.get("median_seconds"), f"D0 {label} median")
        candidate_p95 = _as_positive_float(candidate_timing.get("p95_seconds"), f"candidate {label} p95")
        baseline_p95 = _as_positive_float(baseline_timing.get("p95_seconds"), f"D0 {label} p95")
        candidate_rss = _as_positive_float(candidate_case.get("peak_rss_bytes"), f"candidate {label} RSS")
        baseline_rss = _as_positive_float(baseline_case.get("peak_rss_bytes"), f"D0 {label} RSS")
        if candidate_median > baseline_median * 1.10:
            differences.append(f"{label} median regresses by more than 10%")
        if candidate_p95 > baseline_p95 * 1.15:
            differences.append(f"{label} p95 regresses by more than 15%")
        if candidate_rss > baseline_rss * 1.10:
            differences.append(f"{label} RSS regresses by more than 10%")
        if candidate_median <= baseline_median * 0.80:
            improvements.append(label)
        if candidate_rss <= baseline_rss * 0.70:
            rss_improvements.append(label)
        provenance = _require_mapping(candidate_case.get("provenance"), f"candidate {label} provenance")
        if (
            provenance.get("commit") != _git(candidate_root, "rev-parse", "HEAD")
            or provenance.get("dirty") is not False
        ):
            differences.append(f"{label} profile provenance is not the clean candidate")
        baseline_provenance = _require_mapping(baseline_case.get("provenance"), f"D0 {label} provenance")
        differences.extend(
            f"{label} {field} differs from the D0 measurement platform"
            for field in ("platform_label", "python", "numpy", "pandas")
            if provenance.get(field) != baseline_provenance.get(field)
        )
    performance_script = _frozen_tool(bundle, "scripts/check_performance.py")
    absolute_record = _run_frozen_command(
        command=[sys.executable, "-I", str(performance_script)],
        candidate_root=candidate_root,
        timeout_seconds=300,
    )
    improvement_met = len(improvements) >= 3 or (len(improvements) >= 2 and bool(rss_improvements))
    if not improvement_met:
        differences.append("candidate does not meet the required hotspot improvement threshold")
    return {
        "absolute_budget": absolute_record,
        "candidate_profile_sha256": _sha256_file(output),
        "improved_median_cases": improvements,
        "improved_rss_cases": rss_improvements,
        "profile": profile_record,
        "regressions_or_differences": differences,
        "verdict": "PASS" if not differences and absolute_record["exit_code"] == 0 else "FAIL",
    }


def _run_installed_gate(args: argparse.Namespace, bundle: dict, candidate_root: Path) -> dict:
    required_profiles = {
        "core",
        "factor-analysis",
        "visualization",
        "report-pdf",
        "report-xlsx",
        "bayesian",
        "all",
    }
    profiles = tuple(args.profiles)
    if set(profiles) != required_profiles or len(profiles) != len(required_profiles):
        raise RunnerBlockedError("installed requires the frozen direct-capability profile set exactly once")
    if args.data_providers != ["all"]:
        raise RunnerBlockedError("installed requires --data-providers all")
    if set(args.dependency_lanes) != {"minimum", "latest"} or len(args.dependency_lanes) != 2:
        raise RunnerBlockedError("installed requires exactly --dependency-lanes minimum latest")
    wheel, distribution, _ = _require_candidate_distribution(args, candidate_root)
    consumer = _frozen_tool(bundle, "scripts/test_installed_wheel.py")
    _frozen_tool(bundle, "scripts/_0042_r2_tooling.py")
    consumer_record = _run_frozen_command(
        command=[
            sys.executable,
            "-I",
            str(consumer),
            "--dist",
            str(distribution),
            "--profiles",
            *profiles,
            "--data-providers",
            "all",
        ],
        candidate_root=candidate_root,
        timeout_seconds=3600,
    )
    dependency_checker = _frozen_tool(bundle, "scripts/check_dependency_matrix.py")
    lanes = [
        _run_frozen_command(
            command=[
                sys.executable,
                "-I",
                str(dependency_checker),
                "--constraints",
                str(_tooling_root() / "constraints"),
                "--wheel",
                str(wheel),
                "--lane",
                lane,
            ],
            candidate_root=candidate_root,
            timeout_seconds=1800,
        )
        for lane in ("minimum", "latest")
    ]
    return {
        "consumer": consumer_record,
        "dependency_lanes": dict(zip(("minimum", "latest"), lanes, strict=True)),
        "wheel_sha256": _sha256_file(wheel),
        "verdict": "PASS"
        if consumer_record["exit_code"] == 0 and all(lane["exit_code"] == 0 for lane in lanes)
        else "FAIL",
    }


def _frozen_nodeid(nodeid: object) -> str:
    raw = _require_string(nodeid, "capability scenario nodeid")
    path, separator, suffix = raw.partition("::")
    if not path.startswith("tests/") or not path.endswith(".py"):
        raise RunnerBlockedError(f"capability scenario nodeid is not a frozen test path: {raw}")
    absolute = str(_tooling_root() / path)
    return absolute if not separator else absolute + separator + suffix


def _required_capability_nodeids(ledger: dict, field: str) -> list[str]:
    entries = ledger.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RunnerBlockedError("D0 capability ledger has no entries")
    nodeids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("disposition") != "required":
            continue
        raw_nodeids = entry.get(field)
        if not isinstance(raw_nodeids, list) or not raw_nodeids:
            raise RunnerBlockedError(f"required capability lacks {field}")
        nodeids.update(_frozen_nodeid(value) for value in raw_nodeids)
    if not nodeids:
        raise RunnerBlockedError(f"D0 capability ledger has no required {field}")
    return sorted(nodeids)


def _run_parity_gate(args: argparse.Namespace, bundle: dict, candidate_root: Path, output_dir: Path) -> dict:
    if args.families != ["all"]:
        raise RunnerBlockedError("parity requires --families all")
    _require_requested_flag(args.require_source_wheel_equal, "require-source-wheel-equal", "parity")
    wheel, _, _ = _require_candidate_distribution(args, candidate_root)
    baseline = _artifact_json(bundle, "capability_baseline")
    if (
        baseline.get("artifact_type") != "capability_baseline_capture"
        or baseline.get("evaluation_status") != "evaluated_source"
    ):
        raise RunnerBlockedError("D0 capability baseline is not an evaluated source artifact")
    evaluation = _require_mapping(baseline.get("evaluation"), "D0 capability evaluation")
    ledger_info = _require_mapping(evaluation.get("ledger"), "D0 capability evaluation ledger")
    ledger_relative = _require_string(ledger_info.get("path"), "D0 capability evaluation ledger path")
    baseline_root = _materialize_baseline_source(bundle)
    try:
        ledger_path = (baseline_root / ledger_relative).resolve()
        try:
            ledger_path.relative_to(baseline_root)
        except ValueError as exc:
            raise RunnerBlockedError("D0 capability ledger path escapes the baseline source") from exc
        if not ledger_path.is_file() or _sha256_file(ledger_path) != _require_sha256(
            ledger_info.get("sha256"), "D0 capability evaluation ledger SHA256"
        ):
            raise RunnerBlockedError("D0 capability ledger bytes do not match the evaluated baseline")
        ledger = _read_json(ledger_path, "D0 capability ledger")
        source_nodeids = _required_capability_nodeids(ledger, "source_nodeids")
        wheel_nodeids = _required_capability_nodeids(ledger, "wheel_nodeids")
        parity_checker = _frozen_tool(bundle, "scripts/check_feature_parity.py")
        d0_check_output = output_dir / "d0-parity-check.json"
        if d0_check_output.exists():
            raise RunnerUsageError(f"D0 parity output already exists: {d0_check_output}")
        d0_record = _run_frozen_command(
            command=[
                sys.executable,
                "-I",
                str(parity_checker),
                "--baseline",
                str(bundle["artifacts"]["capability_baseline"]),
                "--ledger",
                str(ledger_path),
                "--families",
                "all",
                "--output",
                str(d0_check_output),
            ],
            candidate_root=candidate_root,
            timeout_seconds=600,
        )
        source_record = _run_frozen_pytest(
            candidate_root=candidate_root,
            pytest_arguments=[
                "-o",
                "addopts=",
                "-p",
                "no:cacheprovider",
                "-p",
                "no:rerunfailures",
                "--tb=short",
                "--maxfail=0",
                *source_nodeids,
            ],
            timeout_seconds=1800,
        )
        wheel_target = output_dir / "wheel-target"
        if wheel_target.exists():
            raise RunnerUsageError(f"wheel parity target already exists: {wheel_target}")
        wheel_target.mkdir(parents=True)
        install_record = _run_frozen_command(
            command=[
                sys.executable,
                "-I",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(wheel_target),
                str(wheel),
            ],
            candidate_root=candidate_root,
            timeout_seconds=900,
        )
        wheel_record = (
            _run_frozen_pytest(
                candidate_root=candidate_root,
                package_root=wheel_target,
                pytest_arguments=[
                    "-o",
                    "addopts=",
                    "-p",
                    "no:cacheprovider",
                    "-p",
                    "no:rerunfailures",
                    "--tb=short",
                    "--maxfail=0",
                    *wheel_nodeids,
                ],
                timeout_seconds=1800,
            )
            if install_record["exit_code"] == 0
            else None
        )
    finally:
        _cleanup_materialized_source(baseline_root)
    return {
        "d0_baseline_check": d0_record,
        "source_scenarios": source_record,
        "wheel_install": install_record,
        "wheel_scenarios": wheel_record,
        "wheel_sha256": _sha256_file(wheel),
        "verdict": (
            "PASS"
            if d0_record["exit_code"] == 0
            and source_record["exit_code"] == 0
            and install_record["exit_code"] == 0
            and wheel_record is not None
            and wheel_record["exit_code"] == 0
            else "FAIL"
        ),
    }


def _run_report_gate(args: argparse.Namespace, candidate_root: Path, output_dir: Path) -> dict:
    if args.real_browser != "chromium":
        raise RunnerBlockedError("report requires --real-browser chromium")
    for name, value in (("real-html", args.real_html), ("real-pdf", args.real_pdf), ("real-xlsx", args.real_xlsx)):
        _require_requested_flag(value, name, "report")
    if set(args.interactive_backends) != {"plotly", "bokeh"} or len(args.interactive_backends) != 2:
        raise RunnerBlockedError("report requires exactly --interactive-backends plotly bokeh")
    artifact_root = output_dir / "artifacts"
    probe = (
        "import json,sys,sysconfig; from pathlib import Path; import numpy as np; import pandas as pd; "
        "root=Path(sys.argv[1]).resolve(); target=Path(sys.argv[2]).resolve(); "
        "sys.path[:0]=[str(root),sysconfig.get_paths()['purelib']]; "
        "from fincore.report.portfolio.compute import build_portfolio_report; "
        "from fincore.report.renderers.html import write_html; "
        "from fincore.report.renderers.pdf import write_pdf; "
        "from fincore.report.renderers.xlsx import write_xlsx; "
        "from fincore.report.renderers.interactive import render_bokeh,render_plotly; "
        "index=pd.date_range('2024-01-02',periods=32,freq='B'); "
        "returns=pd.Series(np.resize([0.01,-0.004,0.002,0.003],len(index)),index=index); "
        "document=build_portfolio_report(returns,rolling_window=8); "
        "html=write_html(document,target/'report.html'); pdf=write_pdf(document,target/'report.pdf'); "
        "xlsx=write_xlsx(document,target/'report.xlsx'); plotly=render_plotly(document); bokeh=render_bokeh(document); "
        "assert html.named_artifacts['file'].is_file(); assert pdf.named_artifacts['file'].is_file(); "
        "assert xlsx.named_artifacts['file'].is_file(); assert plotly.named_artifacts['figure'].data; "
        "assert bokeh.named_artifacts['figure'].renderers; "
        "print(json.dumps({name:(target/name).stat().st_size for name in ('report.html','report.pdf','report.xlsx')},sort_keys=True))"
    )
    artifact_root.mkdir(parents=True, exist_ok=False)
    record = _run_frozen_command(
        command=[sys.executable, "-I", "-c", probe, str(candidate_root), str(artifact_root)],
        candidate_root=candidate_root,
        timeout_seconds=900,
    )
    return {
        "artifact_paths": {name: str(artifact_root / name) for name in ("report.html", "report.pdf", "report.xlsx")},
        "command": record,
        "verdict": "PASS" if record["exit_code"] == 0 else "FAIL",
    }


def _current_matrix_os() -> str:
    system = platform.system().casefold()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    raise RunnerBlockedError(f"unsupported matrix operating system: {platform.system()}")


def _run_matrix_cell_gate(
    args: argparse.Namespace,
    bundle: dict,
    candidate: dict[str, str],
    tooling_identity: dict[str, str],
    output_dir: Path,
) -> dict:
    operating_system = _require_string(args.matrix_os, "--os")
    if operating_system != _current_matrix_os():
        raise RunnerBlockedError("--os does not match the executing matrix operating system")
    full_python = _require_string(args.python_full_version, "--python-full-version")
    if full_python != platform.python_version():
        raise RunnerBlockedError("--python-full-version does not match the executing interpreter")
    if full_python not in bundle["python_support_window"]:
        raise RunnerBlockedError("the executing Python is outside the D0-frozen support window")
    if args.dependency_lane not in {"minimum", "latest", "pinned"}:
        raise RunnerUsageError("matrix-cell requires --dependency-lane minimum, latest, or pinned")
    dependency_profile = _require_string(args.dependency_profile, "--dependency-profile")
    runner_image = _require_string(args.runner_image, "--runner-image")
    wheel = _require_existing_path(args.candidate_wheel, "--candidate-wheel")
    if not wheel.is_file():
        raise RunnerUsageError("--candidate-wheel must identify a regular file")
    wheel_target = output_dir / "wheel-target"
    if wheel_target.exists():
        raise RunnerUsageError(f"matrix wheel target already exists: {wheel_target}")
    wheel_target.mkdir(parents=True)
    install = _run_frozen_command(
        command=[
            sys.executable,
            "-I",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(wheel_target),
            str(wheel),
        ],
        candidate_root=Path(candidate["root"]),
        timeout_seconds=900,
    )
    test_record = (
        _run_frozen_pytest(
            candidate_root=Path(candidate["root"]),
            package_root=wheel_target,
            pytest_arguments=[
                "-o",
                "addopts=",
                "-p",
                "no:cacheprovider",
                "-p",
                "no:rerunfailures",
                "--tb=short",
                "--maxfail=0",
                "-m",
                "not integration_online and not benchmark",
                str(_tooling_root() / "tests"),
            ],
            timeout_seconds=1800,
        )
        if install["exit_code"] == 0
        else None
    )
    cell = {
        "argv_digest": _canonical_sha256(
            {
                "gate": "matrix-cell",
                "selector": "not integration_online and not benchmark",
                "tooling_tests": "tests",
            }
        ),
        "candidate_commit": candidate["commit"],
        "candidate_tree": candidate["tree"],
        "d0_bundle_digest": bundle["manifest_sha256"],
        "d0_tooling_digest": _canonical_sha256(bundle["manifest"]["tooling"]),
        "dependency_lane": args.dependency_lane,
        "dependency_profile": dependency_profile,
        "evidence_time": _utc_now(),
        "matrix_contract_version": 1,
        "os": operating_system,
        "output_digest": _canonical_sha256({"install": install, "tests": test_record}),
        "python_full_version": full_python,
        "runner_image": runner_image,
        "verdict": "PASS"
        if install["exit_code"] == 0 and test_record is not None and test_record["exit_code"] == 0
        else "FAIL",
        "wheel_sha256": _sha256_file(wheel),
    }
    cell_path = output_dir / "matrix-cell.json"
    if cell_path.exists():
        raise RunnerUsageError(f"matrix cell output already exists: {cell_path}")
    cell_path.write_text(json.dumps(cell, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "cell": cell,
        "cell_path": str(cell_path),
        "install": install,
        "tests": test_record,
        "verdict": cell["verdict"],
    }


def _candidate_environment(candidate_root: Path) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONHOME", "PYTHONPATH"}}
    environment.update(
        {
            "FINCORE_0042R2_SOURCE_ROOT": str(candidate_root),
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _run_frozen_command(
    *,
    command: list[str],
    candidate_root: Path,
    timeout_seconds: int,
) -> dict:
    try:
        completed = subprocess.run(
            command,
            cwd=candidate_root,
            env=_candidate_environment(candidate_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunnerBlockedError(f"frozen command could not complete: {exc}") from exc
    return {
        "argv": command,
        "exit_code": completed.returncode,
        "stderr_sha256": _sha256_bytes(completed.stderr.encode("utf-8")),
        "stdout_sha256": _sha256_bytes(completed.stdout.encode("utf-8")),
    }


def _materialize_baseline_source(bundle: dict) -> Path:
    baseline = bundle["baseline_source"]
    with tempfile.TemporaryDirectory(prefix="fincore-0042-r2-baseline-source-") as temporary:
        temporary_root = Path(temporary)
        checkout = temporary_root / "source"
        cloned = subprocess.run(
            ["git", "clone", "--quiet", str(bundle["baseline_source_bundle"]), str(checkout)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env={key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
        )
        if cloned.returncode != 0:
            detail = (cloned.stderr or cloned.stdout).strip()
            raise RunnerBlockedError(f"cannot materialize the baseline source bundle: {detail or 'git clone failed'}")
        checked_out = subprocess.run(
            ["git", "checkout", "--quiet", "--detach", str(baseline["commit"])],
            cwd=checkout,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if checked_out.returncode != 0:
            detail = (checked_out.stderr or checked_out.stdout).strip()
            raise RunnerBlockedError(
                f"cannot check out the D0 baseline source commit: {detail or 'git checkout failed'}"
            )
        if _git(checkout, "rev-parse", "HEAD^{tree}") != baseline["tree"]:
            raise RunnerBlockedError("materialized D0 baseline source tree does not match the bundle manifest")
        if _git(checkout, "status", "--porcelain=v1", "--untracked-files=all"):
            raise RunnerBlockedError("materialized D0 baseline source worktree is not clean")
        persistent = Path(tempfile.mkdtemp(prefix="fincore-0042-r2-baseline-source-ready-"))
        moved = persistent / "source"
        checkout.rename(moved)
        return moved


def _cleanup_materialized_source(source_root: Path) -> None:
    """Best-effort cleanup of runner-owned scratch after an architecture command."""

    resolved_source = source_root.resolve()
    parent = resolved_source.parent
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if (
        resolved_source.name != "source"
        or not parent.name.startswith("fincore-0042-r2-baseline-source-ready-")
        or parent.parent != temporary_root
    ):
        raise RunnerBlockedError("refusing to remove a baseline source path that is not runner-owned")
    shutil.rmtree(parent)


def _legacy_source_probe(candidate_root: Path) -> dict:
    probe = (
        "import importlib.util,json,sys,sysconfig; from pathlib import Path; "
        "root=Path(sys.argv[1]).resolve(); sys.path[:0]=[str(root),sysconfig.get_paths()['purelib']]; "
        "import fincore; modules=json.loads(sys.argv[2]); exports=json.loads(sys.argv[3]); "
        "print(json.dumps({'modules':[name for name in modules if importlib.util.find_spec(name) is not None],"
        "'exports':[name for name in exports if hasattr(fincore,name)]},sort_keys=True))"
    )
    command = [
        sys.executable,
        "-S",
        "-E",
        "-c",
        probe,
        str(candidate_root),
        json.dumps(_LEGACY_MODULES),
        json.dumps(_LEGACY_ROOT_EXPORTS),
    ]
    completed = subprocess.run(
        command,
        cwd=candidate_root,
        capture_output=True,
        text=True,
        env=_candidate_environment(candidate_root),
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RunnerBlockedError(
            f"legacy source probe could not run: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RunnerBlockedError("legacy source probe did not emit JSON") from exc
    if not isinstance(result, dict):
        raise RunnerBlockedError("legacy source probe emitted an invalid payload")
    return {
        "argv": command,
        "exports": result.get("exports"),
        "modules": result.get("modules"),
        "output_sha256": _sha256_bytes(completed.stdout.encode("utf-8")),
        "verdict": "PASS" if result.get("modules") == [] and result.get("exports") == [] else "FAIL",
    }


def _artifact_json(bundle: dict, name: str) -> dict:
    return _read_json(bundle["artifacts"][name], f"D0 {name} artifact")


def _architecture_validation_passed(validation: dict) -> bool:
    """Match the exact status field emitted by the frozen architecture checker."""

    return validation.get("status") == "passed"


def _run_architecture_gate(
    args: argparse.Namespace,
    bundle: dict,
    candidate_root: Path,
    output_dir: Path,
) -> dict:
    if args.require_loc_reduction is None or args.require_loc_reduction < 0.12:
        raise RunnerBlockedError("architecture requires --require-loc-reduction of at least 0.12")
    if args.require_duplicate_reduction is None or args.require_duplicate_reduction < 0.60:
        raise RunnerBlockedError("architecture requires --require-duplicate-reduction of at least 0.60")
    if not args.require_no_cycles:
        raise RunnerBlockedError("architecture requires --require-no-cycles")
    if not args.require_legacy_zero:
        raise RunnerBlockedError("architecture requires --require-legacy-zero")

    baseline = _artifact_json(bundle, "architecture_baseline")
    if (
        baseline.get("artifact_type") != "fincore_0042_r2_architecture_measurement"
        or baseline.get("baseline_state") != "frozen"
        or baseline.get("verdict") != "architecture_baseline"
    ):
        raise RunnerBlockedError("D0 architecture artifact is not a frozen architecture baseline")
    provenance = _require_mapping(baseline.get("source_provenance"), "D0 architecture source provenance")
    for key in ("commit", "tree"):
        if provenance.get(key) != bundle["baseline_source"].get(key):
            raise RunnerBlockedError("D0 architecture source provenance does not match the provisioned baseline source")

    capture = output_dir / "candidate-architecture.json"
    if capture.exists():
        raise RunnerUsageError(f"architecture capture path already exists: {capture}")
    baseline_root = _materialize_baseline_source(bundle)
    try:
        script = _frozen_tool(bundle, "scripts/check_architecture_convergence.py")
        record = _run_frozen_command(
            command=[
                sys.executable,
                "-I",
                str(script),
                "--source-root",
                str(candidate_root),
                "--package",
                "fincore",
                "--capture",
                str(capture),
                "--baseline",
                str(bundle["artifacts"]["architecture_baseline"]),
                "--baseline-source-root",
                str(baseline_root),
                "--require-no-cycles",
            ],
            candidate_root=candidate_root,
            timeout_seconds=300,
        )
    finally:
        _cleanup_materialized_source(baseline_root)
    if record["exit_code"] != 0:
        return {"command": record, "verdict": "FAIL"}
    candidate_artifact = _read_json(capture, "candidate architecture capture")
    validation = _require_mapping(
        candidate_artifact.get("baseline_validation"), "candidate architecture baseline_validation"
    )
    if not _architecture_validation_passed(validation):
        return {"command": record, "validation": validation, "verdict": "FAIL"}
    legacy = _legacy_source_probe(candidate_root)
    if legacy["verdict"] != "PASS":
        return {"command": record, "legacy_source": legacy, "validation": validation, "verdict": "FAIL"}
    return {
        "candidate_capture_sha256": _sha256_file(capture),
        "command": record,
        "legacy_source": legacy,
        "summary": candidate_artifact.get("measurements", {}).get("summary"),
        "validation": validation,
        "verdict": "PASS",
    }


def _validate_matrix_cell(
    payload: object,
    *,
    candidate: dict[str, str],
    bundle: dict,
    tooling_identity: dict[str, str],
    wheel_sha256: str,
) -> tuple[str, str]:
    cell = _require_mapping(payload, "matrix cell evidence")
    if set(cell) != _MATRIX_CELL_FIELDS:
        raise RunnerBlockedError("matrix cell evidence does not match the frozen field set")
    if cell.get("matrix_contract_version") != 1:
        raise RunnerBlockedError("matrix cell evidence has an unsupported contract version")
    for field in ("candidate_commit", "candidate_tree"):
        _require_git_object(cell.get(field), f"matrix cell {field}")
    for field in ("wheel_sha256", "d0_tooling_digest", "d0_bundle_digest", "argv_digest", "output_digest"):
        _require_sha256(cell.get(field), f"matrix cell {field}")
    if cell.get("candidate_commit") != candidate["commit"] or cell.get("candidate_tree") != candidate["tree"]:
        raise RunnerBlockedError("matrix cell candidate identity does not match the aggregate candidate")
    if cell.get("wheel_sha256") != wheel_sha256:
        raise RunnerBlockedError("matrix cell wheel digest does not match the aggregate wheel")
    if cell.get("d0_bundle_digest") != bundle["manifest_sha256"]:
        raise RunnerBlockedError("matrix cell D0 bundle digest does not match the aggregate bundle")
    tooling_digest = _canonical_sha256(bundle["manifest"]["tooling"])
    if cell.get("d0_tooling_digest") != tooling_digest:
        raise RunnerBlockedError("matrix cell D0 tooling digest does not match the aggregate tooling contract")
    if cell.get("verdict") != "PASS":
        raise RunnerBlockedError("matrix cell does not report PASS")
    operating_system = cell.get("os")
    python_version = cell.get("python_full_version")
    if operating_system not in {"linux", "macos", "windows"} or not isinstance(python_version, str):
        raise RunnerBlockedError("matrix cell has an unsupported operating system or Python version")
    if not _PYTHON_FULL_VERSION.fullmatch(python_version):
        raise RunnerBlockedError("matrix cell Python version must be a full version")
    if cell.get("dependency_lane") not in {"minimum", "latest", "pinned"}:
        raise RunnerBlockedError("matrix cell has an unsupported dependency lane")
    if not isinstance(cell.get("dependency_profile"), str) or not cell["dependency_profile"].strip():
        raise RunnerBlockedError("matrix cell dependency profile is missing")
    if not isinstance(cell.get("runner_image"), str) or not cell["runner_image"].strip():
        raise RunnerBlockedError("matrix cell runner image is missing")
    if not isinstance(cell.get("evidence_time"), str) or not cell["evidence_time"].strip():
        raise RunnerBlockedError("matrix cell evidence time is missing")
    return operating_system, python_version


def _run_matrix_aggregate_gate(
    args: argparse.Namespace,
    bundle: dict,
    candidate: dict[str, str],
    tooling_identity: dict[str, str],
) -> dict:
    matrix_directory = _require_external_directory(
        args.matrix_evidence_dir,
        "--matrix-evidence-dir",
        Path(candidate["root"]),
        _tooling_root(),
    )
    requested_os = tuple(args.require_os)
    if set(requested_os) != {"linux", "macos", "windows"} or len(requested_os) != 3:
        raise RunnerBlockedError("matrix-aggregate requires exactly --require-os linux macos windows")
    if not args.require_support_window_from_bundle:
        raise RunnerBlockedError("matrix-aggregate requires --require-support-window-from-bundle")
    wheel = _require_existing_path(args.candidate_wheel, "--candidate-wheel")
    if not wheel.is_file():
        raise RunnerUsageError("--candidate-wheel must identify a regular file")
    wheel_sha256 = _sha256_file(wheel)
    cells: dict[tuple[str, str], Path] = {}
    for path in sorted(matrix_directory.rglob("matrix-cell.json")):
        key = _validate_matrix_cell(
            _read_json(path, "matrix cell evidence"),
            candidate=candidate,
            bundle=bundle,
            tooling_identity=tooling_identity,
            wheel_sha256=wheel_sha256,
        )
        if key in cells:
            raise RunnerBlockedError(f"matrix evidence contains duplicate cell {key[0]}/{key[1]}")
        cells[key] = path
    expected = {
        (operating_system, version) for operating_system in requested_os for version in bundle["python_support_window"]
    }
    missing = sorted(expected - set(cells))
    unexpected = sorted(set(cells) - expected)
    if missing or unexpected:
        fragments: list[str] = []
        if missing:
            fragments.append("missing " + ", ".join(f"{os_name}/{version}" for os_name, version in missing))
        if unexpected:
            fragments.append("unexpected " + ", ".join(f"{os_name}/{version}" for os_name, version in unexpected))
        raise RunnerBlockedError("matrix evidence is incomplete: " + "; ".join(fragments))
    return {
        "cells": {
            f"{operating_system}/{version}": str(path) for (operating_system, version), path in sorted(cells.items())
        },
        "expected_cells": sorted(f"{operating_system}/{version}" for operating_system, version in expected),
        "wheel_sha256": wheel_sha256,
        "verdict": "PASS",
    }


def _run_final_gate(
    args: argparse.Namespace,
    bundle: dict,
    candidate: dict[str, str],
    tooling_identity: dict[str, str],
    manifest: dict,
) -> dict:
    evidence_directory = _require_external_directory(
        args.evidence_dir,
        "--evidence-dir",
        Path(candidate["root"]),
        _tooling_root(),
    )
    blocked: list[str] = []
    failed: list[str] = []
    verified: list[str] = []
    for gate in manifest["final_requires_gates"]:
        evidence_path = evidence_directory / gate / "evidence.json"
        if not evidence_path.is_file():
            blocked.append(f"{gate}: evidence is missing")
            continue
        evidence = _read_json(evidence_path, f"{gate} evidence")
        if evidence.get("artifact_type") != "run_0042_r2_acceptance_evidence":
            blocked.append(f"{gate}: unexpected evidence artifact type")
            continue
        if evidence.get("d0_bundle_sha256") != bundle["manifest_sha256"]:
            blocked.append(f"{gate}: D0 bundle digest mismatch")
            continue
        if evidence.get("candidate") != candidate:
            blocked.append(f"{gate}: candidate identity mismatch")
            continue
        if evidence.get("runner") != tooling_identity:
            blocked.append(f"{gate}: runner identity mismatch")
            continue
        verdict = evidence.get("verdict")
        if verdict == "PASS":
            verified.append(gate)
        elif verdict == "FAIL":
            failed.append(gate)
        else:
            blocked.append(f"{gate}: verdict {verdict!r}")
    verdict = "FAIL" if failed else "BLOCKED" if blocked else "PASS"
    return {"blocked": blocked, "failed": failed, "verified": verified, "verdict": verdict}


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


def _run_bundle_gate(args: argparse.Namespace, gate: str, manifest: dict, tooling_identity: dict[str, str]) -> dict:
    gate_spec = manifest["gates"][gate]
    candidate_root = _require_existing_path(args.candidate_root, "--candidate-root")
    bundle = _require_existing_path(args.expected_bundle, "--expected-bundle")
    _require_bundle_outside_candidate(bundle, candidate_root)
    if not gate_spec.get("consumes_d0_bundle"):
        raise RunnerUsageError(f"gate {gate} does not consume a D0 bundle")
    bundle_data = _load_d0_bundle(bundle, tooling_identity)
    _validate_frozen_tooling(bundle_data, tooling_identity)
    candidate = _candidate_identity(candidate_root, args.candidate_head)
    if gate == "tests":
        return _run_tests_gate(args, candidate_root)
    if gate == "static":
        output_dir = _require_external_directory(args.output_dir, "--output-dir", candidate_root, _tooling_root())
        return _run_static_gate(candidate_root, output_dir)
    if gate == "package":
        return _run_package_gate(args, bundle_data, candidate_root)
    if gate == "quality":
        output_dir = _require_external_directory(args.output_dir, "--output-dir", candidate_root, _tooling_root())
        return _run_quality_gate(args, bundle_data, candidate_root, output_dir)
    if gate == "parity":
        output_dir = _require_external_directory(args.output_dir, "--output-dir", candidate_root, _tooling_root())
        return _run_parity_gate(args, bundle_data, candidate_root, output_dir)
    if gate == "performance":
        output_dir = _require_external_directory(args.output_dir, "--output-dir", candidate_root, _tooling_root())
        return _run_performance_gate(args, bundle_data, candidate_root, output_dir)
    if gate == "report":
        output_dir = _require_external_directory(args.output_dir, "--output-dir", candidate_root, _tooling_root())
        return _run_report_gate(args, candidate_root, output_dir)
    if gate == "installed":
        return _run_installed_gate(args, bundle_data, candidate_root)
    if gate == "matrix-cell":
        output_dir = _require_external_directory(args.output_dir, "--output-dir", candidate_root, _tooling_root())
        return _run_matrix_cell_gate(args, bundle_data, candidate, tooling_identity, output_dir)
    if gate == "architecture":
        output_dir = _require_external_directory(args.output_dir, "--output-dir", candidate_root, _tooling_root())
        return _run_architecture_gate(args, bundle_data, candidate_root, output_dir)
    if gate == "matrix-aggregate":
        return _run_matrix_aggregate_gate(args, bundle_data, candidate, tooling_identity)
    if gate == "final":
        return _run_final_gate(args, bundle_data, candidate, tooling_identity, manifest)
    raise RunnerBlockedError(
        f"gate {gate} has a validated D0 bundle but its detached execution contract is not implemented yet"
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
    parser.add_argument("--os", dest="matrix_os")
    parser.add_argument("--python-full-version")
    parser.add_argument("--dependency-lane")
    parser.add_argument("--dependency-profile")
    parser.add_argument("--runner-image")
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
            details = _run_bundle_gate(args, gate, manifest, identity)
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
        "candidate": None,
        "candidate_head": args.candidate_head,
        "candidate_root": args.candidate_root,
        "d0_bundle_sha256": None,
        "details": details,
        "gate": args.gate,
        "recorded_at": _utc_now(),
        "reasons": reasons,
        "runner": identity,
        "schema_version": 1,
        "verdict": verdict,
    }
    if args.candidate_root:
        try:
            candidate_root = Path(args.candidate_root).expanduser().resolve()
            if candidate_root.is_dir():
                evidence["candidate"] = _candidate_identity(candidate_root, args.candidate_head)
        except (RunnerBlockedError, RunnerUsageError):
            pass
    if args.expected_bundle:
        try:
            bundle_root = Path(args.expected_bundle).expanduser().resolve()
            if bundle_root.is_dir():
                evidence["d0_bundle_sha256"] = _sha256_file(bundle_root / _D0_BUNDLE_MANIFEST_NAME)
        except OSError:
            pass
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
