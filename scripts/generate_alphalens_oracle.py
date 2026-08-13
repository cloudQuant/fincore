#!/usr/bin/env python
"""Execute a pinned Alphalens tuple in an isolated temporary Conda prefix.

The checked-in metadata supplies an explicit Conda package lock and a fully
hashed pip lock. This command recreates that tuple in a temporary directory,
clones a clean detached checkout of the pinned source, validates every source
blob and runtime fingerprint, and writes an *unreviewed* candidate result. It
never installs into or otherwise mutates the user's base environment.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import locale
import os
import platform
import re
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from copy import deepcopy
from importlib import metadata
from pathlib import Path
from typing import Any

ALPHALENS_PROFILE = "cloudquant-local-3fa17ad"
GIT_TIMEOUT_SECONDS = 30
CONDA_TIMEOUT_SECONDS = 600
PIP_TIMEOUT_SECONDS = 600
ORACLE_TIMEOUT_SECONDS = 180
PROCESS_TERMINATION_GRACE_SECONDS = 2
NONINTERACTIVE_ENV_OVERRIDES = {"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""}
PREFIX_ENVIRONMENT_KEYS = (
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "CONDA_PROMPT_MODIFIER",
    "CONDA_PYTHON_EXE",
    "CONDA_SHLVL",
    "PYTHONHOME",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "__PYVENV_LAUNCHER__",
)
EXPLICIT_HASH = re.compile(r"https://[^\s]+#[0-9a-fA-F]{32}$")
TABLE_FIELDS = {
    "kind",
    "index",
    "index_names",
    "timezone",
    "columns",
    "dtypes",
    "values",
    "nan_mask",
}

ORACLE_WORKER = r"""
import contextlib
import hashlib
import importlib.metadata
import io
import json
import locale
import os
from pathlib import Path
import platform
import site
import sys
import sysconfig
import time

import numpy as np
import pandas as pd


def scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def datetime_index(values, timezone):
    if timezone is None:
        return pd.DatetimeIndex(pd.to_datetime(values))
    parsed = [pd.Timestamp(value) for value in values]
    if any(value.tzinfo is not None for value in parsed):
        return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).tz_convert(timezone)
    return pd.DatetimeIndex(pd.to_datetime(values)).tz_localize(timezone)


def table_index(table):
    raw_index = table["index"]
    names = table["index_names"]
    if raw_index and isinstance(raw_index[0], list):
        first_level = datetime_index([row[0] for row in raw_index], table["timezone"])
        remaining_levels = [[row[position] for row in raw_index] for position in range(1, len(raw_index[0]))]
        return pd.MultiIndex.from_arrays([first_level, *remaining_levels], names=names)
    return datetime_index(raw_index, table["timezone"]).rename(names[0])


def frame(table):
    values = [
        [np.nan if missing else value for value, missing in zip(row, mask_row, strict=True)]
        for row, mask_row in zip(table["values"], table["nan_mask"], strict=True)
    ]
    result = pd.DataFrame(values, columns=table["columns"], index=table_index(table))
    for name, dtype in table["dtypes"].items():
        result[name] = result[name].astype(dtype)
    return result


def series(table):
    result = frame(table).iloc[:, 0]
    result.name = table["columns"][0]
    return result


def serialized_table(value, kind):
    if isinstance(value, pd.Series):
        column = value.name if value.name is not None else "value"
        frame_value = value.to_frame(name=column)
        kind = "series"
    else:
        frame_value = value
    index = frame_value.index
    if isinstance(index, pd.MultiIndex):
        serialized_index = [[scalar(part) for part in item] for item in index.tolist()]
    else:
        serialized_index = [scalar(item) for item in index.tolist()]
    values = []
    nan_mask = []
    for row in frame_value.to_numpy(dtype=object):
        values.append([scalar(value) for value in row])
        nan_mask.append([
            bool(pd.isna(value)) if isinstance(pd.isna(value), (bool, np.bool_)) else False
            for value in row
        ])
    index_timezone = (
        getattr(index.levels[0], "tz", None) if isinstance(index, pd.MultiIndex) else getattr(index, "tz", None)
    )
    return {
        "kind": kind,
        "index": serialized_index,
        "index_names": list(index.names),
        "timezone": str(index_timezone) if index_timezone is not None else None,
        "columns": [str(column) for column in frame_value.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame_value.dtypes.items()},
        "values": values,
        "nan_mask": nan_mask,
    }


def serialize(value):
    if isinstance(value, pd.Series):
        return serialized_table(value, "series")
    if isinstance(value, pd.DataFrame):
        return serialized_table(value, "dataframe")
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [serialize(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [serialize(item) for item in value]}
    if isinstance(value, dict):
        return {"kind": "mapping", "items": {str(key): serialize(item) for key, item in sorted(value.items())}}
    return scalar(value)


def capture(call):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            value = call()
    except Exception as error:
        return {
            "status": "error",
            "error_type": type(error).__name__,
            "message": str(error),
        }
    return {"status": "ok", "value": serialize(value)}


def run_case(case, utils, performance):
    category = case["category"]
    tables = case["tables"]
    parameters = case["parameters"]
    if category in {"daily", "business-day", "intraday", "tz-aware"}:
        factor = series(tables["factor"])
        prices = frame(tables["prices"])
        return capture(lambda: utils.get_clean_factor_and_forward_returns(factor, prices, **parameters))
    if category == "max-loss-boundary":
        factor = series(tables["factor"])
        prices = frame(tables["prices"])
        return {
            "kind": "max-loss-boundary",
            "results": {
                str(max_loss): capture(
                    lambda max_loss=max_loss: utils.get_clean_factor_and_forward_returns(
                        factor, prices, periods=(1,), max_loss=max_loss
                    )
                )
                for max_loss in parameters["max_loss"]
            },
        }
    factor_data = frame(tables["factor_data"])
    if category == "ties-nan-zero":
        return capture(lambda: utils.quantize_factor(factor_data, **parameters))
    if category == "bins-quantiles":
        return {
            "kind": "bins-and-quantiles",
            "quantile": capture(lambda: utils.quantize_factor(factor_data, **parameters["quantile_call"])),
            "bins": capture(lambda: utils.quantize_factor(factor_data, **parameters["bin_call"])),
        }
    if category == "group-neutral":
        return capture(lambda: performance.factor_weights(factor_data, **parameters))
    if category == "pre-cleaned-performance":
        returns = performance.factor_returns(
            factor_data,
            demeaned=parameters["long_short"],
            group_adjust=parameters["group_neutral"],
        )
        return {
            "kind": "pre-cleaned-performance",
            "information_coefficient": capture(lambda: performance.factor_information_coefficient(factor_data)),
            "factor_returns": {"status": "ok", "value": serialize(returns)},
            "alpha_beta": capture(lambda: performance.factor_alpha_beta(factor_data, returns=returns)),
            "mean_return_by_quantile": capture(lambda: performance.mean_return_by_quantile(factor_data)),
            "turnover": capture(
                lambda: performance.quantile_turnover(factor_data["factor_quantile"], quantile=2, period=1)
            ),
        }
    if category == "event-window":
        returns = frame(tables["returns"])
        return capture(lambda: performance.average_cumulative_return_by_quantile(factor_data, returns, **parameters))
    if category == "pyfolio-input":
        return capture(
            lambda: performance.create_pyfolio_input(
                factor_data,
                period=parameters["period"],
                capital=parameters["capital"],
                long_short=parameters["long_short"],
                group_neutral=parameters["group_neutral"],
            )
        )
    raise RuntimeError(f"unsupported deterministic oracle case category: {category}")


def runtime_raw():
    distributions = {}
    for distribution in importlib.metadata.distributions(path=site.getsitepackages()):
        name = distribution.metadata.get("Name")
        if name:
            distributions[name] = distribution.version
    blas_details = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
    blas = {
        "configuration": blas_details.get("openblas configuration"),
        "found": blas_details.get("found"),
        "name": blas_details.get("name"),
        "version": blas_details.get("version"),
    }
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "soabi": sysconfig.get_config_var("SOABI"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "byteorder": sys.byteorder,
        },
        "locale": locale.setlocale(locale.LC_ALL, None),
        "timezone": {"TZ": os.environ.get("TZ"), "tzname": list(time.tzname)},
        "blas": blas,
        "distributions": dict(sorted(distributions.items())),
    }


checkout = Path(sys.argv[1]).resolve()
cases_path = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(checkout))
import alphalens
from alphalens import performance, utils

module_path = Path(alphalens.__file__).resolve()
try:
    module_path.relative_to(checkout)
except ValueError as error:
    raise RuntimeError(f"oracle imported Alphalens outside isolated checkout: {module_path}") from error

cases = json.loads(cases_path.read_text(encoding="utf-8"))
payload = {
    "runtime_raw": runtime_raw(),
    "import_path": module_path.relative_to(checkout).as_posix(),
    "execution_context": {
        "cwd": Path.cwd().as_posix(),
        "isolated": bool(sys.flags.isolated),
        "prefix": Path(sys.prefix).resolve().as_posix(),
    },
    "case_results": [
        {"case_id": case["case_id"], "category": case["category"], "result": run_case(case, utils, performance)}
        for case in cases["cases"]
    ],
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""


def _error(message: str) -> ValueError:
    return ValueError(f"Alphalens oracle validation failed: {message}")


def _run_git(root: Path, arguments: list[str], operation: str) -> bytes:
    """Use only bounded, noninteractive Git commands against the supplied source."""
    result = _run_process(
        ["git", *arguments],
        operation,
        cwd=root,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    assert isinstance(result.stdout, bytes)
    return result.stdout


def _run_process(
    command: list[str],
    operation: str,
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    unset_environment_keys: tuple[str, ...] = (),
    timeout: int,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    """Run one bounded subprocess in its own session and reap it on timeout."""
    child_environment = os.environ.copy()
    for key in unset_environment_keys:
        child_environment.pop(key, None)
    child_environment.update(NONINTERACTIVE_ENV_OVERRIDES)
    if environment is not None:
        child_environment.update(environment)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=child_environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        raise _error(f"{operation} timed out after {timeout}s") from exc
    if process.returncode != 0:
        detail = stderr or stdout or ""
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        raise _error(f"{operation} failed: {detail.strip() or process.returncode}")
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    """Terminate only the session created for this process, then reap its leader."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.communicate(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _error(f"cannot read {label} JSON {path}") from exc
    if not isinstance(payload, dict):
        raise _error(f"{label} JSON must be an object")
    return payload


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, label: str) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise _error(f"cannot read {label} {path}") from exc


def _json_digest(payload: dict[str, Any], *, without_oracle_verification: bool = False) -> str:
    """Return a stable digest, excluding a circular review record when requested."""
    normalized = deepcopy(payload)
    if without_oracle_verification:
        normalized.pop("oracle_verification", None)
    return _sha256_bytes(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _error(message)


def _source_blob_records(root: Path, commit: str, source_files: list[dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for item in source_files:
        path = item.get("path")
        expected_blob = item.get("git_blob")
        expected_sha = item.get("sha256")
        _require(isinstance(path, str) and path and not Path(path).is_absolute(), "source file path is not portable")
        _require(isinstance(expected_blob, str) and isinstance(expected_sha, str), "source evidence lacks hashes")
        blob = _run_git(root, ["rev-parse", f"{commit}:{path}"], f"resolve pinned blob {path}").decode().strip()
        contents = _run_git(root, ["show", f"{commit}:{path}"], f"read pinned blob {path}")
        actual_sha = _sha256_bytes(contents)
        _require(blob == expected_blob, f"pinned blob mismatch for {path}")
        _require(actual_sha == expected_sha, f"pinned SHA256 mismatch for {path}")
        records.append({"path": path, "git_blob": blob, "sha256": actual_sha})
    return records


def _validate_source(root: Path, commit: str, environment: dict[str, Any]) -> list[dict[str, str]]:
    _require(root.is_dir(), f"source is not a directory: {root}")
    expected_source = environment.get("source")
    _require(isinstance(expected_source, dict), "environment has no source section")
    _require(expected_source.get("commit") == commit, "environment source commit does not match --commit")
    _run_git(root, ["cat-file", "-e", f"{commit}^{{commit}}"], "validate requested pinned commit")
    actual_head = _run_git(root, ["rev-parse", "HEAD"], "resolve source HEAD").decode().strip()
    _require(actual_head == commit, f"source HEAD is {actual_head}, expected {commit}")
    if expected_source.get("worktree_required_clean") is True:
        status = _run_git(root, ["status", "--porcelain=v1", "--untracked-files=all"], "check source dirty state")
        _require(not status.strip(), "source checkout is dirty")
    files = expected_source.get("source_files")
    _require(isinstance(files, list) and files, "environment source_files is missing")
    return _source_blob_records(root, commit, files)


def _validate_environment_static(
    environment: dict[str, Any], explicit_lock: Path, requirements: Path, commit: str
) -> None:
    _require(environment.get("profile") == ALPHALENS_PROFILE, "environment profile is not the pinned profile")
    _require(environment.get("source", {}).get("commit") == commit, "environment source commit is wrong")
    lock = environment.get("explicit_lock")
    requirement_lock = environment.get("requirements")
    _require(isinstance(lock, dict), "environment explicit_lock section is missing")
    _require(isinstance(requirement_lock, dict), "environment requirements section is missing")
    _require(lock.get("sha256") == _sha256_file(explicit_lock, "explicit lock"), "explicit lock SHA256 mismatch")
    _require(
        requirement_lock.get("sha256") == _sha256_file(requirements, "requirements lock"),
        "requirements lock SHA256 mismatch",
    )


def _validate_explicit_lock(path: Path, environment: dict[str, Any]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    _require("@EXPLICIT" in lines, "explicit lock lacks @EXPLICIT")
    package_lines = [line for line in lines if line and not line.startswith("#") and line != "@EXPLICIT"]
    _require(package_lines, "explicit lock contains no package URLs")
    _require(all(EXPLICIT_HASH.fullmatch(line) for line in package_lines), "explicit lock has unhashed package URL")
    metadata_lock = environment["explicit_lock"]
    _require(metadata_lock.get("package_url_count") == len(package_lines), "explicit lock package count mismatch")
    _require(
        metadata_lock.get("status") == "complete-executable-conda-packages",
        "explicit lock is not marked as an executable complete Conda lock",
    )


def _validate_requirements_lock(path: Path, environment: dict[str, Any]) -> None:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    active = [line for line in lines if line and not line.startswith("#")]
    _require(active, "requirements lock contains no reviewed pip requirements")
    _require("--require-hashes" in active, "requirements lock does not enable --require-hashes")
    requirements = [line for line in active if not line.startswith("--")]
    _require(requirements, "requirements lock contains no pip requirement")
    _require(
        all(" --hash=sha256:" in line for line in requirements),
        "requirements lock has an unhashed pip requirement",
    )
    _require(
        environment["requirements"].get("status") == "complete-executable-pip-hash-lock",
        "requirements lock is not marked as executable with wheel hashes",
    )
    expected_count = environment["requirements"].get("package_count")
    _require(expected_count == len(requirements), "requirements lock package count mismatch")


def _conda_executable() -> Path:
    configured = os.environ.get("CONDA_EXE")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path(sys.prefix) / "bin" / "conda")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise _error("cannot locate the controlling Conda executable")


def _prefix_python(prefix: Path) -> Path:
    interpreter = prefix / "bin" / "python"
    _require(interpreter.is_file(), f"isolated prefix has no Python interpreter: {prefix}")
    return interpreter


def _normalized_conda_inventory(prefix: Path) -> list[dict[str, str | int]]:
    result = _run_process(
        [_conda_executable().as_posix(), "list", "--json", "--prefix", prefix.as_posix()],
        "read isolated prefix package inventory",
        timeout=GIT_TIMEOUT_SECONDS,
        text=True,
    )
    try:
        raw_records = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise _error("isolated prefix package inventory is not JSON") from exc
    _require(isinstance(raw_records, list), "isolated prefix package inventory is not a list")
    records: list[dict[str, str | int]] = []
    for record in raw_records:
        _require(isinstance(record, dict), "isolated prefix package inventory has a malformed record")
        name = record.get("name")
        version = record.get("version")
        build = record.get("build_string")
        channel = record.get("channel")
        platform_name = record.get("platform")
        _require(
            all(isinstance(value, str) for value in (name, version, build, channel, platform_name)),
            "isolated prefix package inventory lacks an exact name/version/build/channel/platform",
        )
        records.append(
            {
                "name": name,
                "version": version,
                "build": build,
                "channel": channel,
                "platform": platform_name,
            }
        )
    return sorted(records, key=lambda record: (str(record["name"]).lower(), str(record["version"])))


def _validate_distribution_inventory(environment: dict[str, Any], prefix: Path) -> list[dict[str, str | int]]:
    inventory = environment.get("distribution_inventory")
    _require(isinstance(inventory, dict), "environment has no full distribution inventory")
    expected = inventory.get("records")
    _require(isinstance(expected, list) and expected, "environment full distribution inventory is missing")
    observed = _normalized_conda_inventory(prefix)
    _require(observed == expected, "isolated prefix full distribution inventory mismatch")
    _require(inventory.get("sha256") == _json_digest({"records": observed}), "distribution inventory SHA256 mismatch")
    return observed


def _create_isolated_prefix(explicit_lock: Path, temporary_root: Path) -> Path:
    prefix = temporary_root / "prefix"
    _run_process(
        [
            _conda_executable().as_posix(),
            "create",
            "--yes",
            "--prefix",
            prefix.as_posix(),
            "--file",
            explicit_lock.as_posix(),
        ],
        "create isolated prefix from the explicit lock",
        timeout=CONDA_TIMEOUT_SECONDS,
    )
    _prefix_python(prefix)
    return prefix


def _install_hashed_requirements(prefix: Path, requirements: Path) -> None:
    _run_process(
        [
            _prefix_python(prefix).as_posix(),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--disable-pip-version-check",
            "--no-input",
            "--no-cache-dir",
            "--require-hashes",
            "--only-binary=:all:",
            "-r",
            requirements.as_posix(),
        ],
        "install isolated hashed pip requirements",
        unset_environment_keys=PREFIX_ENVIRONMENT_KEYS,
        timeout=PIP_TIMEOUT_SECONDS,
    )
    _run_process(
        [_prefix_python(prefix).as_posix(), "-m", "pip", "check"],
        "validate isolated pip requirements",
        unset_environment_keys=PREFIX_ENVIRONMENT_KEYS,
        timeout=GIT_TIMEOUT_SECONDS,
    )


def _clone_clean_checkout(source: Path, commit: str, temporary_root: Path) -> Path:
    checkout = temporary_root / "checkout"
    _run_process(
        ["git", "clone", "--quiet", "--no-checkout", source.as_posix(), checkout.as_posix()],
        "clone isolated Alphalens source checkout",
        timeout=GIT_TIMEOUT_SECONDS,
    )
    _run_process(
        ["git", "checkout", "--quiet", "--detach", commit],
        "checkout isolated pinned Alphalens source",
        cwd=checkout,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    _require(
        _run_git(checkout, ["rev-parse", "HEAD"], "resolve isolated source HEAD").decode().strip() == commit,
        "isolated source checkout commit mismatch",
    )
    _require(
        not _run_git(
            checkout, ["status", "--porcelain=v1", "--untracked-files=all"], "check isolated source state"
        ).strip(),
        "isolated source checkout is dirty",
    )
    return checkout


def _execute_oracle_worker(prefix: Path, checkout: Path, cases: Path) -> dict[str, Any]:
    cases_path = cases.resolve()
    result = _run_process(
        [_prefix_python(prefix).as_posix(), "-I", "-c", ORACLE_WORKER, checkout.as_posix(), cases_path.as_posix()],
        "execute isolated deterministic Alphalens cases",
        cwd=checkout,
        environment={"MPLBACKEND": "Agg", "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""},
        unset_environment_keys=PREFIX_ENVIRONMENT_KEYS,
        timeout=ORACLE_TIMEOUT_SECONDS,
        text=True,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise _error("isolated oracle worker did not emit JSON") from exc
    _require(isinstance(payload, dict), "isolated oracle worker payload is not an object")
    _require(
        payload.get("import_path") == "alphalens/__init__.py", "isolated oracle imported an unexpected module path"
    )
    _require(isinstance(payload.get("runtime_raw"), dict), "isolated oracle omitted its raw runtime fingerprint")
    _require(isinstance(payload.get("case_results"), list) and payload["case_results"], "isolated oracle omitted cases")
    return payload


def _validate_isolated_runtime(environment: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    expected = environment.get("runtime")
    _require(isinstance(expected, dict), "environment runtime fingerprint is missing")
    normalized = _normalize_runtime_fingerprint(raw)
    _require(raw == expected.get("raw"), "raw execution fingerprint mismatch")
    _require(normalized == expected.get("normalized"), "normalized execution fingerprint mismatch")
    return normalized


def _runtime_observation(environment: dict[str, Any]) -> dict[str, Any]:
    """Collect raw runtime facts without importing Alphalens."""
    important = environment.get("distribution_inventory", {}).get("important_distributions", [])
    versions: dict[str, str | None] = {}
    for package in important:
        name = package.get("name")
        if not isinstance(name, str):
            raise _error("important distribution has no name")
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "soabi": sysconfig.get_config_var("SOABI"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "byteorder": sys.byteorder,
        },
        "locale": locale.setlocale(locale.LC_ALL, None),
        "timezone": {"TZ": os.environ.get("TZ"), "tzname": list(time.tzname)},
        "blas": environment.get("runtime", {}).get("raw", {}).get("blas"),
        "distributions": versions,
    }


def _normalize_runtime_fingerprint(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize platform identity while retaining the host's raw system name.

    Python reports Darwin; the human-facing, portable profile calls that macOS.
    Keeping both prevents an accidental Darwin/macOS spelling mismatch from
    being mistaken for a different oracle environment.
    """
    platform_fingerprint = raw["platform"]
    raw_system = platform_fingerprint["system"]
    normalized_os = {"Darwin": "macOS"}.get(raw_system, raw_system)
    return {
        "python": deepcopy(raw["python"]),
        "platform": {
            "os": normalized_os,
            "raw_system": raw_system,
            "release": platform_fingerprint["release"],
            "machine": platform_fingerprint["machine"],
            "processor": platform_fingerprint["processor"],
            "byteorder": platform_fingerprint["byteorder"],
        },
        "locale": raw["locale"],
        "timezone": deepcopy(raw["timezone"]),
        "blas": raw["blas"],
    }


def _validate_runtime(environment: dict[str, Any]) -> dict[str, Any]:
    raw = _runtime_observation(environment)
    observed = _normalize_runtime_fingerprint(raw)
    expected = environment.get("runtime")
    _require(isinstance(expected, dict), "environment runtime fingerprint is missing")
    expected_raw = expected.get("raw")
    expected_normalized = expected.get("normalized")
    _require(raw == expected_raw, "raw execution fingerprint mismatch")
    _require(observed == expected_normalized, "normalized OS/architecture fingerprint mismatch")
    return observed


def _is_matrix(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(row, list) for row in value)


def _same_matrix_shape(left: Any, right: Any) -> bool:
    return (
        _is_matrix(left)
        and _is_matrix(right)
        and len(left) == len(right)
        and all(len(left_row) == len(right_row) for left_row, right_row in zip(left, right, strict=True))
    )


def _validate_cases(cases: dict[str, Any], commit: str) -> None:
    _require(cases.get("profile") == ALPHALENS_PROFILE, "case fixture profile is wrong")
    _require(cases.get("commit") == commit, "case fixture commit is wrong")
    _require(cases.get("serializer", {}).get("name") == "fincore-compat-json-table-v1", "case serializer is wrong")
    records = cases.get("cases")
    _require(isinstance(records, list) and records, "case fixture has no cases")
    identifiers = set()
    for case in records:
        _require(isinstance(case, dict), "case record is not an object")
        identifier = case.get("case_id")
        _require(isinstance(identifier, str) and identifier not in identifiers, "case IDs must be unique")
        identifiers.add(identifier)
        _require(case.get("serializer") == "fincore-compat-json-table-v1", f"case {identifier} has wrong serializer")
        _require("expected_output" not in case, f"case {identifier} invents an unreviewed output")
        tables = case.get("tables")
        _require(isinstance(tables, dict) and tables, f"case {identifier} has no tables")
        for table_name, table in tables.items():
            _require(
                isinstance(table, dict) and set(table) >= TABLE_FIELDS,
                f"case {identifier} table {table_name} lacks serializer fields",
            )
            _require(
                _same_matrix_shape(table["values"], table["nan_mask"]),
                f"case {identifier} table {table_name} has invalid NaN mask",
            )
            for row, mask_row in zip(table["values"], table["nan_mask"], strict=True):
                for value, is_nan in zip(row, mask_row, strict=True):
                    _require(isinstance(is_nan, bool), f"case {identifier} table {table_name} NaN mask is not boolean")
                    _require(
                        (value is None) == is_nan, f"case {identifier} table {table_name} has inconsistent NaN encoding"
                    )


def _validate_output_target(output: Path, source: Path, inputs: tuple[Path, ...]) -> Path:
    """Keep candidate writes outside the source checkout and immutable inputs."""
    resolved = output.resolve()
    try:
        resolved.relative_to(source)
    except ValueError:
        pass
    else:
        raise _error("output must not be inside the source checkout")
    _require(resolved not in {path.resolve() for path in inputs}, "output must not replace an oracle input")
    return resolved


def _candidate_payload(
    *,
    commit: str,
    source_files: list[dict[str, str]],
    environment_path: Path,
    explicit_lock: Path,
    requirements: Path,
    cases_path: Path,
    runtime: dict[str, Any],
    worker: dict[str, Any],
) -> dict[str, Any]:
    environment = _read_json(environment_path, "environment")
    case_results = worker["case_results"]
    payload = {
        "schema_version": 1,
        "profile": ALPHALENS_PROFILE,
        "commit": commit,
        "source_files": source_files,
        "environment": {
            "path": environment_path.name,
            "sha256": _sha256_file(environment_path, "environment metadata"),
            "semantic_digest": _json_digest(environment, without_oracle_verification=True),
            "explicit_lock_sha256": _sha256_file(explicit_lock, "explicit lock"),
            "requirements_sha256": _sha256_file(requirements, "requirements lock"),
        },
        "cases": {"path": cases_path.name, "sha256": _sha256_file(cases_path, "case fixture")},
        "runtime": runtime,
        "execution": "isolated-prefix-clean-checkout-deterministic-case-execution",
        "case_results": case_results,
        "case_results_digest": _sha256_bytes(
            json.dumps(case_results, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "reviewed": False,
    }
    payload["candidate_digest"] = _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return payload


def _write_candidate(path: Path, candidate: dict[str, Any]) -> None:
    """Write only the requested candidate path, atomically after every validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_name = temporary.name
            json.dump(candidate, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--explicit-lock", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requirements = args.explicit_lock.with_name("requirements-alphalens-0.4.0-cloudquant.txt")
    output = _validate_output_target(
        args.output,
        args.source.resolve(),
        (args.environment, args.explicit_lock, requirements, args.cases),
    )
    environment = _read_json(args.environment, "environment")
    cases = _read_json(args.cases, "case fixture")
    source_files = _validate_source(args.source.resolve(), args.commit, environment)
    _validate_environment_static(
        environment,
        args.explicit_lock,
        requirements,
        args.commit,
    )
    _validate_cases(cases, args.commit)
    _require(
        environment.get("execution_status") in {"executable-unreviewed-tuple", "reviewed-executable-tuple"},
        "environment metadata is not an executable isolated tuple",
    )
    _validate_explicit_lock(args.explicit_lock, environment)
    _validate_requirements_lock(requirements, environment)
    with tempfile.TemporaryDirectory(prefix="fincore-alphalens-oracle-") as temporary:
        temporary_root = Path(temporary)
        prefix = _create_isolated_prefix(args.explicit_lock, temporary_root)
        _install_hashed_requirements(prefix, requirements)
        _validate_distribution_inventory(environment, prefix)
        checkout = _clone_clean_checkout(args.source.resolve(), args.commit, temporary_root)
        _source_blob_records(checkout, args.commit, environment["source"]["source_files"])
        worker = _execute_oracle_worker(prefix, checkout, args.cases)
        runtime = _validate_isolated_runtime(environment, worker["runtime_raw"])
    _write_candidate(
        output,
        _candidate_payload(
            commit=args.commit,
            source_files=source_files,
            environment_path=args.environment,
            explicit_lock=args.explicit_lock,
            requirements=requirements,
            cases_path=args.cases,
            runtime=runtime,
            worker=worker,
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc
