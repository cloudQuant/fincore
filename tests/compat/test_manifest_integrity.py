from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parents[2]
GENERATOR = ROOT / "scripts" / "generate_compat_manifest.py"
ALPHALENS_ORACLE = ROOT / "scripts" / "generate_alphalens_oracle.py"
ALPHALENS_ROOT = ROOT.parent / "alphalens"
_EXISTING_FIXTURES = (
    "empyrical-0.6.0-api.json",
    "fincore-flat-api-migrations.json",
    "pyfolio-0.9.6-api.json",
    "pyfolio-0.9.6-portfolio-contracts.json",
)
ALPHALENS_COMMIT = "3fa17ad4c3edb025d1410de7aeba9673cba7791c"
ALPHALENS_PROFILE = "cloudquant-local-3fa17ad"
FACTOR_PARTITIONS = {
    "style": ["momentum", "size", "value", "reversal_short_term", "volatility"],
    "sector": [
        "basic_materials",
        "consumer_cyclical",
        "financial_services",
        "real_estate",
        "consumer_defensive",
        "health_care",
        "utilities",
        "communication_services",
        "energy",
        "industrials",
        "technology",
    ],
}


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _sha256_text_file(path: Path) -> str:
    """SHA256 of a text file with canonical LF line endings.

    Recorded digests come from the reference-host checkout; Windows checkouts
    may materialize the same blob with CRLF endings.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location("compat_manifest_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_alphalens_oracle() -> Any:
    spec = importlib.util.spec_from_file_location("alphalens_oracle_generator", ALPHALENS_ORACLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _current_python_prefix() -> Path:
    prefix = Path(sys.executable).resolve().parent.parent
    if not (prefix / "bin" / "python").is_file():
        # Windows layout: the interpreter lives at the prefix root.
        prefix = Path(sys.prefix)
    assert (prefix / "bin" / "python").is_file() or (prefix / "python.exe").is_file()
    return prefix


def _write_minimal_alphalens_checkout(checkout: Path, *, utils_source: str | None = None) -> Path:
    package = checkout / "alphalens"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from . import performance, utils\n", encoding="utf-8")
    (package / "performance.py").write_text("\n", encoding="utf-8")
    (package / "utils.py").write_text(
        utils_source or "def quantize_factor(factor_data, **_kwargs):\n    return factor_data\n",
        encoding="utf-8",
    )
    return checkout


def _write_worker_cases(path: Path, table: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "worker-round-trip",
                        "category": "ties-nan-zero",
                        "parameters": {},
                        "tables": {"factor_data": table},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _factor_data_table(
    index: list[list[str]], *, timezone: str | None, index_names: list[str] | None = None
) -> dict[str, Any]:
    return {
        "kind": "dataframe",
        "index": index,
        "index_names": index_names or ["date", "asset"],
        "timezone": timezone,
        "columns": ["factor"],
        "dtypes": {"factor": "float64"},
        "values": [[1.0] for _ in index],
        "nan_mask": [[False] for _ in index],
    }


def _process_has_exited(pid: int, *, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _terminate_owned_test_child(pid: int) -> None:
    """Clean up only the PID written by the bounded test subprocess itself."""
    if _process_has_exited(pid, timeout=0):
        return
    try:
        os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    except ProcessLookupError:
        return
    assert _process_has_exited(pid)


def _by_symbol(entries: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
    return next(entry for entry in entries if entry["symbol"] == symbol)


def _parameter(entry: dict[str, Any], name: str) -> dict[str, Any]:
    return next(parameter for parameter in entry["parameters"] if parameter["name"] == name)


def _assert_portable_provenance(data: dict[str, Any]) -> None:
    serialized = json.dumps(data, sort_keys=True)
    assert "/Users/" not in serialized
    assert "\\Users\\" not in serialized
    for source in data["source_files"]:
        assert not Path(source["path"]).is_absolute()
        assert len(source["sha256"]) == 64


def _assert_compatibility_levels(entry: dict[str, Any]) -> None:
    assert set(entry["compatibility"]) == {"C0", "C1", "C2", "C3", "C4"}


def _assert_reconstructable_signature(signature: str) -> None:
    ast.parse(f"def _f{signature}:\n    pass\n")
    namespace: dict[str, Any] = {}
    exec(f"def _f{signature}:\n    pass\n", {"__builtins__": {}}, namespace)
    assert str(inspect.signature(namespace["_f"])) == signature


def _reviewable_alphalens_evidence(generator: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return one small independent review tuple for attestation regression tests."""
    generated = {
        "commit": "a" * 40,
        "source_files": [{"path": "alphalens/utils.py", "sha256": "1" * 64}],
        "evidence_files": [{"path": "LICENSE", "sha256": "2" * 64}],
        "reported_versions": {"versioneer": "0.4.0", "setup_fallback": "1.0.0+dev"},
        "entries": [{"module": "utils", "symbol": "quantize_factor", "source_sha256": "1" * 64}],
        "oracle_verification": {"status": "not-run", "reviewed": False},
    }
    companions = {
        "cases": {
            "commit": "a" * 40,
            "serializer": {"name": "fincore-compat-json-table-v1"},
            "cases": [{"case_id": "small", "value": 1}],
            "oracle_verification": {"status": "not-run", "reviewed": False},
        },
        "environment": {
            "source": {"commit": "a" * 40},
            "runtime": {"fingerprint": {"platform": {"os": "macOS"}}},
            "oracle_verification": {"status": "not-run", "reviewed": False},
        },
        "explicit_lock_sha256": "4" * 64,
        "requirements_sha256": "5" * 64,
    }
    initial = generator._merge_alphalens_oracle_attestation(generated, None, companions)
    api_manifest_digest = generator._alphalens_json_digest(generated)
    environment_digest = generator._alphalens_json_digest(companions["environment"])
    attestation = {
        "api_manifest_digest": api_manifest_digest,
        "candidate_digest": "6" * 64,
        "environment_digest": environment_digest,
        "evidence_key": generator._alphalens_review_evidence_key(
            generated,
            companions,
            candidate_digest="6" * 64,
            environment_digest=environment_digest,
        ),
        "reviewed": True,
        "reviewed_at": "2026-08-13",
        "reviewer": "compat-reviewer@example.test",
        "status": "captured-reviewed",
    }
    reviewed = copy.deepcopy(initial)
    reviewed["oracle_verification"] = copy.deepcopy(attestation)
    companions["cases"]["oracle_verification"] = copy.deepcopy(attestation)
    companions["environment"]["oracle_verification"] = copy.deepcopy(attestation)
    return generated, reviewed, companions


def _reviewable_real_alphalens_manifest(generator: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build an attested tuple from the checked-in Alphalens profile, not a toy API."""
    generated = (
        generator._generate_alphalens(ALPHALENS_ROOT)
        if ALPHALENS_ROOT.is_dir()
        else _load("alphalens-0.4.0-cloudquant-api.json")
    )
    environment_path = Path(__file__).parent / "oracle" / "alphalens-0.4.0-cloudquant-environment.json"
    explicit_lock = environment_path.with_name("alphalens-0.4.0-cloudquant-conda-explicit.txt")
    requirements = environment_path.with_name("requirements-alphalens-0.4.0-cloudquant.txt")
    companions = {
        "cases": _load("alphalens-0.4.0-cloudquant-cases.json"),
        "environment": json.loads(environment_path.read_text(encoding="utf-8")),
        "explicit_lock_sha256": hashlib.sha256(explicit_lock.read_bytes()).hexdigest(),
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
    }
    initial = generator._merge_alphalens_oracle_attestation(generated, None, companions)
    api_manifest_digest = generator._alphalens_json_digest(generated)
    environment_digest = generator._alphalens_json_digest(companions["environment"])
    attestation = {
        "api_manifest_digest": api_manifest_digest,
        "candidate_digest": "d" * 64,
        "environment_digest": environment_digest,
        "evidence_key": generator._alphalens_review_evidence_key(
            generated,
            companions,
            candidate_digest="d" * 64,
            environment_digest=environment_digest,
        ),
        "reviewed": True,
        "reviewed_at": "2026-08-14",
        "reviewer": "compat-reviewer@example.test",
        "status": "captured-reviewed",
    }
    reviewed = copy.deepcopy(initial)
    reviewed["oracle_verification"] = copy.deepcopy(attestation)
    companions["cases"]["oracle_verification"] = copy.deepcopy(attestation)
    companions["environment"]["oracle_verification"] = copy.deepcopy(attestation)
    return generated, reviewed, companions


def test_alphalens_manifest_is_pinned_and_complete() -> None:
    data = _load("alphalens-0.4.0-cloudquant-api.json")
    assert data["profile"] == ALPHALENS_PROFILE
    assert data["commit"] == ALPHALENS_COMMIT
    assert data["reported_versions"] == {
        "setup_fallback": "1.0.0+dev",
        "versioneer": "0.4.0",
    }
    assert data["counts"] == {"classes": 3, "definitions": 64, "functions": 61}
    assert set(data["modules"]) == {"performance", "plotting", "tears", "utils"}
    assert {entry["path"] for entry in data["source_files"]} == {
        "alphalens/__init__.py",
        "alphalens/performance.py",
        "alphalens/plotting.py",
        "alphalens/tears.py",
        "alphalens/utils.py",
    }
    assert {entry["path"] for entry in data["evidence_files"]} == {
        "LICENSE",
        "README.md",
        "alphalens/_version.py",
        "setup.py",
    }
    _assert_portable_provenance(data)
    for file_entry in [*data["source_files"], *data["evidence_files"]]:
        assert len(file_entry["git_blob"]) == 40
        assert len(file_entry["sha256"]) == 64

    entries = data["entries"]
    assert len(entries) == 64
    assert len({(entry["module"], entry["symbol"]) for entry in entries}) == 64
    source_hashes = {entry["path"]: entry["sha256"] for entry in data["source_files"]}
    for entry in entries:
        assert {
            "module",
            "symbol",
            "kind",
            "source_signature",
            "introspection_signature",
            "accepted_call_cases",
            "source_line",
            "source_sha256",
            "C0",
            "C1",
            "C2",
            "C3",
            "C4",
        } <= set(entry)
        assert entry["module"] in data["modules"]
        assert entry["source_sha256"] == source_hashes[f"alphalens/{entry['module']}.py"]
        assert isinstance(entry["accepted_call_cases"], list)
        assert entry["C0"] == "not-verified"
        assert entry["C1"] == "not-verified"
        assert entry["C2"] == "not-verified"
        assert entry["C3"] == "not-verified"
        assert entry["C4"] == "not-verified"


def test_alphalens_manifest_freezes_wrapper_signatures_and_tear_sheet_grammar() -> None:
    data = _load("alphalens-0.4.0-cloudquant-api.json")
    entries = {(entry["module"], entry["symbol"]): entry for entry in data["entries"]}
    quantize_factor = entries[("utils", "quantize_factor")]
    assert quantize_factor["source_signature"] == (
        "(factor_data, quantiles=5, bins=None, by_group=False, no_raise=False, zero_aware=False)"
    )
    assert quantize_factor["introspection_signature"] == "(*args, **kwargs)"
    assert quantize_factor["decorator"] == "non_unique_bin_edges_error"

    expected_tear_sheets = {
        "create_summary_tear_sheet",
        "create_returns_tear_sheet",
        "create_information_tear_sheet",
        "create_turnover_tear_sheet",
        "create_full_tear_sheet",
        "create_event_returns_tear_sheet",
        "create_event_study_tear_sheet",
    }
    assert set(data["tear_sheets"]) == expected_tear_sheets
    for symbol in expected_tear_sheets:
        entry = entries[("tears", symbol)]
        assert entry["decorator"] == "plotting.customize"
        assert entry["source_signature"] == entry["introspection_signature"]
        cases = {case["case_id"]: case for case in entry["accepted_call_cases"]}
        assert {"source-visible", "customize-set-context-true", "customize-set-context-false"} <= set(cases)
        assert cases["customize-set-context-true"]["hidden_kwargs"] == {"set_context": True}
        assert cases["customize-set-context-false"]["hidden_kwargs"] == {"set_context": False}


def test_alphalens_dynamic_defaults_do_not_claim_a_runtime_introspection_signature() -> None:
    data = _load("alphalens-0.4.0-cloudquant-api.json")
    entry = next(item for item in data["entries"] if item["module"] == "plotting" and item["symbol"] == "plot_ic_qq")
    assert entry["source_signature"] == "(ic, theoretical_dist=stats.norm, ax=None)"
    assert entry["needs_dynamic_review"] is True
    assert entry["introspection_signature"] is None


def test_alphalens_case_fixture_is_portable_and_unreviewed() -> None:
    data = _load("alphalens-0.4.0-cloudquant-cases.json")
    assert data["profile"] == ALPHALENS_PROFILE
    assert data["commit"] == ALPHALENS_COMMIT
    assert data["oracle_verification"] == {
        "reviewed": False,
        "status": "not-run",
    }
    expected_categories = {
        "daily",
        "business-day",
        "intraday",
        "tz-aware",
        "ties-nan-zero",
        "group-neutral",
        "bins-quantiles",
        "max-loss-boundary",
        "pre-cleaned-performance",
        "event-window",
        "pyfolio-input",
    }
    assert expected_categories <= {case["category"] for case in data["cases"]}
    assert len({case["case_id"] for case in data["cases"]}) == len(data["cases"])
    assert all(case["serializer"] == "fincore-compat-json-table-v1" for case in data["cases"])
    assert all("expected_output" not in case for case in data["cases"])
    _assert_portable_provenance(data)


def test_alphalens_oracle_metadata_is_truthful_unreviewed_observation() -> None:
    environment_path = Path(__file__).parent / "oracle" / "alphalens-0.4.0-cloudquant-environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    explicit_lock = environment_path.with_name("alphalens-0.4.0-cloudquant-conda-explicit.txt")
    requirements = environment_path.with_name("requirements-alphalens-0.4.0-cloudquant.txt")
    assert environment["profile"] == ALPHALENS_PROFILE
    assert environment["source"]["commit"] == ALPHALENS_COMMIT
    assert environment["oracle_verification"] == {"reviewed": False, "status": "not-run"}
    assert environment["execution_status"] == "executable-unreviewed-tuple"
    # Hash canonical LF-normalized bytes: the recorded digests come from the
    # reference-host checkout and Windows checkouts may translate line endings.
    assert environment["explicit_lock"]["sha256"] == _sha256_text_file(explicit_lock)
    assert environment["requirements"]["sha256"] == _sha256_text_file(requirements)
    assert "@EXPLICIT" in explicit_lock.read_text(encoding="utf-8")
    explicit_packages = [
        line
        for line in explicit_lock.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and line != "@EXPLICIT"
    ]
    requirements_packages = [
        line
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and not line.startswith("--")
    ]
    assert len(explicit_packages) == environment["explicit_lock"]["package_url_count"]
    assert all("#" in line for line in explicit_packages)
    assert environment["requirements"]["package_count"] == len(requirements_packages)
    assert "--require-hashes" in requirements.read_text(encoding="utf-8")
    assert all(" --hash=sha256:" in line for line in requirements_packages)
    records = environment["distribution_inventory"]["records"]
    generator = _load_alphalens_oracle()
    assert len(records) == 59
    assert environment["distribution_inventory"]["sha256"] == generator._json_digest({"records": records})
    assert all({"name", "version", "build", "channel", "platform"} <= set(record) for record in records)
    assert environment["runtime"]["raw"]["platform"]["system"] == "Darwin"
    assert environment["runtime"]["normalized"]["platform"] == {
        "byteorder": "little",
        "machine": "arm64",
        "os": "macOS",
        "processor": "arm",
        "raw_system": "Darwin",
        "release": "25.5.0",
    }
    serialized = json.dumps(environment, sort_keys=True)
    assert "/Users/" not in serialized
    assert "\\Users\\" not in serialized
    assert "/private/" not in serialized
    _assert_portable_provenance({"source_files": environment["source"]["source_files"]})


def test_alphalens_generation_does_not_rewrite_existing_manifests(tmp_path: Path) -> None:
    if not ALPHALENS_ROOT.is_dir():
        pytest.skip("pinned Alphalens sibling root is not available; frozen fixture remains CI input")
    pinned = {name: (FIXTURES / name).read_bytes() for name in _EXISTING_FIXTURES}
    command = [
        sys.executable,
        GENERATOR.as_posix(),
        "--alphalens-root",
        ALPHALENS_ROOT.as_posix(),
        "--target",
        "alphalens",
        "--output",
        tmp_path.as_posix(),
    ]
    subprocess.run(command, check=True)
    assert {name: (FIXTURES / name).read_bytes() for name in _EXISTING_FIXTURES} == pinned
    generated = tmp_path / "alphalens-0.4.0-cloudquant-api.json"
    assert generated.read_bytes()
    first = generated.read_bytes()
    subprocess.run(command, check=True)
    assert generated.read_bytes() == first
    assert generated.read_bytes() == (FIXTURES / generated.name).read_bytes()


def test_alphalens_manifest_hashes_are_read_from_pinned_git_blobs() -> None:
    if not ALPHALENS_ROOT.is_dir():
        pytest.skip("pinned Alphalens sibling root is not available; frozen fixture remains CI input")
    data = _load("alphalens-0.4.0-cloudquant-api.json")
    generator = _load_generator()
    source = generator.PinnedGitSource(ALPHALENS_ROOT, ALPHALENS_COMMIT)
    for entry in [*data["source_files"], *data["evidence_files"]]:
        assert source.sha256(entry["path"]) == entry["sha256"]
        assert source.blob_id(entry["path"]) == entry["git_blob"]


def test_alphalens_oracle_refuses_nonexecutable_environment_without_writing_output(tmp_path: Path) -> None:
    if not ALPHALENS_ROOT.is_dir():
        pytest.skip("pinned Alphalens sibling root is not available; frozen fixture remains CI input")
    environment_path = Path(__file__).parent / "oracle" / "alphalens-0.4.0-cloudquant-environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["execution_status"] = "unreviewed-current-base-observation"
    invalid_environment = tmp_path / "environment.json"
    invalid_environment.write_text(json.dumps(environment), encoding="utf-8")
    output = tmp_path / "candidate.json"
    result = subprocess.run(
        [
            sys.executable,
            (ROOT / "scripts" / "generate_alphalens_oracle.py").as_posix(),
            "--source",
            ALPHALENS_ROOT.as_posix(),
            "--commit",
            ALPHALENS_COMMIT,
            "--environment",
            invalid_environment.as_posix(),
            "--explicit-lock",
            (Path(__file__).parent / "oracle" / "alphalens-0.4.0-cloudquant-conda-explicit.txt").as_posix(),
            "--cases",
            (FIXTURES / "alphalens-0.4.0-cloudquant-cases.json").as_posix(),
            "--output",
            output.as_posix(),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "executable isolated tuple" in f"{result.stdout}\n{result.stderr}".lower()
    assert not output.exists()


def test_alphalens_oracle_rejects_a_dirty_pinned_checkout_without_writing_output(tmp_path: Path) -> None:
    if not ALPHALENS_ROOT.is_dir():
        pytest.skip("pinned Alphalens sibling root is not available; frozen fixture remains CI input")
    source = tmp_path / "alphalens"
    subprocess.run(["git", "clone", "--quiet", ALPHALENS_ROOT.as_posix(), source.as_posix()], check=True)
    subprocess.run(["git", "checkout", "--quiet", ALPHALENS_COMMIT], cwd=source, check=True)
    (source / "untracked-oracle-marker").write_text("dirty\n", encoding="utf-8")
    output = tmp_path / "candidate.json"
    result = subprocess.run(
        [
            sys.executable,
            (ROOT / "scripts" / "generate_alphalens_oracle.py").as_posix(),
            "--source",
            source.as_posix(),
            "--commit",
            ALPHALENS_COMMIT,
            "--environment",
            (Path(__file__).parent / "oracle" / "alphalens-0.4.0-cloudquant-environment.json").as_posix(),
            "--explicit-lock",
            (Path(__file__).parent / "oracle" / "alphalens-0.4.0-cloudquant-conda-explicit.txt").as_posix(),
            "--cases",
            (FIXTURES / "alphalens-0.4.0-cloudquant-cases.json").as_posix(),
            "--output",
            output.as_posix(),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "dirty" in f"{result.stdout}\n{result.stderr}".lower()
    assert not output.exists()


def test_alphalens_oracle_refuses_to_write_inside_the_source_checkout() -> None:
    if not ALPHALENS_ROOT.is_dir():
        pytest.skip("pinned Alphalens sibling root is not available; frozen fixture remains CI input")
    output = ALPHALENS_ROOT / "oracle-candidate-must-not-exist.json"
    assert not output.exists()
    result = subprocess.run(
        [
            sys.executable,
            (ROOT / "scripts" / "generate_alphalens_oracle.py").as_posix(),
            "--source",
            ALPHALENS_ROOT.as_posix(),
            "--commit",
            ALPHALENS_COMMIT,
            "--environment",
            (Path(__file__).parent / "oracle" / "alphalens-0.4.0-cloudquant-environment.json").as_posix(),
            "--explicit-lock",
            (Path(__file__).parent / "oracle" / "alphalens-0.4.0-cloudquant-conda-explicit.txt").as_posix(),
            "--cases",
            (FIXTURES / "alphalens-0.4.0-cloudquant-cases.json").as_posix(),
            "--output",
            output.as_posix(),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "source checkout" in f"{result.stdout}\n{result.stderr}".lower()
    assert not output.exists()


def test_alphalens_worker_isolated_from_poisoned_caller_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    oracle = _load_alphalens_oracle()
    checkout = _write_minimal_alphalens_checkout(tmp_path / "clean-checkout")
    poison = tmp_path / "poisoned-caller"
    poison.mkdir()
    _write_worker_cases(
        poison / "cases.json",
        _factor_data_table([["2020-01-02T09:30:00", "A"]], timezone=None),
    )
    marker = poison / "sitecustomize-ran"
    (poison / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({marker.as_posix()!r}).write_text('poisoned', encoding='utf-8')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(poison)
    monkeypatch.setenv("PYTHONPATH", poison.as_posix())

    payload = oracle._execute_oracle_worker(_current_python_prefix(), checkout, Path("cases.json"))

    assert not marker.exists()
    context = payload["execution_context"]
    assert context["isolated"] is True
    assert Path(context["cwd"]).resolve() == checkout.resolve()
    assert Path(context["prefix"]).resolve() == _current_python_prefix().resolve()
    assert payload["import_path"] == "alphalens/__init__.py"
    assert payload["case_results"][0]["result"]["status"] == "ok"


def test_alphalens_worker_round_trips_named_zone_dst_multiindex(tmp_path: Path) -> None:
    oracle = _load_alphalens_oracle()
    checkout = _write_minimal_alphalens_checkout(tmp_path / "clean-checkout")
    input_index = [
        ["2020-11-01T01:30:00-04:00", "A"],
        ["2020-11-01T01:30:00-05:00", "B"],
    ]
    cases = _write_worker_cases(
        tmp_path / "dst-cases.json",
        _factor_data_table(input_index, timezone="America/New_York"),
    )

    payload = oracle._execute_oracle_worker(_current_python_prefix(), checkout, cases)

    value = payload["case_results"][0]["result"]["value"]
    assert value["timezone"] == "America/New_York"
    assert value["index_names"] == ["date", "asset"]
    assert value["index"] == input_index


def test_alphalens_worker_serializes_plain_index_without_timezone_attribute(tmp_path: Path) -> None:
    oracle = _load_alphalens_oracle()
    checkout = _write_minimal_alphalens_checkout(
        tmp_path / "clean-checkout",
        utils_source="def quantize_factor(factor_data, **_kwargs):\n    return factor_data.reset_index(drop=True)\n",
    )
    cases = _write_worker_cases(
        tmp_path / "plain-index-cases.json",
        _factor_data_table([["2020-01-02T09:30:00", "A"]], timezone=None),
    )

    payload = oracle._execute_oracle_worker(_current_python_prefix(), checkout, cases)

    value = payload["case_results"][0]["result"]["value"]
    assert value["timezone"] is None
    assert value["index_names"] == [None]
    assert value["index"] == [0]


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX process groups and session reaping do not exist on Windows",
)
def test_alphalens_oracle_timeout_terminates_owned_process_group_and_cleans_prefix(tmp_path: Path) -> None:
    oracle = _load_alphalens_oracle()
    child_pid_path = tmp_path / "child.pid"
    script = tmp_path / "spawn-child.py"
    script.write_text(
        "from pathlib import Path\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    temporary_prefix: Path | None = None
    child_pid: int | None = None
    child_exited = False
    try:
        with tempfile.TemporaryDirectory(dir=tmp_path, prefix="oracle-prefix-") as temporary:
            temporary_prefix = Path(temporary)
            with pytest.raises(ValueError, match="bounded child.*timed out"):
                oracle._run_process(
                    [sys.executable, script.as_posix(), child_pid_path.as_posix()],
                    "bounded child",
                    cwd=temporary_prefix,
                    timeout=1,
                )
            assert child_pid_path.is_file()
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
            child_exited = _process_has_exited(child_pid)
        assert temporary_prefix is not None
        assert not temporary_prefix.exists()
        assert child_exited
    finally:
        if child_pid is None and child_pid_path.is_file():
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        if child_pid is not None:
            _terminate_owned_test_child(child_pid)


@pytest.mark.integration
@pytest.mark.integration_online
@pytest.mark.serial
def test_alphalens_oracle_executed_tuple_end_to_end(tmp_path: Path) -> None:
    if os.environ.get("FINCORE_RUN_ALPHALENS_ORACLE_E2E") != "1":
        pytest.skip("set FINCORE_RUN_ALPHALENS_ORACLE_E2E=1 to recreate the pinned external tuple")
    if not ALPHALENS_ROOT.is_dir():
        pytest.skip("pinned Alphalens sibling root is not available")
    environment_path = Path(__file__).parent / "oracle" / "alphalens-0.4.0-cloudquant-environment.json"
    explicit_lock = environment_path.with_name("alphalens-0.4.0-cloudquant-conda-explicit.txt")
    requirements = environment_path.with_name("requirements-alphalens-0.4.0-cloudquant.txt")
    cases = FIXTURES / "alphalens-0.4.0-cloudquant-cases.json"
    output = tmp_path / "candidate.json"
    temporary_root = tmp_path / "temporary-prefix-root"
    temporary_root.mkdir()
    environment = os.environ.copy()
    environment["TMPDIR"] = temporary_root.as_posix()

    result = subprocess.run(
        [
            sys.executable,
            ALPHALENS_ORACLE.as_posix(),
            "--source",
            ALPHALENS_ROOT.as_posix(),
            "--commit",
            ALPHALENS_COMMIT,
            "--environment",
            environment_path.as_posix(),
            "--explicit-lock",
            explicit_lock.as_posix(),
            "--cases",
            cases.as_posix(),
            "--output",
            output.as_posix(),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1500,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    candidate = json.loads(output.read_text(encoding="utf-8"))
    assert candidate["execution"] == "isolated-prefix-clean-checkout-deterministic-case-execution"
    assert candidate["commit"] == ALPHALENS_COMMIT
    assert candidate["reviewed"] is False
    assert len(candidate["case_results"]) == len(_load("alphalens-0.4.0-cloudquant-cases.json")["cases"])
    assert candidate["environment"]["explicit_lock_sha256"] == hashlib.sha256(explicit_lock.read_bytes()).hexdigest()
    assert candidate["environment"]["requirements_sha256"] == hashlib.sha256(requirements.read_bytes()).hexdigest()
    assert not list(temporary_root.glob("fincore-alphalens-oracle-*"))


def test_alphalens_oracle_review_attestation_invalidates_for_every_review_relevant_digest() -> None:
    generator = _load_generator()
    generated, reviewed, companions = _reviewable_alphalens_evidence(generator)
    preserved = generator._merge_alphalens_oracle_attestation(generated, reviewed, companions)
    assert preserved["oracle_verification"] == reviewed["oracle_verification"]

    mutations = {
        "api-source": lambda api, evidence: api["source_files"][0].__setitem__("sha256", "7" * 64),
        "api-evidence": lambda api, evidence: api["evidence_files"][0].__setitem__("sha256", "8" * 64),
        "cases": lambda api, evidence: evidence["cases"]["cases"][0].__setitem__("value", 2),
        "environment": lambda api, evidence: evidence["environment"]["runtime"]["fingerprint"].__setitem__(
            "schema", "changed"
        ),
        "conda-explicit-lock": lambda api, evidence: evidence.__setitem__("explicit_lock_sha256", "9" * 64),
        "pip-requirements-lock": lambda api, evidence: evidence.__setitem__("requirements_sha256", "a" * 64),
        "candidate-output-digest": lambda api, evidence: evidence["cases"]["oracle_verification"].__setitem__(
            "candidate_digest", "b" * 64
        ),
        "candidate-environment-digest": lambda api, evidence: evidence["environment"][
            "oracle_verification"
        ].__setitem__("environment_digest", "c" * 64),
    }
    for name, mutate in mutations.items():
        drifted_api = copy.deepcopy(generated)
        drifted_companions = copy.deepcopy(companions)
        mutate(drifted_api, drifted_companions)
        invalidated = generator._merge_alphalens_oracle_attestation(drifted_api, reviewed, drifted_companions)
        assert invalidated["oracle_verification"]["reviewed"] is False, name
        assert invalidated["oracle_verification"]["status"] == "not-run", name


def test_alphalens_review_attestation_binds_the_entire_real_api_manifest() -> None:
    generator = _load_generator()
    generated, reviewed, companions = _reviewable_real_alphalens_manifest(generator)
    assert generated["profile"] == ALPHALENS_PROFILE
    assert generated["counts"]["functions"] == 61
    assert (
        generator._merge_alphalens_oracle_attestation(generated, reviewed, companions)["oracle_verification"][
            "reviewed"
        ]
        is True
    )

    mutations = {
        "profile": lambda manifest: manifest.__setitem__("profile", "changed-profile"),
        "counts.functions": lambda manifest: manifest["counts"].__setitem__("functions", 62),
    }
    for name, mutate in mutations.items():
        drifted = copy.deepcopy(generated)
        mutate(drifted)
        invalidated = generator._merge_alphalens_oracle_attestation(drifted, reviewed, copy.deepcopy(companions))
        assert invalidated["oracle_verification"]["reviewed"] is False, name
        assert invalidated["oracle_verification"]["status"] == "not-run", name


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reviewer", ""),
        ("reviewed_at", "not-a-date"),
        ("api_manifest_digest", None),
        ("candidate_digest", None),
        ("environment_digest", None),
    ],
)
def test_alphalens_reviewed_attestation_requires_reviewer_date_and_matching_digests(field: str, value: Any) -> None:
    generator = _load_generator()
    generated, reviewed, companions = _reviewable_alphalens_evidence(generator)
    reviewed["oracle_verification"][field] = value
    for sidecar in (companions["cases"], companions["environment"]):
        sidecar["oracle_verification"][field] = value

    invalidated = generator._merge_alphalens_oracle_attestation(generated, reviewed, companions)
    assert invalidated["oracle_verification"]["reviewed"] is False
    assert invalidated["oracle_verification"]["status"] == "not-run"


def test_alphalens_oracle_approved_tuple_validates_darwin_as_macos() -> None:
    oracle = _load_alphalens_oracle()
    raw_runtime = {
        "python": {"implementation": "CPython", "version": "3.11.8", "soabi": "cpython-311-darwin"},
        "platform": {
            "system": "Darwin",
            "release": "26.5.1",
            "machine": "arm64",
            "processor": "arm",
            "byteorder": "little",
        },
        "locale": "C/C.UTF-8/C/C/C/C",
        "timezone": {"TZ": None, "tzname": ["CST", "CST"]},
        "blas": {"configuration": "known", "found": True, "name": "openblas", "version": "1"},
        "distributions": {},
    }
    normalized_runtime = oracle._normalize_runtime_fingerprint(raw_runtime)
    assert normalized_runtime["platform"]["os"] == "macOS"
    assert normalized_runtime["platform"]["raw_system"] == "Darwin"
    environment = {
        "runtime": {"raw": raw_runtime, "normalized": normalized_runtime},
        "execution_status": "reviewed-executable-tuple",
    }
    assert oracle._validate_isolated_runtime(environment, raw_runtime) == normalized_runtime


def test_empyrical_manifest_is_pinned_and_complete() -> None:
    data = _load("empyrical-0.6.0-api.json")
    assert data["version"] == "0.6.0"
    assert data["commit"] == "74655e974ed2935563820c548c339731f1fe0621"
    assert len(data["public_symbols"]) == 54
    assert len(data["callables"]) == 49
    assert {"calmar_ratio", "beta", "perf_attrib", "DAILY"} <= set(data["public_symbols"])
    assert len(set(data["public_symbols"])) == 54
    assert all(entry["extraction"] == "static_ast_from_pinned_git_blob" for entry in data["symbols"])
    assert all(entry["reviewed"] is False for entry in data["symbols"])
    assert all(entry["signature"].startswith("(") for entry in data["callables"])
    for entry in data["callables"]:
        _assert_reconstructable_signature(entry["signature"])
    dynamic_review = [entry for entry in data["symbols"] if entry["needs_dynamic_review"]]
    assert len(dynamic_review) == 9
    assert all(entry["factory"].startswith("_create_") for entry in dynamic_review)
    for entry in data["symbols"]:
        _assert_compatibility_levels(entry)
    _assert_portable_provenance(data)


def test_empyrical_manifest_includes_pinned_utils_blob() -> None:
    data = _load("empyrical-0.6.0-api.json")
    source_hashes = {source["path"]: source["sha256"] for source in data["source_files"]}
    expected_hash = "aff1a9d686b576ad971e7985b22a24f0460100a90e4cb2ab6c7b7f8ca6dc76d9"
    assert source_hashes["utils.py"] == expected_hash

    empyrical_root = ROOT.parent / "empyrical"
    if empyrical_root.is_dir():
        generator = _load_generator()
        pinned = generator.PinnedGitSource(empyrical_root, data["commit"])
        assert pinned.sha256("empyrical/utils.py") == expected_hash


def test_empyrical_canonical_signatures_use_runtime_defaults() -> None:
    data = _load("empyrical-0.6.0-api.json")
    calmar = _by_symbol(data["callables"], "calmar_ratio")
    beta = _by_symbol(data["callables"], "beta")
    assert calmar["signature"] == "(returns, period='daily', annualization=None)"
    assert _parameter(calmar, "period") == {
        "annotation": None,
        "default": "daily",
        "default_expression": "DAILY",
        "kind": "POSITIONAL_OR_KEYWORD",
        "name": "period",
        "required": False,
        "resolved": True,
    }
    assert beta["signature"] == "(returns, factor_returns, risk_free=0.0, out=None)"
    assert _parameter(beta, "risk_free")["default"] == 0.0
    assert _parameter(beta, "risk_free")["kind"] == "POSITIONAL_OR_KEYWORD"


def test_pyfolio_canonical_signatures_resolve_constants_and_expressions() -> None:
    profile = _load("pyfolio-0.9.6-api.json")["compatibility_profile"]
    capacity = profile["create_capacity_tear_sheet"]
    full = profile["create_full_tear_sheet"]
    assert capacity["signature"] == (
        "(returns, positions, transactions, market_data, "
        "liquidation_daily_vol_limit=0.2, trade_daily_vol_limit=0.05, "
        "last_n_days=126, days_to_liquidate_limit=1, estimate_intraday='infer', "
        "run_flask_app=False)"
    )
    assert _parameter(capacity, "last_n_days") == {
        "annotation": None,
        "default": 126,
        "default_expression": "utils.APPROX_BDAYS_PER_MONTH * 6",
        "kind": "POSITIONAL_OR_KEYWORD",
        "name": "last_n_days",
        "required": False,
        "resolved": True,
    }
    expected_full = (
        "(returns, positions=None, transactions=None, market_data=None, "
        "benchmark_rets=None, slippage=None, live_start_date=None, "
        "sector_mappings=None, bayesian=False, round_trips=False, "
        "estimate_intraday='infer', hide_positions=False, cone_std=(1.0, 1.5, 2.0), "
        "bootstrap=False, unadjusted_returns=None, style_factor_panel=None, "
        "sectors=None, caps=None, shares_held=None, volumes=None, percentile=None, "
        "turnover_denom='AGB', set_context=True, factor_returns=None, "
        "factor_loadings=None, pos_in_dollars=True, header_rows=None, "
        f"factor_partitions={FACTOR_PARTITIONS!r})"
    )
    assert full["signature"] == expected_full
    assert _parameter(full, "factor_partitions")["default"] == FACTOR_PARTITIONS
    assert _parameter(full, "factor_partitions")["resolved"] is True
    assert _parameter(full, "factor_partitions")["kind"] == "POSITIONAL_OR_KEYWORD"


def test_unresolved_defaults_are_not_labeled_signature_ready() -> None:
    generator = _load_generator()
    tree = ast.parse("def target(value=MISSING):\n    pass\n")
    function = tree.body[0]
    resolver = generator.StaticConstantResolver({"module": tree})
    parameters = generator._parameters(function, resolver, "module")
    assert parameters[0]["resolved"] is False
    assert parameters[0]["default"] is None
    assert parameters[0]["default_expression"] == "MISSING"
    assert "unknown name" in parameters[0]["unresolved_reason"]
    assert generator._canonical_signature(function, parameters) is None


def test_constant_resolver_rejects_huge_power_before_operator_runs(monkeypatch) -> None:
    generator = _load_generator()
    tree = ast.parse("VALUE = 2 ** 1000000")
    resolver = generator.StaticConstantResolver({"module": tree})

    def forbidden_power(_left: object, _right: object) -> object:
        pytest.fail("unsafe exponent reached operator.pow")

    monkeypatch.setitem(generator.SAFE_BINARY_OPERATORS, ast.Pow, forbidden_power)
    value, resolved = resolver.resolve(tree.body[0].value, "module")
    assert value is None
    assert resolved is False
    assert "exponent" in resolver.last_unresolved_reason


def test_constant_resolver_rejects_oversized_container_before_children(monkeypatch) -> None:
    generator = _load_generator()
    node = ast.List(
        elts=[ast.Constant(value=index) for index in range(generator.MAX_CONTAINER_ITEMS + 1)],
        ctx=ast.Load(),
    )
    resolver = generator.StaticConstantResolver({"module": ast.Module(body=[], type_ignores=[])})
    original = resolver._resolve
    visits = 0

    def counting_resolve(*args: Any, **kwargs: Any) -> Any:
        nonlocal visits
        visits += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(resolver, "_resolve", counting_resolve)
    assert resolver.resolve(node, "module") == (None, False)
    assert visits == 1
    assert "container" in resolver.last_unresolved_reason


@pytest.mark.parametrize(
    "node",
    [
        ast.Constant(value=b"bytes"),
        ast.Constant(value=1 + 2j),
        ast.Set(elts=[ast.Constant(value=1)]),
        ast.Constant(value=float("inf")),
        ast.Constant(value=float("nan")),
    ],
)
def test_constant_resolver_rejects_non_json_or_non_finite_values(node: ast.expr) -> None:
    generator = _load_generator()
    resolver = generator.StaticConstantResolver({"module": ast.Module(body=[], type_ignores=[])})
    assert resolver.resolve(node, "module") == (None, False)
    assert resolver.last_unresolved_reason


def test_constant_resolver_enforces_depth_and_visit_budgets() -> None:
    generator = _load_generator()
    deep: ast.expr = ast.Constant(value=1)
    for _ in range(generator.MAX_RESOLVE_DEPTH + 1):
        deep = ast.UnaryOp(op=ast.UAdd(), operand=deep)
    resolver = generator.StaticConstantResolver({"module": ast.Module(body=[], type_ignores=[])})
    assert resolver.resolve(deep, "module") == (None, False)
    assert "depth" in resolver.last_unresolved_reason

    nodes: list[ast.expr] = [ast.Constant(value=1) for _ in range(2048)]
    while len(nodes) > 1:
        nodes = [
            ast.BinOp(left=nodes[index], op=ast.Add(), right=nodes[index + 1]) for index in range(0, len(nodes), 2)
        ]
    resolver = generator.StaticConstantResolver({"module": ast.Module(body=[], type_ignores=[])})
    assert resolver.resolve(nodes[0], "module") == (None, False)
    assert "node visits" in resolver.last_unresolved_reason


def test_pyfolio_manifest_is_pinned() -> None:
    data = _load("pyfolio-0.9.6-api.json")
    assert data["version"] == "0.9.6"
    assert data["commit"] == "724bbd7dbed9a88bb47e1057f2ca29b3409d8e7a"
    assert "create_full_tear_sheet" in data["compatibility_profile"]
    assert "create_risk_tear_sheet" in data["compatibility_profile"]
    assert len(data["compatibility_profile"]) == 11
    for entry in data["compatibility_profile"].values():
        assert entry["public_path"].startswith("pyfolio.")
        assert isinstance(entry["parameters"], list)
        assert entry["signature"].startswith("(")
        _assert_reconstructable_signature(entry["signature"])
        _assert_compatibility_levels(entry)
    _assert_portable_provenance(data)


def test_flat_api_migration_manifest_is_explicit() -> None:
    data = _load("fincore-flat-api-migrations.json")
    assert data["fincore_version"] == "0.4.0.dev0"
    assert data["policy"] == "preserve-0.3.x-flat-api"
    assert data["entries"]
    for entry in data["entries"]:
        assert {
            "symbol",
            "current_target",
            "recommended_target",
            "deprecate_in",
            "remove_or_switch_in",
        } <= set(entry)


def test_generator_default_path_is_ast_only() -> None:
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])
    assert {"empyrical", "pyfolio", "importlib", "inspect"}.isdisjoint(imported)
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
        and node.attr == "path"
        for node in ast.walk(tree)
    )
    assert _load("empyrical-0.6.0-api.json")["oracle_verification"] == {
        "reviewed": False,
        "status": "not_run",
    }


def test_static_exports_preserve_aliases_and_expand_star() -> None:
    generator = _load_generator()
    trees = {
        "__init__": ast.parse("from .direct import original as public_alias\nfrom .star import *\n"),
        "direct": ast.parse("def original():\n    pass\n"),
        "star": ast.parse("__all__ = ['chosen']\nfrom .impl import source_name as chosen\nhidden = 1\n"),
        "impl": ast.parse("def source_name():\n    pass\n"),
    }
    exports = generator._resolve_public_exports("__init__", trees)
    assert [(item.public_name, item.source_name, item.source_module) for item in exports] == [
        ("public_alias", "original", "direct"),
        ("chosen", "source_name", "impl"),
    ]


def test_static_star_without_all_uses_python_public_names() -> None:
    generator = _load_generator()
    trees = {
        "__init__": ast.parse("from .star import *\n"),
        "star": ast.parse("from .impl import source as alias\ndef local():\n    pass\n_hidden = 1\n"),
        "impl": ast.parse("def source():\n    pass\n"),
    }
    exports = generator._resolve_public_exports("__init__", trees)
    assert [(item.public_name, item.source_name, item.source_module) for item in exports] == [
        ("alias", "source", "impl"),
        ("local", "local", "star"),
    ]


def test_review_attestation_is_preserved_only_for_identical_evidence() -> None:
    generator = _load_generator()
    generated = {
        "project": "empyrical",
        "commit": "abc",
        "source_files": [{"path": "stats.py", "sha256": "1" * 64}],
        "oracle_verification": {"status": "not_run", "reviewed": False},
        "symbols": [{"symbol": "alpha", "signature": "(returns)", "reviewed": False}],
    }
    previous = json.loads(json.dumps(generated))
    previous["symbols"][0]["reviewed"] = True
    merged = generator._merge_review_attestations(generated, previous, "symbols")
    assert merged["symbols"][0]["reviewed"] is True
    drifted = json.loads(json.dumps(generated))
    drifted["symbols"][0]["signature"] = "(returns, factor=None)"
    invalidated = generator._merge_review_attestations(drifted, previous, "symbols")
    assert invalidated["symbols"][0]["reviewed"] is False


def test_pinned_source_reads_committed_blob_when_worktree_is_dirty(tmp_path: Path) -> None:
    generator = _load_generator()
    root = tmp_path / "upstream"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    source = root / "package.py"
    source.write_text("VALUE = 'pinned'\n", encoding="utf-8")
    subprocess.run(["git", "add", "package.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "pin"], cwd=root, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    source.write_text("VALUE = 'dirty'\n", encoding="utf-8")
    pinned = generator.PinnedGitSource(root, commit)
    assert pinned.read_text("package.py") == "VALUE = 'pinned'\n"
    assert pinned.sha256("package.py") != generator.hashlib.sha256(source.read_bytes()).hexdigest()


def test_pinned_git_read_timeout_names_operation(tmp_path: Path, monkeypatch) -> None:
    generator = _load_generator()
    source = object.__new__(generator.PinnedGitSource)
    source.root = tmp_path
    source.commit = "abc123"

    def time_out(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(generator.subprocess, "run", time_out)
    with pytest.raises(ValueError, match="read pinned Git blob.*timed out"):
        source.read_bytes("package.py")


def test_oracle_execution_timeout_names_operation(tmp_path: Path, monkeypatch) -> None:
    generator = _load_generator()

    def time_out(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(generator.subprocess, "run", time_out)
    with pytest.raises(ValueError, match="execute isolated oracle.*timed out"):
        generator._execute_oracle(Path(sys.executable), tmp_path, "samplepkg", ["target"], [])


def test_generator_subprocess_calls_are_centralized_and_bounded() -> None:
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
    ]
    assert len(calls) == 1
    wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_process")
    assert calls[0] in list(ast.walk(wrapper))
    keyword_names = {keyword.arg for keyword in calls[0].keywords}
    assert {"timeout", "env", "stdin"} <= keyword_names
    source = GENERATOR.read_text(encoding="utf-8")
    assert '"GIT_TERMINAL_PROMPT": "0"' in source
    assert "timeout=LOCAL_GIT_TIMEOUT_SECONDS" in source


def test_oracle_rejects_wrong_source_version(tmp_path: Path) -> None:
    generator = _load_generator()
    root = tmp_path / "wrong"
    package = root / "samplepkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "__version__ = '9.9.9'\ndef target(value=1):\n    return value\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "samplepkg/__init__.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "pin"], cwd=root, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    with pytest.raises(ValueError, match="oracle version"):
        generator._run_oracle(
            Path(sys.executable),
            root,
            "samplepkg",
            ["target"],
            expected_version="1.0.0",
            expected_commit=commit,
            expected_source_files=[{"path": "__init__.py", "sha256": "unused"}],
        )


def test_oracle_rejects_same_named_installed_package_outside_pin(tmp_path: Path) -> None:
    generator = _load_generator()
    root = tmp_path / "empty-pin"
    root.mkdir()
    (root / "README.md").write_text("no pytest package in this pin\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "pin"], cwd=root, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    with pytest.raises(ValueError, match="outside pinned root"):
        generator._run_oracle(
            Path(sys.executable),
            root,
            "pytest",
            ["main"],
            expected_version="installed-version-must-not-be-used",
            expected_commit=commit,
            expected_source_files=[],
        )


def test_full_generator_is_byte_idempotent_when_pinned_roots_are_available(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).parents[3]
    empyrical_root = project_root / "empyrical"
    pyfolio_root = project_root / "pyfolio"
    if not (empyrical_root.is_dir() and pyfolio_root.is_dir()):
        pytest.skip("pinned sibling roots are not available; frozen fixtures remain CI input")
    command = [
        sys.executable,
        GENERATOR.as_posix(),
        "--empyrical-root",
        empyrical_root.as_posix(),
        "--pyfolio-root",
        pyfolio_root.as_posix(),
        "--output",
        tmp_path.as_posix(),
    ]
    subprocess.run(command, check=True)
    first = {path.name: path.read_bytes() for path in sorted(tmp_path.glob("*.json"))}
    subprocess.run(command, check=True)
    second = {path.name: path.read_bytes() for path in sorted(tmp_path.glob("*.json"))}
    assert first == second
    assert set(first) == {
        "empyrical-0.6.0-api.json",
        "fincore-flat-api-migrations.json",
        "pyfolio-0.9.6-api.json",
        "pyfolio-0.9.6-portfolio-contracts.json",
    }
    for name, contents in first.items():
        assert contents == (FIXTURES / name).read_bytes()
