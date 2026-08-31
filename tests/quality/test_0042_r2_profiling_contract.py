"""Fail-closed contracts for the 0042-R2 workload profiling tooling.

The plan pre-registers six hotspot families and one measurement contract
(warmups + measured repeats, output digest verification, same platform and
threading lane).  These tests freeze the orchestrator's schema, selection
validation, and per-case fail-closed validation without re-running workloads.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts" / "profile_workloads.py"
HOTSPOT_SCRIPT = REPOSITORY_ROOT / "scripts" / "profile_hotspots.py"


def _load_module():
    specification = importlib.util.spec_from_file_location("profile_workloads_contract_test", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    original = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = original
    return module


def _load_hotspot_module():
    specification = importlib.util.spec_from_file_location("profile_hotspots_contract_test", HOTSPOT_SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    original = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[specification.name] = module
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = original
        sys.modules.pop(specification.name, None)
    return module


def _valid_case(module) -> dict:
    warmups, repeats = 2, 5
    digest = "a" * 64
    return {
        "schema": module.HOTSPOT_PROFILE_SCHEMA,
        "kind": "metrics",
        "measurement": module._measurement_contract(warmups, repeats, True),
        "workload": {"size": "small", "input_digest": digest},
        "execution_input_digest": digest,
        "output_digest": digest,
        "warmup_output_digests": [digest] * warmups,
        "measured_output_digests": [digest] * repeats,
        "profiled_output_digest": digest,
        "timing_samples_seconds": [0.01] * repeats,
        "timing": {
            "minimum_seconds": 0.01,
            "median_seconds": 0.01,
            "p95_seconds": 0.01,
            "maximum_seconds": 0.01,
        },
    }


def _assert_case_rejected(module, case: dict, *, size: str = "small") -> None:
    with pytest.raises(RuntimeError):
        module._validate_case(
            case,
            size=size,
            kind="metrics",
            warmups=case["measurement"]["warmups"],
            repeats=case["measurement"]["repeats"],
            require_output_digest=case["measurement"]["require_output_digest"],
        )


def test_preregistered_hotspot_kinds_and_sizes_are_frozen() -> None:
    module = _load_module()

    assert module.WORKLOAD_KINDS == ("metrics", "rolling", "transactions", "factor", "risk", "report")
    assert module.SIZES == ("small", "medium", "large")
    assert module.WORKLOAD_PROFILE_SCHEMA == "fincore-workload-profiles-v2"
    assert module.HOTSPOT_PROFILE_SCHEMA == "fincore-hotspot-profile-v2"


def test_profiler_selects_the_explicit_pre_breaking_adapter_only_for_old_source_layouts(tmp_path: Path) -> None:
    """D0 must execute the old implementation without restoring its API in the candidate."""

    module = _load_hotspot_module()
    old_root = tmp_path / "old"
    canonical_root = tmp_path / "canonical"
    (old_root / "fincore" / "metrics").mkdir(parents=True)
    (old_root / "fincore" / "metrics" / "round_trips.py").write_text("# legacy\n", encoding="utf-8")
    (canonical_root / "fincore" / "portfolio").mkdir(parents=True)
    (canonical_root / "fincore" / "portfolio" / "round_trips.py").write_text("# canonical\n", encoding="utf-8")

    assert module._source_api_generation(old_root) == "pre_breaking"
    assert module._source_api_generation(canonical_root) == "canonical"


def test_measurement_contract_is_deterministic_and_auditable() -> None:
    module = _load_module()

    assert module._measurement_contract(2, 5, True) == {
        "warmups": 2,
        "repeats": 5,
        "require_output_digest": True,
        "timing_unit": "seconds",
        "percentile_method": "linear",
    }


def test_selection_validation_is_fail_closed() -> None:
    module = _load_module()

    with pytest.raises(ValueError):
        module._validate_selection((), ("metrics",), 0, 1)
    with pytest.raises(ValueError):
        module._validate_selection(("small",), (), 0, 1)
    with pytest.raises(ValueError):
        module._validate_selection(("small",), ("unknown",), 0, 1)
    with pytest.raises(ValueError):
        module._validate_selection(("small",), ("metrics", "metrics"), 0, 1)
    with pytest.raises(ValueError):
        module._validate_selection(("small", "small"), ("metrics",), 0, 1)
    with pytest.raises(ValueError):
        module._validate_selection(("small",), ("metrics",), -1, 1)
    with pytest.raises(ValueError):
        module._validate_selection(("small",), ("metrics",), 0, 0)
    with pytest.raises(ValueError):
        module._validate_selection(("small",), ("metrics",), True, 1)
    with pytest.raises(ValueError):
        module._validate_selection(("small",), ("metrics",), 0, True)


def test_valid_case_passes_contract_validation() -> None:
    module = _load_module()

    module._validate_case(
        _valid_case(module),
        size="small",
        kind="metrics",
        warmups=2,
        repeats=5,
        require_output_digest=True,
    )


def test_case_validation_rejects_schema_kind_and_contract_drift() -> None:
    module = _load_module()

    case = _valid_case(module)
    case["schema"] = "unexpected-schema"
    _assert_case_rejected(module, case)

    case = _valid_case(module)
    case["kind"] = "rolling"
    _assert_case_rejected(module, case)

    case = _valid_case(module)
    case["measurement"]["repeats"] = 4
    _assert_case_rejected(module, case)

    case = _valid_case(module)
    _assert_case_rejected(module, case, size="medium")


def test_case_validation_rejects_missing_or_unstable_digests() -> None:
    module = _load_module()

    case = _valid_case(module)
    case["workload"]["input_digest"] = "not-a-digest"
    _assert_case_rejected(module, case)

    case = _valid_case(module)
    case["execution_input_digest"] = "z" * 64
    _assert_case_rejected(module, case)

    case = _valid_case(module)
    case["output_digest"] = "b" * 64
    _assert_case_rejected(module, case)

    case = _valid_case(module)
    case["measured_output_digests"] = case["measured_output_digests"][:-1]
    _assert_case_rejected(module, case)

    case = _valid_case(module)
    case["warmup_output_digests"] = case["warmup_output_digests"][:-1]
    _assert_case_rejected(module, case)


def test_case_validation_rejects_invalid_timing_evidence() -> None:
    module = _load_module()

    case = _valid_case(module)
    case["timing_samples_seconds"] = [0.01] * 4
    _assert_case_rejected(module, case)

    case = _valid_case(module)
    case["timing_samples_seconds"] = [0.01, 0.01, 0.01, 0.01, -1.0]
    _assert_case_rejected(module, case)

    case = _valid_case(module)
    case["timing"].pop("p95_seconds")
    _assert_case_rejected(module, case)

    case = _valid_case(module)
    case["timing"]["median_seconds"] = 0.0
    _assert_case_rejected(module, case)


def test_cli_rejects_unknown_kinds_and_sizes_with_usage_errors() -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--kinds", "unknown_kind", "--output", "/tmp/never-written.json"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2

    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--sizes", "gigantic", "--output", "/tmp/never-written.json"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
