"""Contract tests for the fail-closed 0042-R2 architecture checker."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

SCRIPT = Path(__file__).parents[2] / "scripts" / "check_architecture_convergence.py"


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, check=False, text=True)


def _load_checker_module() -> tuple[str, Any]:
    module_name = "fincore_0042_r2_architecture_checker_test"
    specification = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module_name, module


def _make_clean_source(tmp_path: Path) -> Path:
    """Create a small, committed source tree with measurable architecture facts."""
    source_root = tmp_path / "source"
    package = source_root / "samplepkg"
    package.mkdir(parents=True)
    (source_root / "pyproject.toml").write_text(
        """[project]\nname = \"samplepkg\"\nversion = \"0.0.0\"\n\n[project.optional-dependencies]\nviz = [\"optlib>=1\"]\n""",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text("from . import first\nAPI = 'sample-api'\n", encoding="utf-8")
    (package / "first.py").write_text(
        """try:\n    import optlib\n\n    def delayed(default=__import__(\"optlib\")):\n        import optlib as delayed_optlib\n        return default\n\n    class Deferred:\n        def method(self):\n            import optlib as method_optlib\n            return method_optlib\n\n    deferred_lambda = lambda: importlib.import_module(\"optlib\")\n    deferred_generator = (__import__(\"optlib\") for _ in __import__(\"optlib\"))\nexcept ImportError:\n    import optlib as fallback_optlib\nelse:\n    import optlib as else_optlib\nfinally:\n    import optlib as final_optlib\n\ntry:\n    import optlib as rethrown_optlib\nexcept ImportError as import_error:\n    raise import_error\n\ntry:\n    import optlib as class_rethrown_optlib\nexcept ImportError:\n    class Rethrow:\n        raise ImportError()\n\nimport importlib as importlib_alias\nfrom importlib import import_module as import_module_alias\naliased_module = importlib_alias.import_module(\"optlib\")\naliased_function = import_module_alias(\"optlib\")\nassigned_module_alias = importlib_alias.import_module\nassigned_function_alias = import_module_alias\nassigned_module_alias(\"optlib\")\nassigned_function_alias(\"optlib\")\nmodule_alias = importlib_alias\nmodule_alias.import_module(\"optlib\")\ndynamic_module_alias = __import__(\"importlib\")\ndynamic_module_alias.import_module(\"optlib\")\n\ndef shadows_importlib_alias(importlib_alias, import_module_alias):\n    return importlib_alias.import_module(\"optlib\"), import_module_alias(\"optlib\")\n\nfrom . import second\nfrom samplepkg import API, first\n\ndef distinctive(value):\n    intermediate = value + 1\n    return intermediate * 2\n\ndef duplicate_one(value):\n    temporary = value + 42\n    return temporary\n""",
        encoding="utf-8",
    )
    (package / "second.py").write_text(
        """import optlib\nimport importlib as conditional_importlib\nimport samplepkg.second\nfrom .first import duplicate_one\n\nif unknown_condition:\n    conditional_importlib = None\nconditional_importlib.import_module(\"optlib\")\ntuple_alias, ignored = (conditional_importlib.import_module, None)\ntuple_alias(\"optlib\")\n\ndef global_alias_path():\n    global conditional_importlib\n    if unknown_condition:\n        conditional_importlib = None\n    return conditional_importlib.import_module(\"optlib\")\n\ndef outer_alias_path():\n    import importlib as nonlocal_importlib\n    def inner_alias_path():\n        nonlocal nonlocal_importlib\n        if unknown_condition:\n            nonlocal_importlib = None\n        return nonlocal_importlib.import_module(\"optlib\")\n    return inner_alias_path\n\ndef duplicate_two(item):\n    other_name = item + 9\n    return other_name\n""",
        encoding="utf-8",
    )
    for command in (
        ["git", "init", "--quiet"],
        ["git", "config", "user.email", "architecture@example.invalid"],
        ["git", "config", "user.name", "Architecture Checker"],
        ["git", "add", "."],
        ["git", "commit", "--quiet", "-m", "test source"],
    ):
        completed = _run(command, cwd=source_root)
        assert completed.returncode == 0, completed.stderr
    return source_root


def _make_reduced_candidate(tmp_path: Path) -> Path:
    """Create a smaller candidate without the baseline cycle, duplication, or optional leakage."""
    source_root = tmp_path / "candidate"
    package = source_root / "samplepkg"
    package.mkdir(parents=True)
    (source_root / "pyproject.toml").write_text(
        '[project]\nname = "samplepkg"\nversion = "0.0.1"\n\n[project.optional-dependencies]\nviz = ["optlib>=1"]\n',
        encoding="utf-8",
    )
    (package / "__init__.py").write_text("from .first import distinctive\n", encoding="utf-8")
    (package / "first.py").write_text(
        "def distinctive(value):\n    return (value + 1) * 2\n",
        encoding="utf-8",
    )
    for command in (
        ["git", "init", "--quiet"],
        ["git", "config", "user.email", "architecture@example.invalid"],
        ["git", "config", "user.name", "Architecture Checker"],
        ["git", "add", "."],
        ["git", "commit", "--quiet", "-m", "reduced candidate"],
    ):
        completed = _run(command, cwd=source_root)
        assert completed.returncode == 0, completed.stderr
    return source_root


def _collect(source_root: Path, output: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(source_root),
            "--package",
            "samplepkg",
            "--capture",
            str(output),
            *arguments,
        ],
        cwd=source_root,
    )


def _commit_threshold_policy(
    source_root: Path,
    *,
    policy_state: str = "frozen",
    loc_reduction: float = 0.12,
) -> Path:
    """Add one source-tracked policy whose limits derive from clean baseline facts."""
    policy_path = source_root / "architecture-threshold-policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "artifact_type": "fincore_0042_r2_architecture_threshold_policy",
                "policy_state": policy_state,
                "rules": {
                    "max_duplicate_body_occurrences": {"kind": "relative_reduction", "reduction": 0.60},
                    "max_implementation_fingerprint_count": {"kind": "relative_reduction", "reduction": 0.0},
                    "max_internal_cycle_count": {"kind": "absolute_maximum", "maximum": 0},
                    "max_logical_loc": {"kind": "relative_reduction", "reduction": loc_reduction},
                    "max_optional_import_leakage_count": {"kind": "absolute_maximum", "maximum": 0},
                    "max_physical_loc": {"kind": "relative_reduction", "reduction": loc_reduction},
                },
                "schema_version": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for command in (
        ["git", "add", policy_path.name],
        ["git", "commit", "--quiet", "-m", "add frozen threshold policy"],
    ):
        completed = _run(command, cwd=source_root)
        assert completed.returncode == 0, completed.stderr
    return policy_path


def _seal_baseline_for_test(source_root: Path, output: Path) -> dict[str, Any]:
    policy_path = _commit_threshold_policy(source_root)
    result = _collect(
        source_root,
        output,
        "--seal-baseline",
        "--threshold-policy",
        str(policy_path),
    )
    assert result.returncode == 0, result.stderr
    return cast("dict[str, Any]", json.loads(output.read_text(encoding="utf-8")))


def test_capture_is_deterministic_and_collects_all_required_measurement_facts(tmp_path: Path) -> None:
    source_root = _make_clean_source(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_result = _collect(source_root, first)
    second_result = _collect(source_root, second)

    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr
    assert first.read_bytes() == second.read_bytes()
    artifact = json.loads(first.read_text(encoding="utf-8"))

    assert artifact["schema_version"] == 1
    assert artifact["artifact_type"] == "fincore_0042_r2_architecture_measurement"
    assert artifact["baseline_state"] == "captured"
    assert artifact["verdict"] == "measurement_only"
    assert artifact["package"] == "samplepkg"
    assert artifact["source_provenance"]["clean"] is True
    assert len(artifact["source_provenance"]["commit"]) == 40
    assert artifact["source_provenance"]["tree"]
    assert "timestamp" not in first.read_text(encoding="utf-8")
    assert str(source_root) not in first.read_text(encoding="utf-8")

    measurements = artifact["measurements"]
    assert measurements["summary"]["physical_loc"] > 0
    assert measurements["summary"]["logical_loc"] > 0
    assert measurements["summary"]["internal_cycle_count"] == 1
    assert measurements["summary"]["optional_import_leakage_count"] == 20
    assert measurements["summary"]["duplicate_body_occurrences"] >= 1
    assert [item["path"] for item in measurements["files"]] == [
        "samplepkg/__init__.py",
        "samplepkg/first.py",
        "samplepkg/second.py",
    ]
    assert measurements["optional_import_policy"]["effective_module_roots"] == ["optlib"]
    assert {fact["path"] for fact in measurements["optional_import_leakage"]} == {
        "samplepkg/first.py",
        "samplepkg/second.py",
    }
    assert len([fact for fact in measurements["optional_imports"] if fact["guarded"]]) == 3
    assert "dynamic_import" in {fact["import_kind"] for fact in measurements["optional_import_leakage"]}
    assert all(
        "samplepkg" in edge["from"] and "samplepkg" in edge["to"]
        for edge in measurements["internal_import_graph"]["edges"]
    )
    assert measurements["internal_import_graph"]["cycles"] == [["samplepkg", "samplepkg.first", "samplepkg.second"]]
    assert not any(edge["from"] == edge["to"] for edge in measurements["internal_import_graph"]["edges"])

    duplicate_groups = measurements["normalized_ast_duplication"]["groups"]
    assert any(
        {occurrence["qualname"] for occurrence in group["occurrences"]} == {"duplicate_one", "duplicate_two"}
        for group in duplicate_groups
    )
    assert {fact["qualname"] for fact in measurements["implementation_fingerprints"]} >= {
        "distinctive",
        "duplicate_one",
        "duplicate_two",
    }


def test_dynamic_optional_imports_cover_keyword_builtins_and_relative_package_spellings(tmp_path: Path) -> None:
    source_root = _make_reduced_candidate(tmp_path)
    source_path = source_root / "samplepkg" / "first.py"
    source_path.write_text(
        """import builtins
import importlib

importlib.import_module(name="optlib")
__builtins__.__import__("optlib")
__builtins__["__import__"]("optlib")
builtins.__import__(name="optlib")
importlib.import_module(".submodule", package="optlib")
""",
        encoding="utf-8",
    )
    for command in (
        ["git", "add", "samplepkg/first.py"],
        ["git", "commit", "--quiet", "-m", "exercise literal dynamic optional imports"],
    ):
        completed = _run(command, cwd=source_root)
        assert completed.returncode == 0, completed.stderr

    output = tmp_path / "dynamic-imports.json"
    result = _collect(source_root, output)

    assert result.returncode == 0, result.stderr
    measurements = json.loads(output.read_text(encoding="utf-8"))["measurements"]
    leakage = measurements["optional_import_leakage"]
    assert measurements["summary"]["optional_import_leakage_count"] == 5
    assert {fact["imported_root"] for fact in leakage} == {"optlib"}
    assert {fact["target"] for fact in leakage} == {"optlib", ".submodule"}
    assert {fact["import_kind"] for fact in leakage} == {"dynamic_import"}


def test_rejects_dirty_source_and_output_inside_source_without_overwriting(tmp_path: Path) -> None:
    source_root = _make_clean_source(tmp_path)
    output = tmp_path / "architecture.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")
    (source_root / "dirty-marker.txt").write_text("dirty\n", encoding="utf-8")

    dirty_result = _collect(source_root, output)

    assert dirty_result.returncode != 0
    assert "clean" in dirty_result.stderr.lower()
    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'

    (source_root / "dirty-marker.txt").unlink()
    inside_result = _collect(source_root, source_root / "architecture.json")

    assert inside_result.returncode != 0
    assert "outside" in inside_result.stderr.lower()
    assert not (source_root / "architecture.json").exists()


def test_capture_write_keeps_the_validated_parent_descriptor_across_a_path_swap(tmp_path: Path) -> None:
    source_root = _make_clean_source(tmp_path / "source")
    output_parent = tmp_path / "capture"
    output_parent.mkdir()
    protected_parent = tmp_path / "protected"
    protected_parent.mkdir()
    output = output_parent / "artifact.json"
    module_name, checker = _load_checker_module()
    target = checker._validate_capture_path(source_root, output)
    _, provenance = checker._source_provenance(source_root)
    original_parent = tmp_path / "capture-original"

    output_parent.rename(original_parent)
    output_parent.symlink_to(protected_parent, target_is_directory=True)
    try:
        checker._write_capture_atomically(target, b'{"bound": true}\n', source_root, provenance)
    finally:
        checker._close_capture_output(target)
        sys.modules.pop(module_name, None)

    assert (original_parent / output.name).read_bytes() == b'{"bound": true}\n'
    assert not (protected_parent / output.name).exists()


def test_baseline_validation_fails_closed_for_missing_pending_platform_schema_and_threshold_errors(
    tmp_path: Path,
) -> None:
    baseline_source_root = _make_clean_source(tmp_path / "d0")
    baseline_path = tmp_path / "baseline.json"
    baseline = _seal_baseline_for_test(baseline_source_root, baseline_path)
    candidate_root = _make_reduced_candidate(tmp_path)

    def validate(output: Path) -> subprocess.CompletedProcess[str]:
        return _collect(
            candidate_root,
            output,
            "--baseline",
            str(baseline_path),
            "--baseline-source-root",
            str(baseline_source_root),
        )

    validated_path = tmp_path / "validated.json"
    validated_result = validate(validated_path)

    assert validated_result.returncode == 0, validated_result.stderr
    assert json.loads(validated_path.read_text(encoding="utf-8"))["baseline_validation"]["status"] == "passed"

    missing = _collect(
        candidate_root,
        tmp_path / "missing.json",
        "--baseline",
        str(tmp_path / "absent.json"),
        "--baseline-source-root",
        str(baseline_source_root),
    )
    assert missing.returncode != 0
    assert "baseline" in missing.stderr.lower()

    pending = copy.deepcopy(baseline)
    pending["baseline_state"] = "pending"
    baseline_path.write_text(json.dumps(pending), encoding="utf-8")
    pending_result = validate(tmp_path / "pending.json")
    assert pending_result.returncode != 0
    assert "frozen" in pending_result.stderr.lower()

    platform_mismatch = copy.deepcopy(baseline)
    platform_mismatch["source_provenance"]["platform"]["system"] = ""
    baseline_path.write_text(json.dumps(platform_mismatch), encoding="utf-8")
    platform_result = validate(tmp_path / "platform.json")
    assert platform_result.returncode != 0
    assert "platform" in platform_result.stderr.lower()

    provenance_mismatch = copy.deepcopy(baseline)
    provenance_mismatch["source_provenance"]["platform"]["system"] = "different-platform"
    baseline_path.write_text(json.dumps(provenance_mismatch), encoding="utf-8")
    provenance_result = validate(tmp_path / "provenance.json")
    assert provenance_result.returncode != 0
    assert "baseline-source-root" in provenance_result.stderr

    schema_mismatch = copy.deepcopy(baseline)
    schema_mismatch["schema_version"] = 999
    baseline_path.write_text(json.dumps(schema_mismatch), encoding="utf-8")
    schema_result = validate(tmp_path / "schema.json")
    assert schema_result.returncode != 0
    assert "schema" in schema_result.stderr.lower()

    missing_measurements = copy.deepcopy(baseline)
    del missing_measurements["measurements"]
    baseline_path.write_text(json.dumps(missing_measurements), encoding="utf-8")
    missing_measurements_result = validate(tmp_path / "missing-measurements.json")
    assert missing_measurements_result.returncode != 0
    assert "measurements" in missing_measurements_result.stderr.lower()

    missing_tool_provenance = copy.deepcopy(baseline)
    del missing_tool_provenance["tool_provenance"]
    baseline_path.write_text(json.dumps(missing_tool_provenance), encoding="utf-8")
    missing_tool_result = validate(tmp_path / "missing-tool-provenance.json")
    assert missing_tool_result.returncode != 0
    assert "tool_provenance" in missing_tool_result.stderr.lower()

    tooling_mismatch = copy.deepcopy(baseline)
    tooling_mismatch["tool_provenance"]["script_sha256"] = "0" * 64
    baseline_path.write_text(json.dumps(tooling_mismatch), encoding="utf-8")
    tooling_mismatch_result = validate(tmp_path / "tooling-mismatch.json")
    assert tooling_mismatch_result.returncode != 0
    assert "tool_provenance" in tooling_mismatch_result.stderr

    threshold_failure = copy.deepcopy(baseline)
    threshold_failure["thresholds"]["max_logical_loc"] = 0
    baseline_path.write_text(json.dumps(threshold_failure), encoding="utf-8")
    threshold_result = validate(tmp_path / "threshold.json")
    assert threshold_result.returncode != 0
    assert "threshold" in threshold_result.stderr.lower()

    policy_mismatch = copy.deepcopy(baseline)
    policy_mismatch["threshold_policy"]["sha256"] = "0" * 64
    baseline_path.write_text(json.dumps(policy_mismatch), encoding="utf-8")
    policy_result = validate(tmp_path / "policy.json")
    assert policy_result.returncode != 0
    assert "threshold_policy" in policy_result.stderr

    summary_mismatch = copy.deepcopy(baseline)
    summary_mismatch["measurements"]["summary"]["physical_loc"] += 1
    summary_mismatch["thresholds"]["max_physical_loc"] = (
        summary_mismatch["measurements"]["summary"]["physical_loc"] * 88 // 100
    )
    baseline_path.write_text(json.dumps(summary_mismatch), encoding="utf-8")
    summary_result = validate(tmp_path / "summary.json")
    assert summary_result.returncode != 0
    assert "measurements" in summary_result.stderr

    file_mismatch = copy.deepcopy(baseline)
    file_mismatch["measurements"]["files"][0]["sha256"] = "0" * 64
    baseline_path.write_text(json.dumps(file_mismatch), encoding="utf-8")
    file_result = validate(tmp_path / "file.json")
    assert file_result.returncode != 0
    assert "measurements" in file_result.stderr

    explicit_roots_mismatch = copy.deepcopy(baseline)
    explicit_roots_mismatch["measurements"]["optional_import_policy"]["explicit_module_roots"] = ["optlib"]
    baseline_path.write_text(json.dumps(explicit_roots_mismatch), encoding="utf-8")
    explicit_roots_result = validate(tmp_path / "explicit-roots.json")
    assert explicit_roots_result.returncode != 0
    assert "explicit_module_roots" in explicit_roots_result.stderr

    tree_mismatch = copy.deepcopy(baseline)
    tree_mismatch["source_provenance"]["tree"] = "0" * 40
    baseline_path.write_text(json.dumps(tree_mismatch), encoding="utf-8")
    tree_result = validate(tmp_path / "tree.json")
    assert tree_result.returncode != 0
    assert "baseline-source-root" in tree_result.stderr

    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    unbound_candidate = _make_reduced_candidate(tmp_path / "unbound")
    (unbound_candidate / "pyproject.toml").write_text(
        '[project]\nname = "samplepkg"\nversion = "0.0.2"\n', encoding="utf-8"
    )
    for command in (
        ["git", "add", "pyproject.toml"],
        ["git", "commit", "--quiet", "-m", "drop optional dependency declaration"],
    ):
        completed = _run(command, cwd=unbound_candidate)
        assert completed.returncode == 0, completed.stderr
    unbound_result = _collect(
        unbound_candidate,
        tmp_path / "unbound.json",
        "--baseline",
        str(baseline_path),
        "--baseline-source-root",
        str(baseline_source_root),
    )
    assert unbound_result.returncode != 0
    assert "effective_module_roots" in unbound_result.stderr


def test_seal_derives_candidate_thresholds_from_a_frozen_source_policy(tmp_path: Path) -> None:
    source_root = _make_clean_source(tmp_path)
    policy_path = _commit_threshold_policy(source_root)
    sealed_path = tmp_path / "sealed-baseline.json"

    sealed_result = _collect(
        source_root,
        sealed_path,
        "--seal-baseline",
        "--threshold-policy",
        str(policy_path),
    )

    assert sealed_result.returncode == 0, sealed_result.stderr
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    assert sealed["baseline_state"] == "frozen"
    assert sealed["verdict"] == "architecture_baseline"
    assert sealed["threshold_policy"]["path"] == policy_path.name
    summary = sealed["measurements"]["summary"]
    assert sealed["thresholds"]["max_physical_loc"] == summary["physical_loc"] * 88 // 100
    assert sealed["thresholds"]["max_logical_loc"] == summary["logical_loc"] * 88 // 100
    assert sealed["thresholds"]["max_duplicate_body_occurrences"] == summary["duplicate_body_occurrences"] * 40 // 100
    assert sealed["thresholds"]["max_internal_cycle_count"] == 0
    assert sealed["thresholds"]["max_optional_import_leakage_count"] == 0

    candidate_root = _make_reduced_candidate(tmp_path)
    validated_path = tmp_path / "validated.json"
    validated_result = _collect(
        candidate_root,
        validated_path,
        "--baseline",
        str(sealed_path),
        "--baseline-source-root",
        str(source_root),
    )
    assert validated_result.returncode == 0, validated_result.stderr
    assert json.loads(validated_path.read_text(encoding="utf-8"))["baseline_validation"]["status"] == "passed"

    unsealed_path = tmp_path / "unsealed.json"
    missing_policy = _collect(source_root, unsealed_path, "--seal-baseline")
    assert missing_policy.returncode != 0
    assert "threshold-policy" in missing_policy.stderr
    assert not unsealed_path.exists()

    explicit_source_root = _make_clean_source(tmp_path / "explicit")
    explicit_policy_path = _commit_threshold_policy(explicit_source_root)
    explicit_result = _collect(
        explicit_source_root,
        tmp_path / "explicit-baseline.json",
        "--seal-baseline",
        "--threshold-policy",
        str(explicit_policy_path),
        "--optional-module",
        "other_optional",
    )
    assert explicit_result.returncode != 0
    assert "optional-module" in explicit_result.stderr

    pending_source = _make_clean_source(tmp_path / "pending")
    pending_policy = _commit_threshold_policy(pending_source, policy_state="pending")
    pending_path = tmp_path / "pending.json"
    pending_result = _collect(
        pending_source,
        pending_path,
        "--seal-baseline",
        "--threshold-policy",
        str(pending_policy),
    )
    assert pending_result.returncode != 0
    assert "frozen" in pending_result.stderr
    assert not pending_path.exists()

    nonfinite_source = _make_clean_source(tmp_path / "nonfinite")
    nonfinite_policy = _commit_threshold_policy(nonfinite_source, loc_reduction=float("nan"))
    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_result = _collect(
        nonfinite_source,
        nonfinite_path,
        "--seal-baseline",
        "--threshold-policy",
        str(nonfinite_policy),
    )
    assert nonfinite_result.returncode != 0
    assert "number" in nonfinite_result.stderr
    assert not nonfinite_path.exists()


def test_sealing_resolves_a_source_policy_path_through_a_directory_alias(tmp_path: Path) -> None:
    source_root = _make_clean_source(tmp_path / "source")
    policy_path = _commit_threshold_policy(source_root)
    source_alias = tmp_path / "source-alias"
    source_alias.symlink_to(source_root, target_is_directory=True)

    result = _collect(
        source_root,
        tmp_path / "aliased-policy-baseline.json",
        "--seal-baseline",
        "--threshold-policy",
        str(source_alias / policy_path.name),
    )

    assert result.returncode == 0, result.stderr


def test_sealing_ignores_git_replace_refs_when_binding_source_provenance(tmp_path: Path) -> None:
    source_root = _make_clean_source(tmp_path / "source")
    policy_path = _commit_threshold_policy(source_root)
    canonical_commit = _run(["git", "--no-replace-objects", "rev-parse", "HEAD"], cwd=source_root)
    canonical_tree = _run(["git", "--no-replace-objects", "rev-parse", "HEAD^{tree}"], cwd=source_root)
    assert canonical_commit.returncode == 0, canonical_commit.stderr
    assert canonical_tree.returncode == 0, canonical_tree.stderr

    first_module = source_root / "samplepkg" / "first.py"
    first_module.write_text(
        first_module.read_text(encoding="utf-8") + "\nreplacement_marker = True\n", encoding="utf-8"
    )
    for command in (
        ["git", "add", "samplepkg/first.py"],
        ["git", "commit", "--quiet", "-m", "replacement source"],
    ):
        completed = _run(command, cwd=source_root)
        assert completed.returncode == 0, completed.stderr
    replacement_commit = _run(["git", "rev-parse", "HEAD"], cwd=source_root)
    assert replacement_commit.returncode == 0, replacement_commit.stderr
    checkout = _run(["git", "checkout", "--quiet", canonical_commit.stdout.strip()], cwd=source_root)
    assert checkout.returncode == 0, checkout.stderr
    replace = _run(
        ["git", "replace", canonical_commit.stdout.strip(), replacement_commit.stdout.strip()],
        cwd=source_root,
    )
    assert replace.returncode == 0, replace.stderr
    replaced_tree = _run(["git", "rev-parse", "HEAD^{tree}"], cwd=source_root)
    assert replaced_tree.returncode == 0, replaced_tree.stderr
    assert replaced_tree.stdout.strip() != canonical_tree.stdout.strip()

    output = tmp_path / "canonical-baseline.json"
    result = _collect(
        source_root,
        output,
        "--seal-baseline",
        "--threshold-policy",
        str(policy_path),
    )

    assert result.returncode == 0, result.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["source_provenance"]["commit"] == canonical_commit.stdout.strip()
    assert artifact["source_provenance"]["tree"] == canonical_tree.stdout.strip()


def test_rejects_capture_path_that_would_replace_its_baseline_input(tmp_path: Path) -> None:
    source_root = _make_clean_source(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text("{}\n", encoding="utf-8")
    original = baseline_path.read_bytes()

    result = _collect(source_root, baseline_path, "--baseline", str(baseline_path))

    assert result.returncode != 0
    assert "replace" in result.stderr.lower()
    assert baseline_path.read_bytes() == original


def test_strict_flags_require_explicit_evidence_instead_of_claiming_task8_results(tmp_path: Path) -> None:
    source_root = _make_clean_source(tmp_path)
    no_cycles = _collect(source_root, tmp_path / "no-cycles.json", "--require-no-cycles")
    legacy_zero = _collect(
        source_root,
        tmp_path / "legacy-zero.json",
        "--require-legacy-zero",
    )
    both_strict = _collect(
        _make_reduced_candidate(tmp_path),
        tmp_path / "both-strict.json",
        "--require-no-cycles",
        "--require-legacy-zero",
    )

    assert no_cycles.returncode != 0
    assert "cycle" in no_cycles.stderr.lower()
    assert legacy_zero.returncode != 0
    assert "legacy" in legacy_zero.stderr.lower()
    assert both_strict.returncode != 0
    assert "legacy" in both_strict.stderr.lower()
