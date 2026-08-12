from __future__ import annotations

import inspect

import pytest

from .conftest import (
    load_pyfolio_profile,
    run_isolated_import_probe,
    run_isolated_workflow_dependency_probe,
)


@pytest.mark.parametrize("entry", load_pyfolio_profile(), ids=lambda entry: entry["name"])
def test_full_pyfolio_profile_has_frozen_path_and_signature(entry: dict[str, object]) -> None:
    from fincore import pyfolio

    public = getattr(pyfolio, str(entry["name"]))
    assert callable(public)
    assert str(inspect.signature(public)) == entry["signature"]
    assert "pyfolio_instance" not in inspect.signature(public).parameters
    assert "empyrical_instance" not in inspect.signature(public).parameters


@pytest.mark.parametrize("entry", load_pyfolio_profile(), ids=lambda entry: entry["name"])
def test_strict_workflow_call_binding_delegates_all_bound_arguments(
    entry: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fincore import pyfolio

    public = getattr(pyfolio, str(entry["name"]))
    signature = inspect.signature(public)
    required = [object() for parameter in signature.parameters.values() if parameter.default is inspect.Parameter.empty]
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_invoke(name: str, arguments: dict[str, object]) -> object:
        captured["name"] = name
        captured["arguments"] = arguments
        return sentinel

    monkeypatch.setattr(pyfolio, "_invoke_workflow", fake_invoke)
    assert public(*required) is sentinel
    assert captured["name"] == entry["name"]
    assert captured["arguments"] == signature.bind(*required).arguments


def test_strict_full_signature_has_no_non_pinned_flask_argument() -> None:
    from fincore import pyfolio

    parameters = inspect.signature(pyfolio.create_full_tear_sheet).parameters
    assert "set_context" in parameters
    assert "run_flask_app" not in parameters


def test_pyfolio_implementation_import_is_lazy_and_backend_neutral() -> None:
    public_names = tuple(str(entry["name"]) for entry in load_pyfolio_profile())
    result = run_isolated_import_probe("fincore.pyfolio", public_names)
    assert result.backend_unchanged
    assert result.eager_optional_modules == ()
    assert not result.private_implementation_loaded


def test_workflow_registry_uses_multi_surface_contract_schema() -> None:
    from fincore.contracts.workflows import WORKFLOW_REGISTRY, WorkflowSpec

    assert set(WorkflowSpec.__dataclass_fields__) == {
        "surface",
        "public_name",
        "variant",
        "signature_manifest_key",
        "workflow_ref",
        "adapter_ref",
        "validation_profile",
        "result_contract_key",
        "result_projection",
    }
    strict = {
        key: spec for key, spec in WORKFLOW_REGISTRY.items() if key[0] == "pyfolio_module" and key[2] == "strict-0.9.6"
    }
    assert len(strict) == 11
    assert all(key == (spec.surface, spec.public_name, spec.variant) for key, spec in strict.items())
    assert all(":" in spec.workflow_ref and ":" in spec.adapter_ref for spec in strict.values())


def test_explicit_pyfolio_class_resolution_is_backend_neutral() -> None:
    result = run_isolated_import_probe("fincore.pyfolio", ("Pyfolio",))
    assert result.backend_unchanged
    assert result.private_implementation_loaded


def test_common_utils_display_contract_remains_lazy_and_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    from fincore.utils import common_utils

    assert isinstance(common_utils.HAS_IPYTHON, bool)
    assert callable(common_utils.display)
    assert callable(common_utils.HTML)

    imported: list[str] = []
    real_import = common_utils.importlib.import_module

    def tracking_import(name: str):
        imported.append(name)
        return real_import(name)

    monkeypatch.setattr(common_utils.importlib, "import_module", tracking_import)
    html = common_utils.HTML("<b>fincore</b>")
    assert "fincore" in str(getattr(html, "data", html))
    assert imported == ["IPython.display"]


def test_common_utils_import_does_not_eagerly_load_ipython() -> None:
    result = run_isolated_import_probe("fincore.utils.common_utils")
    assert not [name for name in result.eager_optional_modules if name.startswith("IPython")]


def _missing_module(name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(f"No module named {name!r}", name=name)


def test_import_time_visual_dependency_error_names_the_viz_extra() -> None:
    result = run_isolated_workflow_dependency_probe(
        "create_returns_tear_sheet",
        "matplotlib",
    )

    assert result.error_type == "ImportError"
    assert "pip install fincore[viz]" in result.message
    assert "fincore[pyfolio]" not in result.message


def test_call_time_bayesian_dependency_error_names_both_extras() -> None:
    result = run_isolated_workflow_dependency_probe(
        "create_bayesian_tear_sheet",
        "pymc",
    )

    assert result.error_type == "ImportError"
    assert "pip install fincore[viz,bayesian]" in result.message


def test_unrelated_call_time_module_error_is_not_relabelled(monkeypatch: pytest.MonkeyPatch) -> None:
    from fincore.contracts import workflows

    spec = workflows.get_workflow_spec("pyfolio_module", "create_returns_tear_sheet", "strict-0.9.6")
    original = _missing_module("fincore.internal_missing")

    def broken_workflow(_public_name: str, _arguments: dict[str, object]) -> object:
        raise original

    def fake_resolve(reference: str):
        if reference == spec.workflow_ref:
            return broken_workflow
        return workflows.strict_pyfolio_adapter

    monkeypatch.setattr(workflows, "resolve_ref", fake_resolve)

    with pytest.raises(ModuleNotFoundError) as caught:
        workflows.invoke_workflow(spec, {})

    assert caught.value is original
