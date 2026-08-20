"""Operation catalog contract tests."""

from __future__ import annotations

from fincore._registry import METRIC_REGISTRY
from fincore.api import OperationCatalog, build_builtin_catalog
from fincore.contracts.workflows import WORKFLOW_REGISTRY


def test_catalog_contains_all_registry_entries() -> None:
    catalog = build_builtin_catalog()
    expected = len(METRIC_REGISTRY) + len(WORKFLOW_REGISTRY)
    assert len(catalog.bindings) == expected, (
        f"catalog has {len(catalog.bindings)} bindings, expected {expected}"
    )


def test_definitions_are_unique_per_profile() -> None:
    catalog = build_builtin_catalog()
    keys = [(d.operation_id, d.semantic_profile) for d in catalog.definitions]
    assert len(keys) == len(set(keys)), "duplicate operation_id+semantic_profile"


def test_public_paths_are_unique() -> None:
    catalog = build_builtin_catalog()
    paths = [b.public_path for b in catalog.bindings]
    assert len(paths) == len(set(paths)), "duplicate public_path"


def test_bindings_reference_valid_definitions() -> None:
    catalog = build_builtin_catalog()
    for binding in catalog.bindings:
        definition = catalog.resolve_definition(binding.operation_id, binding.semantic_profile)
        assert definition.operation_id == binding.operation_id


def test_definitions_are_shared_across_surfaces() -> None:
    catalog = build_builtin_catalog()
    assert len(catalog.definitions) < len(catalog.bindings), (
        "definitions must be merged across surfaces, not duplicated per binding"
    )


def test_strict_profiles_are_stable_enhanced_are_experimental() -> None:
    catalog = build_builtin_catalog()
    for definition in catalog.definitions:
        if definition.semantic_profile.startswith("strict_"):
            assert definition.stability == "stable", definition.operation_id
        else:
            assert definition.stability == "experimental", definition.operation_id


def test_catalog_is_an_immutable_dataclass() -> None:
    catalog = build_builtin_catalog()
    assert isinstance(catalog, OperationCatalog)
    try:
        catalog.definitions = ()  # type: ignore[misc]
        raise AssertionError("catalog should be frozen")
    except (AttributeError, TypeError):
        pass
