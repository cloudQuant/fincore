"""Branch-completion tests for contract-registry and pyfolio wrapper validation."""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest


def test_signature_from_static_text_rejects_unknown() -> None:
    from fincore.contracts.factor_analysis import _signature_from_static_text

    with pytest.raises(ValueError, match="static registry"):
        _signature_from_static_text("def not_in_registry(x): ...")


def test_workflow_variants_rejects_non_signature() -> None:
    from fincore.contracts.factor_workflows import _workflow_variants

    with pytest.raises(TypeError, match="Signature"):
        _workflow_variants("not a signature")  # type: ignore[arg-type]


def test_workflow_variants_without_by_group() -> None:
    from fincore.contracts.factor_workflows import _workflow_variants

    sig = inspect.Signature([inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)])
    assert _workflow_variants(sig) == ()


def test_pyfolio_make_strict_wrapper_unknown_manifest_key() -> None:
    from fincore.contracts.workflows import WORKFLOW_REGISTRY
    from fincore.pyfolio import _make_strict_wrapper

    spec = next(
        spec
        for (surface, _name, variant), spec in WORKFLOW_REGISTRY.items()
        if surface == "pyfolio_module" and variant == "strict-0.9.6"
    )
    bad_spec = replace(spec, signature_manifest_key="no-such-key")
    with pytest.raises(KeyError, match="manifest key"):
        _make_strict_wrapper(bad_spec)


def test_pyfolio_make_strict_wrapper_name_mismatch() -> None:
    from fincore.contracts.workflows import WORKFLOW_REGISTRY
    from fincore.pyfolio import _make_strict_wrapper

    spec = next(
        spec
        for (surface, _name, variant), spec in WORKFLOW_REGISTRY.items()
        if surface == "pyfolio_module" and variant == "strict-0.9.6"
    )
    bad_spec = replace(spec, public_name="some_other_name")
    with pytest.raises(ValueError, match="does not match"):
        _make_strict_wrapper(bad_spec)
