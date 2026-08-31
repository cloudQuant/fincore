"""Contracts for runtime-owned generic validation and numerical primitives."""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest


def test_validate_finite_array_copies_input_and_rejects_nonfinite_or_wrong_dimension() -> None:
    from fincore.runtime.validation import validate_finite_array

    source = np.array([1.0, 2.0])
    validated = validate_finite_array(source, name="returns", ndim=1)
    source[0] = 99.0

    assert validated.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="finite"):
        validate_finite_array([1.0, np.nan], name="returns")
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_finite_array([[1.0]], name="returns", ndim=1)


def test_numpy_backend_only_exposes_generic_dense_operations() -> None:
    from fincore.runtime.backends import NumPyBackend

    backend = NumPyBackend()

    assert backend.name == "numpy"
    assert backend.cumulative_product([2.0, 3.0]).tolist() == [2.0, 6.0]
    assert backend.sample_standard_deviation([1.0, 3.0]) == pytest.approx(np.sqrt(2.0))


def test_optional_module_loader_preserves_lazy_success_and_structured_missing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fincore.exceptions import DependencyError
    from fincore.runtime import validation

    loaded = ModuleType("optional_fixture")
    monkeypatch.setattr(validation.importlib, "import_module", lambda name: loaded)

    assert validation.load_optional_module("optional_fixture", dependency="fixture") is loaded

    def missing(name: str) -> ModuleType:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(validation.importlib, "import_module", missing)
    with pytest.raises(DependencyError) as error:
        validation.load_optional_module(
            "optional_fixture",
            dependency="fixture",
            extra="fixture-extra",
            message="fixture dependency is required",
        )
    assert error.value.dependency == "fixture"
    assert error.value.extra == "fixture-extra"


def test_optional_module_loader_converts_a_broken_optional_sdk_to_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fincore.exceptions import DependencyError
    from fincore.runtime import validation

    def broken(name: str) -> ModuleType:
        raise AttributeError(f"{name} is installed but broken")

    monkeypatch.setattr(validation.importlib, "import_module", broken)

    with pytest.raises(DependencyError, match="fixture is unavailable") as error:
        validation.load_optional_module(
            "optional_fixture",
            dependency="fixture",
            extra="fixture-extra",
            message="fixture is unavailable",
        )

    assert isinstance(error.value.__cause__, AttributeError)
