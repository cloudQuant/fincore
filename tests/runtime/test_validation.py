"""Contracts for runtime-owned generic validation and numerical primitives."""

from __future__ import annotations

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
