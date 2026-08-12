from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from fincore import Empyrical


def _returns() -> pd.Series:
    index = pd.date_range("2024-01-01", periods=30, freq="B")
    return pd.Series(np.linspace(-0.01, 0.02, len(index)), index=index)


def test_documented_instance_binding_supplies_returns() -> None:
    returns = _returns()
    emp = Empyrical(returns=returns)
    assert emp.sharpe_ratio() == Empyrical.sharpe_ratio(returns)


def test_instance_binding_supplies_returns_and_factor_returns() -> None:
    returns = _returns()
    factor_returns = returns * 0.7 + 0.001
    emp = Empyrical(returns=returns, factor_returns=factor_returns)
    assert emp.beta() == Empyrical.beta(returns, factor_returns)


def test_class_calls_still_require_explicit_data() -> None:
    with pytest.raises(TypeError):
        Empyrical.sharpe_ratio()
    with pytest.raises(TypeError):
        Empyrical.beta()


def test_instance_binding_preserves_public_callable_signatures() -> None:
    emp = Empyrical(returns=_returns())
    class_signature = inspect.signature(Empyrical.sharpe_ratio)
    instance_signature = inspect.signature(emp.sharpe_ratio)
    assert next(iter(class_signature.parameters)) == "returns"
    assert list(instance_signature.parameters) == list(class_signature.parameters)[1:]


def test_constructor_does_not_create_unused_eager_context() -> None:
    emp = Empyrical(returns=_returns())
    assert "_ctx" not in vars(emp)
