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


def test_instance_unary_positional_argument_binds_after_returns_state() -> None:
    returns = _returns()
    emp = Empyrical(returns=returns)
    assert emp.sharpe_ratio(0.1) == Empyrical.sharpe_ratio(returns, 0.1)


def test_instance_factor_positional_argument_binds_after_both_state_inputs() -> None:
    returns = _returns()
    factor_returns = returns * 0.7 + 0.001
    emp = Empyrical(returns=returns, factor_returns=factor_returns)
    assert emp.beta(0.1) == Empyrical.beta(returns, factor_returns, 0.1)


def test_instance_rolling_first_optional_positional_argument_uses_public_signature() -> None:
    returns = _returns()
    emp = Empyrical(returns=returns)
    expected = Empyrical.roll_sharpe_ratio(returns, 5)
    pd.testing.assert_series_equal(emp.roll_sharpe_ratio(5), expected)


def test_instance_rolling_multiple_optional_positionals_use_public_signature() -> None:
    returns = _returns()
    factor_returns = returns * 0.7 + 0.001
    emp = Empyrical(returns=returns, factor_returns=factor_returns)
    expected = Empyrical.roll_beta(returns, factor_returns, 5, 0.1, "weekly", 52)
    pd.testing.assert_series_equal(emp.roll_beta(5, 0.1, "weekly", 52), expected)


def test_instance_rejects_explicit_state_keywords_from_removed_public_signature() -> None:
    returns = _returns()
    factor_returns = returns * 0.7 + 0.001
    emp = Empyrical(returns=returns, factor_returns=factor_returns)
    with pytest.raises(TypeError):
        emp.sharpe_ratio(returns=returns)
    with pytest.raises(TypeError):
        emp.beta(factor_returns=factor_returns)


def test_class_calls_accept_explicit_state_keywords() -> None:
    returns = _returns()
    factor_returns = returns * 0.7 + 0.001
    assert Empyrical.sharpe_ratio(returns=returns) == Empyrical.sharpe_ratio(returns)
    assert Empyrical.beta(returns=returns, factor_returns=factor_returns) == Empyrical.beta(returns, factor_returns)


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
