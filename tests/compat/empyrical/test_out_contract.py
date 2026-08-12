from __future__ import annotations

import numpy as np

import fincore.empyrical as ep

RETURNS = np.array([0.01, 0.02, -0.01, 0.03, 0.005, -0.004])
FACTOR = np.array([0.01, 0.01, -0.02, 0.02, 0.004, -0.002])


def test_beta_fourth_positional_argument_is_out() -> None:
    out = np.full((), 999.0)
    result = ep.beta(RETURNS, FACTOR, 0.0, out)
    assert out.item() == result
    assert out.item() != 999.0


def test_scalar_metric_out_is_mutated_and_returned_as_scalar() -> None:
    out = np.full((), 999.0)
    result = ep.sharpe_ratio(RETURNS, out=out)
    assert out.item() == result
    assert out.item() != 999.0


def test_unary_rolling_factory_writes_supplied_out_buffer() -> None:
    out = np.full(4, 999.0)
    result = ep.roll_annual_volatility(RETURNS, 3, out=out, annualization=1)
    assert result is out
    assert np.all(out != 999.0)


def test_binary_rolling_factory_writes_supplied_out_buffer() -> None:
    out = np.full(4, 999.0)
    result = ep.roll_beta(RETURNS, FACTOR, 3, out=out)
    assert result is out
    assert np.all(out != 999.0)
