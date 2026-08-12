from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fincore.empyrical as ep
from fincore.metrics.rolling import rolling_volatility

RETURNS = np.array([-0.02, 0.01, 0.03, -0.01, 0.04, 0.02])
FACTOR_RETURNS = np.array([-0.01, 0.02, 0.01, -0.02, 0.03, 0.01])


def test_short_roll_alpha_beta_uses_one_full_length_factory_window() -> None:
    returns = np.arange(6, dtype=float) / 100
    factor_returns = returns / 2

    result = ep.roll_alpha_beta(returns, factor_returns)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, [[0.0, 2.0]])


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("roll_beta", (RETURNS, FACTOR_RETURNS), [1.057692307692]),
        ("roll_annual_volatility", (RETURNS,), [0.367749915024]),
        ("roll_max_drawdown", (RETURNS,), [-0.02]),
        ("roll_sharpe_ratio", (RETURNS,), [7.99456337008225]),
        ("roll_sortino_ratio", (RETURNS,), [20.28792744466521]),
    ],
)
def test_short_factory_rolls_use_minimum_of_length_and_window(
    name: str,
    args: tuple[np.ndarray, ...],
    expected: list[float],
) -> None:
    result = getattr(ep, name)(*args, window=10)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("name", ["roll_up_capture", "roll_down_capture", "roll_up_down_capture"])
def test_short_capture_family_keeps_legacy_empty_result(name: str) -> None:
    result = getattr(ep, name)(RETURNS, FACTOR_RETURNS, window=10)

    assert isinstance(result, np.ndarray)
    assert result.shape == (0,)


@pytest.mark.parametrize(
    ("window", "expected_rows"),
    [(6, 1), (3, 4)],
)
def test_exact_and_long_enough_roll_alpha_beta_shapes(window: int, expected_rows: int) -> None:
    result = ep.roll_alpha_beta(RETURNS, FACTOR_RETURNS, window=window)

    assert isinstance(result, np.ndarray)
    assert result.shape == (expected_rows, 2)


def test_short_factory_series_uses_last_input_label() -> None:
    index = pd.date_range("2024-01-01", periods=len(RETURNS), freq="D")
    returns = pd.Series(RETURNS, index=index)
    factor_returns = pd.Series(FACTOR_RETURNS, index=index)

    result = ep.roll_beta(returns, factor_returns, window=10)

    assert isinstance(result, pd.Series)
    pd.testing.assert_index_equal(result.index, index[-1:])
    np.testing.assert_allclose(result.to_numpy(), [1.057692307692])


def test_short_roll_alpha_beta_series_returns_two_column_frame() -> None:
    index = pd.date_range("2024-01-01", periods=len(RETURNS), freq="D")
    returns = pd.Series(RETURNS, index=index)
    factor_returns = pd.Series(FACTOR_RETURNS, index=index)

    result = ep.roll_alpha_beta(returns, factor_returns, window=10)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 2)
    pd.testing.assert_index_equal(result.index, index[-1:])
    np.testing.assert_allclose(result.to_numpy(), [[2.191213109003, 1.057692307692]])


def test_short_binary_factory_writes_and_returns_supplied_out_buffer() -> None:
    out = np.full(1, 999.0)

    result = ep.roll_beta(RETURNS, FACTOR_RETURNS, window=10, out=out)

    assert result is out
    np.testing.assert_allclose(out, [1.057692307692])


def test_short_alpha_beta_accepts_out_through_legacy_kwargs() -> None:
    out = np.full((1, 2), 999.0)

    result = ep.roll_alpha_beta(RETURNS, FACTOR_RETURNS, window=10, out=out)

    assert result is out
    np.testing.assert_allclose(out, [[2.191213109003, 1.057692307692]])


def test_enhanced_rolling_api_keeps_full_pandas_shape_for_short_windows() -> None:
    index = pd.date_range("2024-01-01", periods=len(RETURNS), freq="D")
    returns = pd.Series(RETURNS, index=index)

    result = rolling_volatility(returns, rolling_vol_window=10, annualization=1)

    assert isinstance(result, pd.Series)
    pd.testing.assert_index_equal(result.index, index)
    assert result.isna().all()


def test_rolling_calls_do_not_mutate_series_inputs() -> None:
    index = pd.date_range("2024-01-01", periods=len(RETURNS), freq="D")
    returns = pd.Series(RETURNS, index=index)
    factor_returns = pd.Series(FACTOR_RETURNS, index=index)
    returns_before = returns.copy()
    factor_before = factor_returns.copy()

    ep.roll_alpha_beta(returns, factor_returns, window=3)

    pd.testing.assert_series_equal(returns, returns_before)
    pd.testing.assert_series_equal(factor_returns, factor_before)
