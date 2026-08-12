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


@pytest.mark.parametrize(
    ("window", "error_type", "message"),
    [
        (0, ValueError, "0-length window"),
        (-1, ValueError, "negative dimensions"),
    ],
)
def test_unary_factory_invalid_window_matches_pinned_exception(
    window: int,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        ep.roll_max_drawdown(RETURNS, window)


@pytest.mark.parametrize("window", [0, -1])
def test_binary_factory_invalid_window_returns_empty_or_fills_out(window: int) -> None:
    assert ep.roll_beta(RETURNS, FACTOR_RETURNS, window).shape == (0,)
    out = np.full(2, 999.0)

    result = ep.roll_beta(RETURNS, FACTOR_RETURNS, window, out=out)

    assert result is out
    assert np.isnan(out).all()


def test_unary_empty_input_ignores_supplied_out_like_pinned_factory() -> None:
    out = np.full(1, 999.0)

    result = ep.roll_sharpe_ratio(np.array([]), window=3, out=out)

    assert isinstance(result, np.ndarray)
    assert result.shape == (0,)
    np.testing.assert_allclose(out, [999.0])


def test_unary_empty_series_returns_empty_series_with_same_index_type() -> None:
    returns = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)

    result = ep.roll_max_drawdown(returns, window=3)

    assert isinstance(result, pd.Series)
    pd.testing.assert_index_equal(result.index, returns.index)


def test_binary_empty_input_fills_supplied_out_with_nan() -> None:
    out = np.full(2, 999.0)

    result = ep.roll_beta(np.array([]), np.array([]), window=3, out=out)

    assert result is out
    assert np.isnan(out).all()


def test_binary_factory_unequal_lengths_keeps_pinned_broadcast_error() -> None:
    with pytest.raises(ValueError, match="broadcast"):
        ep.roll_beta(RETURNS, FACTOR_RETURNS[:4], window=3)


def test_binary_factory_unequal_lengths_use_independent_window_widths() -> None:
    lhs = np.array([0.1, -0.05, 0.02])
    rhs = np.array([0.05, -0.02])

    result = ep.roll_beta(lhs, rhs, window=2)

    np.testing.assert_allclose(result, [2.1428571428571432, -1.0])


def test_binary_factory_misaligned_series_uses_positional_windows_and_left_labels() -> None:
    left_index = pd.date_range("2024-01-01", periods=4)
    right_index = pd.date_range("2024-01-02", periods=4)
    returns = pd.Series([0.01, 0.02, 0.03, 0.04], index=left_index)
    factor_returns = pd.Series([0.005, 0.01, 0.015, 0.02], index=right_index)

    result = ep.roll_beta(returns, factor_returns, window=2)

    pd.testing.assert_index_equal(result.index, left_index[1:])
    np.testing.assert_allclose(result, [2.0, 2.0, 2.0])


def test_special_roll_alpha_beta_keeps_legacy_outer_series_alignment() -> None:
    left_index = pd.date_range("2024-01-01", periods=4)
    right_index = pd.date_range("2024-01-02", periods=4)
    returns = pd.Series([0.01, 0.02, 0.03, 0.04], index=left_index)
    factor_returns = pd.Series([0.005, 0.01, 0.015, 0.02], index=right_index)

    result = ep.roll_alpha_beta(returns, factor_returns, window=2)

    pd.testing.assert_index_equal(result.index, right_index)
    np.testing.assert_allclose(
        result.to_numpy(),
        [[np.nan, np.nan], [11.274002099240244, 2.0], [11.274002099240244, 2.0], [np.nan, np.nan]],
        equal_nan=True,
    )


def test_strict_roll_max_drawdown_matches_pinned_nan_window_values() -> None:
    returns = np.array([-0.02, 0.01, np.nan, -0.01, 0.04, 0.02])

    result = ep.roll_max_drawdown(returns, window=3)

    np.testing.assert_allclose(result, [-0.02, -0.01, -0.01, -0.01])


def test_strict_roll_sharpe_matches_pinned_infinite_window_values() -> None:
    returns = np.array([0.01, np.inf, 0.02, -0.01])

    result = ep.roll_sharpe_ratio(returns, window=2)

    np.testing.assert_allclose(result, [np.nan, np.nan, 3.7416573867739413], equal_nan=True)


@pytest.mark.parametrize("name", ["roll_up_capture", "roll_down_capture", "roll_up_down_capture"])
def test_capture_roll_rejects_mixed_input_types(name: str) -> None:
    returns = pd.Series(RETURNS)

    with pytest.raises(ValueError, match="not the same"):
        getattr(ep, name)(returns, FACTOR_RETURNS, window=3)


@pytest.mark.parametrize("name", ["roll_up_capture", "roll_down_capture", "roll_up_down_capture"])
def test_capture_roll_misaligned_series_keeps_pinned_indexing_error(name: str) -> None:
    returns = pd.Series(RETURNS[:4], index=pd.date_range("2024-01-01", periods=4))
    factor_returns = pd.Series(FACTOR_RETURNS[:4], index=pd.date_range("2024-01-02", periods=4))

    with pytest.raises(pd.errors.IndexingError, match="Unalignable"):
        getattr(ep, name)(returns, factor_returns, window=2, period="weekly")


@pytest.mark.parametrize("name", ["roll_up_capture", "roll_down_capture", "roll_up_down_capture"])
def test_capture_roll_invalid_windows_keep_utils_roll_shapes(name: str) -> None:
    function = getattr(ep, name)

    assert function(RETURNS, FACTOR_RETURNS, window=0).shape == (len(RETURNS) + 1,)
    assert function(RETURNS, FACTOR_RETURNS, window=-1).shape == (len(RETURNS) + 2,)


@pytest.mark.parametrize("name", ["roll_up_capture", "roll_down_capture", "roll_up_down_capture"])
def test_capture_roll_short_series_returns_empty_series(name: str) -> None:
    index = pd.date_range("2024-01-01", periods=4)
    returns = pd.Series(RETURNS[:4], index=index)
    factor_returns = pd.Series(FACTOR_RETURNS[:4], index=index)

    result = getattr(ep, name)(returns, factor_returns, window=10)

    assert isinstance(result, pd.Series)
    assert isinstance(result.index, type(index))
    assert result.index.empty


def test_capture_roll_negative_series_window_keeps_pinned_duplicate_labels() -> None:
    index = pd.date_range("2024-01-01", periods=4)
    returns = pd.Series(RETURNS[:4], index=index)
    factor_returns = pd.Series(FACTOR_RETURNS[:4], index=index)

    result = ep.roll_up_capture(returns, factor_returns, window=-1)

    pd.testing.assert_index_equal(result.index, index.take([2, 3, 0, 1, 2, 3]))
    assert len(result) == 6


def test_capture_roll_forwards_period_kwargs_to_scalar_capture() -> None:
    returns = np.array([0.01, 0.02, 0.03, 0.04])
    factor_returns = np.array([0.005, 0.01, 0.015, 0.02])

    daily = ep.roll_up_capture(returns, factor_returns, window=4, period="daily")
    weekly = ep.roll_up_capture(returns, factor_returns, window=4, period="weekly")

    np.testing.assert_allclose(daily, [22.73150686582397])
    np.testing.assert_allclose(weekly, [2.8686747164032877])
