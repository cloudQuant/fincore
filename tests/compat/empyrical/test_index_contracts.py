from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import fincore.empyrical as ep
from fincore.core.context import AnalysisContext
from fincore.exceptions import DataAlignmentError
from fincore.metrics.returns import aggregate_returns


def _contract_align(*objects: pd.Series | pd.DataFrame, **kwargs: Any):
    from fincore.contracts.time_series import align_time_series

    return align_time_series(*objects, **kwargs)


def test_legacy_weekly_grouping_matches_pinned_calendar_year_policy() -> None:
    index = pd.to_datetime(["2019-12-30", "2020-01-01"])
    returns = pd.Series([0.01, 0.02], index=index)

    result = ep.aggregate_returns(returns, "weekly")

    expected_index = pd.MultiIndex.from_tuples([(2019, 1), (2020, 1)])
    pd.testing.assert_index_equal(result.index, expected_index)
    np.testing.assert_allclose(result.to_numpy(), [0.01, 0.02])


def test_enhanced_iso_week_groups_by_iso_year() -> None:
    index = pd.to_datetime(["2019-12-30", "2020-01-01"])
    returns = pd.Series([0.01, 0.02], index=index)

    result = aggregate_returns(returns, "weekly", week_year="iso")

    expected_index = pd.MultiIndex.from_tuples([(2020, 1)])
    pd.testing.assert_index_equal(result.index, expected_index)
    np.testing.assert_allclose(result.to_numpy(), [0.0302])


@pytest.mark.parametrize(
    ("convert_to", "expected_index", "expected_values"),
    [
        ("monthly", [(2019, 12), (2020, 1), (2020, 4)], [0.01, 0.0302, -0.03]),
        ("quarterly", [(2019, 4), (2020, 1), (2020, 2)], [0.01, 0.0302, -0.03]),
        ("yearly", [2019, 2020], [0.01, -0.000706]),
    ],
)
def test_legacy_calendar_aggregation_periods(
    convert_to: str,
    expected_index: list[Any],
    expected_values: list[float],
) -> None:
    index = pd.to_datetime(["2019-12-30", "2020-01-01", "2020-01-31", "2020-04-01"])
    returns = pd.Series([0.01, 0.02, 0.01, -0.03], index=index)

    result = ep.aggregate_returns(returns, convert_to)

    if convert_to == "yearly":
        pd.testing.assert_index_equal(result.index, pd.Index(expected_index))
    else:
        pd.testing.assert_index_equal(result.index, pd.MultiIndex.from_tuples(expected_index))
    np.testing.assert_allclose(result.to_numpy(), expected_values)


def test_strict_alignment_requires_identical_labels() -> None:
    left = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    right = pd.Series([3.0, 4.0], index=pd.to_datetime(["2024-01-01", "2024-01-03"]))

    with pytest.raises(DataAlignmentError, match="strict"):
        _contract_align(left, right, policy="strict")


def test_inner_alignment_uses_sorted_partial_intersection_without_mutating_inputs() -> None:
    left = pd.Series([3.0, 1.0, 2.0], index=pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]))
    right = pd.DataFrame(
        {"factor": [20.0, 30.0, 40.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    left_before = left.copy()
    right_before = right.copy()

    left_aligned, right_aligned = _contract_align(left, right, policy="inner")

    expected_index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    pd.testing.assert_index_equal(left_aligned.index, expected_index)
    pd.testing.assert_index_equal(right_aligned.index, expected_index)
    np.testing.assert_allclose(left_aligned.to_numpy(), [2.0, 3.0])
    np.testing.assert_allclose(right_aligned["factor"].to_numpy(), [20.0, 30.0])
    pd.testing.assert_series_equal(left, left_before)
    pd.testing.assert_frame_equal(right, right_before)


def test_inner_alignment_returns_empty_objects_for_no_intersection() -> None:
    left = pd.Series([1.0], index=pd.to_datetime(["2024-01-01"]))
    right = pd.Series([2.0], index=pd.to_datetime(["2024-02-01"]))

    left_aligned, right_aligned = _contract_align(left, right, policy="inner")

    assert left_aligned.empty
    assert right_aligned.empty
    pd.testing.assert_index_equal(left_aligned.index, right_aligned.index)


def test_outer_dropna_removes_missing_labels_and_missing_values() -> None:
    left = pd.Series([1.0, np.nan, 3.0], index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    right = pd.Series([20.0, 30.0, 40.0], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))

    left_aligned, right_aligned = _contract_align(left, right, policy="outer_dropna")

    expected_index = pd.to_datetime(["2024-01-03"])
    pd.testing.assert_index_equal(left_aligned.index, expected_index)
    pd.testing.assert_index_equal(right_aligned.index, expected_index)
    np.testing.assert_allclose(left_aligned.to_numpy(), [3.0])
    np.testing.assert_allclose(right_aligned.to_numpy(), [30.0])


def test_duplicate_time_labels_are_rejected_as_ambiguous() -> None:
    duplicate_index = pd.to_datetime(["2024-01-01", "2024-01-01"])
    left = pd.Series([1.0, 2.0], index=duplicate_index)
    right = pd.Series([3.0, 4.0], index=duplicate_index)

    with pytest.raises(DataAlignmentError, match="duplicate"):
        _contract_align(left, right, policy="strict")


def test_mixed_naive_and_aware_indices_fail_by_default_on_enhanced_alignment() -> None:
    naive = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2))
    aware = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2, tz="UTC"))

    with pytest.raises(DataAlignmentError, match="timezone"):
        _contract_align(naive, aware, policy="inner")


def test_explicit_utc_normalization_aligns_naive_utc_and_asia_shanghai_inputs() -> None:
    utc_index = pd.date_range("2024-01-01", periods=2, tz="UTC")
    naive = pd.Series([1.0, 2.0], index=utc_index.tz_localize(None))
    shanghai = pd.Series([3.0, 4.0], index=utc_index.tz_convert("Asia/Shanghai"))

    naive_aligned, shanghai_aligned = _contract_align(
        naive,
        shanghai,
        policy="strict",
        normalize_tz="UTC",
    )

    pd.testing.assert_index_equal(naive_aligned.index, utc_index)
    pd.testing.assert_index_equal(shanghai_aligned.index, utc_index)


def test_timezone_option_is_validated_before_inspecting_index_type() -> None:
    values = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError, match="only 'UTC'"):
        _contract_align(values, policy="strict", normalize_tz="Asia/Shanghai")


def test_utc_normalization_handles_dst_transition_by_instant() -> None:
    eastern_index = pd.date_range("2024-03-10 01:00", periods=3, freq="h", tz="America/New_York")
    utc_index = eastern_index.tz_convert("UTC")
    eastern = pd.Series([1.0, 2.0, 3.0], index=eastern_index)
    utc = pd.Series([4.0, 5.0, 6.0], index=utc_index)

    eastern_aligned, utc_aligned = _contract_align(
        eastern,
        utc,
        policy="strict",
        normalize_tz="UTC",
    )

    pd.testing.assert_index_equal(eastern_aligned.index, utc_index)
    pd.testing.assert_index_equal(utc_aligned.index, utc_index)


def test_analysis_context_rejects_mixed_timezones_by_default() -> None:
    returns = pd.Series([0.01, 0.02, -0.01], index=pd.date_range("2024-01-01", periods=3))
    factor_returns = pd.Series(
        [0.005, 0.01, -0.005],
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
    )

    with pytest.raises(DataAlignmentError, match="timezone"):
        _ = AnalysisContext(returns, factor_returns=factor_returns).beta


def test_analysis_context_preserves_partial_same_timezone_inputs() -> None:
    returns = pd.Series(
        [0.01, 0.02],
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    factor_returns = pd.Series(
        [0.005, 0.01],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    context = AnalysisContext(returns, factor_returns=factor_returns)

    assert context._returns is not returns
    assert context._factor_returns is not factor_returns
    pd.testing.assert_series_equal(context._returns, returns)
    pd.testing.assert_series_equal(context._factor_returns, factor_returns)


def test_analysis_context_accepts_explicit_utc_normalization() -> None:
    returns = pd.Series([0.01, 0.02, -0.01], index=pd.date_range("2024-01-01", periods=3))
    factor_returns = pd.Series(
        [0.005, 0.01, -0.005],
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
    )

    context = AnalysisContext(returns, factor_returns=factor_returns, normalize_tz="UTC")

    assert context.beta == pytest.approx(2.0)


def test_analysis_context_validates_timezone_across_all_time_indexed_inputs() -> None:
    utc_index = pd.date_range("2024-01-01", periods=3, tz="UTC")
    returns = pd.Series([0.01, 0.02, -0.01], index=utc_index)
    factor_returns = pd.Series([0.005, 0.01, -0.005], index=utc_index)
    positions = pd.DataFrame(
        {"A": [1.0, 1.0, 1.0], "cash": [0.0, 0.0, 0.0]},
        index=utc_index.tz_convert("Asia/Shanghai"),
    )
    transactions = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0],
            "price": [10.0, 11.0, 12.0],
            "symbol": ["A", "A", "A"],
        },
        index=utc_index.tz_localize(None),
    )

    with pytest.raises(DataAlignmentError, match="timezone"):
        AnalysisContext(
            returns,
            factor_returns=factor_returns,
            positions=positions,
            transactions=transactions,
        )


def test_analysis_context_normalizes_all_time_indexed_inputs_to_utc() -> None:
    utc_index = pd.date_range("2024-01-01", periods=3, tz="UTC")
    returns = pd.Series([0.01, 0.02, -0.01], index=utc_index)
    factor_returns = pd.Series([0.005, 0.01, -0.005], index=utc_index)
    positions = pd.DataFrame(
        {"A": [1.0, 1.0, 1.0], "cash": [0.0, 0.0, 0.0]},
        index=utc_index.tz_convert("Asia/Shanghai"),
    )
    transactions = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0],
            "price": [10.0, 11.0, 12.0],
            "symbol": ["A", "A", "A"],
        },
        index=utc_index.tz_localize(None),
    )

    context = AnalysisContext(
        returns,
        factor_returns=factor_returns,
        positions=positions,
        transactions=transactions,
        normalize_tz="UTC",
    )

    for value in (context._returns, context._factor_returns, context._positions, context._transactions):
        pd.testing.assert_index_equal(value.index, utc_index)


@pytest.mark.parametrize("normalize_tz", [None, "UTC"])
def test_analysis_context_preserves_duplicate_transaction_timestamps(
    normalize_tz: str | None,
) -> None:
    utc_index = pd.date_range("2024-01-01", periods=2, tz="UTC")
    returns = pd.Series([0.01, 0.02], index=utc_index)
    duplicate_index = pd.DatetimeIndex([utc_index[0], utc_index[0]])
    transactions = pd.DataFrame(
        {
            "amount": [2.0, 1.0],
            "price": [10.0, 11.0],
            "symbol": ["A", "A"],
            "sequence": ["first", "second"],
        },
        index=duplicate_index,
    )

    context = AnalysisContext(
        returns,
        transactions=transactions,
        normalize_tz=normalize_tz,
    )

    pd.testing.assert_index_equal(context._transactions.index, duplicate_index)
    assert context._transactions["sequence"].tolist() == ["first", "second"]
    assert context._transactions is not transactions
    pd.testing.assert_frame_equal(context._transactions, transactions)


def test_legacy_mixed_timezone_alignment_keeps_pinned_exception_surface() -> None:
    returns = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2))
    factor_returns = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2, tz="UTC"))

    with pytest.raises(TypeError, match="tz-naive"):
        ep.beta(returns, factor_returns)
