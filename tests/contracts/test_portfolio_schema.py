from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from fincore.exceptions import DataAlignmentError, NumericalError, ValidationError


def _validation_module():
    return importlib.import_module("fincore.contracts.validation")


def test_positions_schema_requires_cash_under_the_explicit_net_asset_convention() -> None:
    positions = pd.DataFrame({"AAA": [100.0]}, index=pd.date_range("2024-01-01", periods=1))

    with pytest.raises(ValidationError, match="cash"):
        _validation_module().validate_positions_schema(positions, require_cash=True)


def test_positions_schema_rejects_duplicate_columns_and_non_finite_values() -> None:
    index = pd.date_range("2024-01-01", periods=1)
    duplicate = pd.DataFrame([[100.0, 0.0]], columns=["cash", "cash"], index=index)
    non_finite = pd.DataFrame({"AAA": [np.inf], "cash": [0.0]}, index=index)

    with pytest.raises(ValidationError, match="duplicate"):
        _validation_module().validate_positions_schema(duplicate)
    with pytest.raises(NumericalError, match="finite"):
        _validation_module().validate_positions_schema(non_finite)


def test_positions_schema_classifies_nullable_numeric_missing_values_as_non_finite() -> None:
    positions = pd.DataFrame(
        {
            "AAA": pd.array([1, pd.NA], dtype="Int64"),
            "cash": pd.array([1, 1], dtype="Int64"),
        },
        index=pd.date_range("2024-01-01", periods=2),
    )

    with pytest.raises(NumericalError, match="finite"):
        _validation_module().validate_positions_schema(positions)


@pytest.mark.parametrize("kind", ["unsorted", "duplicate"])
def test_positions_schema_rejects_ambiguous_date_index(kind: str) -> None:
    index = pd.date_range("2024-01-01", periods=3, tz="UTC")
    index = index[[1, 0, 2]] if kind == "unsorted" else pd.DatetimeIndex([index[0], index[0], index[2]])
    positions = pd.DataFrame({"AAA": [100.0, 101.0, 102.0], "cash": [50.0] * 3}, index=index)

    with pytest.raises(DataAlignmentError, match="sorted|duplicate"):
        _validation_module().validate_positions_schema(positions)


def test_stacked_positions_schema_rejects_duplicate_date_asset_keys() -> None:
    timestamp = pd.Timestamp("2024-01-01")
    index = pd.MultiIndex.from_tuples(
        [(timestamp, "AAA"), (timestamp, "AAA")],
        names=["dt", "ticker"],
    )
    positions = pd.Series([100.0, 101.0], index=index)

    with pytest.raises(DataAlignmentError, match="duplicate"):
        _validation_module().validate_positions_schema(positions)


def test_stacked_positions_schema_normalizes_named_datetime_level_to_utc() -> None:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2), ["AAA"]],
        names=["dt", "ticker"],
    )
    positions = pd.Series([100.0, 101.0], index=index)

    actual = _validation_module().validate_positions_schema(positions, normalize_tz="UTC")

    assert str(actual.index.levels[0].tz) == "UTC"
    assert actual.index.names == ["dt", "ticker"]


def test_transactions_schema_preserves_duplicate_timestamps_and_rows() -> None:
    timestamp = pd.Timestamp("2024-01-01 10:00", tz="UTC")
    transactions = pd.DataFrame(
        {
            "amount": [2.0, -1.0],
            "price": [10.0, 11.0],
            "symbol": ["AAA", "AAA"],
        },
        index=pd.DatetimeIndex([timestamp, timestamp]),
    )

    actual = _validation_module().validate_transactions_schema(transactions)

    assert actual is not transactions
    pd.testing.assert_frame_equal(actual, transactions)


def test_transactions_schema_allows_duplicate_timestamps_but_requires_monotonic_order() -> None:
    index = pd.to_datetime(["2024-01-02 10:00", "2024-01-01 10:00"], utc=True)
    transactions = pd.DataFrame(
        {"amount": [2.0, -1.0], "price": [10.0, 11.0], "symbol": ["AAA", "AAA"]},
        index=index,
    )

    with pytest.raises(DataAlignmentError, match="sorted"):
        _validation_module().validate_transactions_schema(transactions)


@pytest.mark.parametrize("missing", ["amount", "price", "symbol"])
def test_transactions_schema_requires_canonical_columns(missing: str) -> None:
    transactions = pd.DataFrame(
        {"amount": [1.0], "price": [10.0], "symbol": ["AAA"]},
        index=pd.date_range("2024-01-01", periods=1),
    ).drop(columns=missing)

    with pytest.raises(ValidationError, match="required"):
        _validation_module().validate_transactions_schema(transactions)


def test_transactions_schema_rejects_non_finite_amount_and_price() -> None:
    transactions = pd.DataFrame(
        {"amount": [np.nan], "price": [10.0], "symbol": ["AAA"]},
        index=pd.date_range("2024-01-01", periods=1),
    )

    with pytest.raises(NumericalError, match="finite"):
        _validation_module().validate_transactions_schema(transactions)


def test_market_data_schema_requires_matching_price_and_volume_frames() -> None:
    index = pd.date_range("2024-01-01", periods=2)
    price = pd.DataFrame({"AAA": [10.0, 11.0]}, index=index)
    volume = pd.DataFrame({"BBB": [100.0, 110.0]}, index=index)

    with pytest.raises(DataAlignmentError, match="columns"):
        _validation_module().validate_market_data_schema({"price": price, "volume": volume})


@pytest.mark.parametrize("defect", ["duplicate_columns", "non_finite", "negative_volume"])
def test_market_data_schema_rejects_invalid_panels(defect: str) -> None:
    index = pd.date_range("2024-01-01", periods=2, tz="UTC")
    price = pd.DataFrame({"AAA": [10.0, 11.0]}, index=index)
    volume = pd.DataFrame({"AAA": [100.0, 110.0]}, index=index)
    if defect == "duplicate_columns":
        price = pd.DataFrame([[10.0, 11.0], [12.0, 13.0]], columns=["AAA", "AAA"], index=index)
        volume = pd.DataFrame([[100.0, 110.0], [120.0, 130.0]], columns=["AAA", "AAA"], index=index)
    elif defect == "non_finite":
        price.iloc[0, 0] = np.inf
    else:
        volume.iloc[0, 0] = -1.0

    expected = ValidationError if defect != "non_finite" else NumericalError
    with pytest.raises(expected, match="duplicate|finite|negative"):
        _validation_module().validate_market_data_schema({"price": price, "volume": volume})


def test_factor_loadings_schema_requires_named_numeric_columns() -> None:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=1), ["AAA"]],
        names=["dt", "ticker"],
    )
    loadings = pd.DataFrame({"value": ["not-numeric"]}, index=index)

    with pytest.raises(ValidationError, match="numeric"):
        _validation_module().validate_factors_schema(loadings)


def test_factor_loadings_schema_normalizes_datetime_multiindex_level_to_utc() -> None:
    dates = pd.date_range("2024-01-01", periods=2)
    index = pd.MultiIndex.from_product([dates, ["AAA"]], names=["dt", "ticker"])
    loadings = pd.DataFrame({"value": [1.0, 2.0]}, index=index)

    actual = _validation_module().validate_factors_schema(loadings, normalize_tz="UTC")

    assert isinstance(actual.index.levels[0], pd.DatetimeIndex)
    assert str(actual.index.levels[0].tz) == "UTC"
    assert actual.index.names == ["dt", "ticker"]


def test_factor_loadings_schema_requires_datetime_first_multiindex_level() -> None:
    index = pd.MultiIndex.from_product([[1, 2], ["AAA"]], names=["dt", "ticker"])
    loadings = pd.DataFrame({"value": [1.0, 2.0]}, index=index)

    with pytest.raises(ValidationError, match="datetime"):
        _validation_module().validate_factors_schema(loadings)


def test_context_requires_exact_positions_date_overlap() -> None:
    returns = pd.Series(
        [0.01, 0.02],
        index=pd.to_datetime(["2024-01-01", "2024-01-03"], utc=True),
    )
    positions = pd.DataFrame(
        {"AAA": [100.0, 101.0], "cash": [50.0, 49.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"], utc=True),
    )

    with pytest.raises(DataAlignmentError, match="overlap"):
        _validation_module().validate_context_inputs(returns=returns, positions=positions)


def test_context_transactions_overlap_returns_by_calendar_day() -> None:
    returns = pd.Series([0.01], index=pd.to_datetime(["2024-01-01"], utc=True))
    transactions = pd.DataFrame(
        {"amount": [1.0], "price": [10.0], "symbol": ["AAA"]},
        index=pd.to_datetime(["2024-01-01 15:30"], utc=True),
    )

    snapshot = _validation_module().validate_context_inputs(returns=returns, transactions=transactions)

    pd.testing.assert_frame_equal(snapshot.transactions, transactions)


@pytest.mark.parametrize("mixed_input", ["factor_returns", "positions", "transactions"])
def test_context_rejects_mixed_timezones_across_every_auxiliary_input(mixed_input: str) -> None:
    returns = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2))
    aware = pd.date_range("2024-01-01", periods=2, tz="UTC")
    inputs: dict[str, object] = {}
    if mixed_input == "factor_returns":
        inputs[mixed_input] = pd.Series([0.005, 0.006], index=aware)
    elif mixed_input == "positions":
        inputs[mixed_input] = pd.DataFrame({"AAA": [1.0, 2.0], "cash": [1.0, 1.0]}, index=aware)
    else:
        inputs[mixed_input] = pd.DataFrame(
            {"amount": [1.0], "price": [10.0], "symbol": ["AAA"]},
            index=pd.DatetimeIndex([aware[0] + pd.Timedelta(hours=1)]),
        )

    with pytest.raises(DataAlignmentError, match="timezone"):
        _validation_module().validate_context_inputs(returns=returns, **inputs)


def test_context_detects_timezone_mismatch_in_stacked_positions_datetime_level() -> None:
    returns = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2, tz="UTC"))
    stacked = pd.Series(
        [100.0, 101.0],
        index=pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=2, tz="Asia/Shanghai"), ["AAA"]],
            names=["dt", "ticker"],
        ),
    )

    with pytest.raises(DataAlignmentError, match="timezone"):
        _validation_module().validate_context_inputs(returns=returns, positions=stacked)


def test_context_explicitly_rejects_stacked_positions_instead_of_skipping_overlap() -> None:
    returns = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2, tz="UTC"))
    stacked = pd.Series(
        [100.0, 101.0],
        index=pd.MultiIndex.from_product(
            [pd.date_range("2025-01-01", periods=2, tz="Asia/Shanghai"), ["AAA"]],
            names=["dt", "ticker"],
        ),
    )

    with pytest.raises(ValidationError, match="wide DataFrame"):
        _validation_module().validate_context_inputs(
            returns=returns,
            positions=stacked,
            normalize_tz="UTC",
        )
