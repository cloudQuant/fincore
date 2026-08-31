from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal

from fincore.exceptions import ValidationError
from fincore.portfolio.transactions import make_transaction_frame

EXPECTED_COLUMNS = [
    "dt",
    "sid",
    "symbol",
    "amount",
    "price",
    "order_id",
    "commission",
    "txn_dollars",
]
REQUIRED_INPUT_COLUMNS = [
    "dt",
    "sid",
    "amount",
    "price",
    "order_id",
    "commission",
]


def _legacy_transactions() -> list[dict[str, Any]]:
    first_dt = pd.Timestamp("2024-06-03 14:30:00", tz="UTC")
    second_dt = pd.Timestamp("2024-06-04 15:45:00", tz="UTC")
    return [
        {
            "dt": first_dt,
            "sid": {"sid": 101, "symbol": "AAA"},
            "amount": 10.0,
            "price": 12.5,
            "order_id": "order-a",
            "commission": 0.25,
        },
        {
            # Duplicate execution timestamps are valid and must not collapse.
            "dt": first_dt,
            "sid": {"sid": 202, "symbol": "BBB"},
            "amount": -4.0,
            "price": 25.0,
            "order_id": "order-b",
            "commission": 0.5,
        },
        {
            "dt": second_dt,
            "sid": {"sid": 101, "symbol": "AAA"},
            "amount": -2.0,
            "price": 13.0,
            "order_id": "order-c",
            "commission": 0.1,
        },
    ]


def _canonical_transactions() -> pd.DataFrame:
    rows = []
    for transaction in _legacy_transactions():
        nested_sid = transaction["sid"]
        rows.append(
            {
                "dt": transaction["dt"],
                "sid": nested_sid["sid"],
                "symbol": nested_sid["symbol"],
                "amount": transaction["amount"],
                "price": transaction["price"],
                "order_id": transaction["order_id"],
                "commission": transaction["commission"],
                "txn_dollars": -transaction["amount"] * transaction["price"],
            }
        )
    frame = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    frame.index = pd.DatetimeIndex(frame["dt"])
    return frame


def _zipline_transactions() -> pd.Series:
    transactions = _legacy_transactions()
    return pd.Series(
        [[transactions[0], transactions[1]], [transactions[2]]],
        index=pd.DatetimeIndex(
            [
                transactions[0]["dt"].normalize(),
                transactions[2]["dt"].normalize(),
            ]
        ),
        dtype=object,
        name="transactions",
    )


def _assert_canonical(result: pd.DataFrame) -> None:
    assert list(result.columns) == EXPECTED_COLUMNS
    assert isinstance(result.index, pd.DatetimeIndex)
    assert_index_equal(
        result.index,
        pd.DatetimeIndex(result["dt"]),
        check_names=False,
    )
    expected_dollars = -result["amount"] * result["price"]
    np.testing.assert_allclose(result["txn_dollars"], expected_dollars)


def _assert_equivalent(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    _assert_canonical(actual)
    assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_flat_and_canonical_transactions_normalize_to_the_same_lossless_schema() -> None:
    flat = _legacy_transactions()
    canonical = _canonical_transactions()
    flat_before = deepcopy(flat)
    canonical_before = canonical.copy(deep=True)

    from_flat = make_transaction_frame(flat)
    from_canonical = make_transaction_frame(canonical)

    _assert_equivalent(from_flat, canonical)
    _assert_equivalent(from_canonical, canonical)
    assert from_canonical is not canonical
    assert flat == flat_before
    assert_frame_equal(canonical, canonical_before)


def test_zipline_date_to_list_transactions_match_flat_transactions() -> None:
    flat = _legacy_transactions()
    zipline = _zipline_transactions()
    zipline_before = deepcopy(zipline.tolist())

    from_flat = make_transaction_frame(flat)
    from_zipline = make_transaction_frame(zipline)

    _assert_equivalent(from_zipline, from_flat)
    assert zipline.tolist() == zipline_before


def test_plain_mapping_date_to_list_transactions_match_flat_transactions() -> None:
    first, second, third = _legacy_transactions()
    transactions = {
        third["dt"].normalize(): [third],
        first["dt"].normalize(): [first, second],
    }
    before = deepcopy(transactions)

    result = make_transaction_frame(transactions)

    _assert_equivalent(result, make_transaction_frame([first, second, third]))
    assert transactions == before


def test_nested_sid_order_and_commission_fields_are_preserved() -> None:
    result = make_transaction_frame(_legacy_transactions())

    _assert_canonical(result)
    assert result["sid"].tolist() == [101, 202, 101]
    assert result["symbol"].tolist() == ["AAA", "BBB", "AAA"]
    assert result["order_id"].tolist() == ["order-a", "order-b", "order-c"]
    assert result["commission"].tolist() == [0.25, 0.5, 0.1]


def test_symbol_is_derived_from_sid_when_the_optional_input_field_is_absent() -> None:
    canonical = _canonical_transactions().drop(columns=["symbol", "txn_dollars"])

    result = make_transaction_frame(canonical)

    _assert_canonical(result)
    # Pinned pyfolio uses the scalar sid itself as symbol when sid is not the
    # nested Zipline {sid, symbol} representation.
    assert result["symbol"].tolist() == [101, 202, 101]


def test_duplicate_transaction_datetimes_are_retained_in_stable_order() -> None:
    result = make_transaction_frame(_legacy_transactions())

    _assert_canonical(result)
    assert len(result) == 3
    assert result.index.duplicated(keep=False).tolist() == [True, True, False]
    assert result["order_id"].tolist() == ["order-a", "order-b", "order-c"]


@pytest.mark.parametrize("protocol", ["flat", "zipline"])
def test_unsorted_transactions_are_sorted_by_dt_stably(protocol: str) -> None:
    first, second, third = _legacy_transactions()
    if protocol == "flat":
        transactions: Any = [third, first, second]
    else:
        transactions = pd.Series(
            [[third], [first, second]],
            index=pd.DatetimeIndex([third["dt"].normalize(), first["dt"].normalize()]),
            dtype=object,
        )

    result = make_transaction_frame(transactions)

    assert result["order_id"].tolist() == ["order-a", "order-b", "order-c"]
    assert result["dt"].is_monotonic_increasing


def test_canonical_transaction_dollars_are_recomputed_from_amount_and_price() -> None:
    canonical = _canonical_transactions()
    canonical["txn_dollars"] = 999999.0

    result = make_transaction_frame(canonical)

    np.testing.assert_allclose(result["txn_dollars"], -result["amount"] * result["price"])


@pytest.mark.parametrize("field", ["amount", "price"])
def test_non_numeric_transaction_value_is_rejected_as_validation_error(field: str) -> None:
    transactions = _legacy_transactions()
    transactions[0][field] = "not-a-number"

    with pytest.raises(ValidationError, match=field):
        make_transaction_frame(transactions)


@pytest.mark.parametrize(
    "missing",
    REQUIRED_INPUT_COLUMNS,
)
def test_canonical_dataframe_rejects_each_missing_required_field(missing: str) -> None:
    incomplete = _canonical_transactions().drop(columns=[missing])

    with pytest.raises(ValidationError, match=missing):
        make_transaction_frame(incomplete)


@pytest.mark.parametrize("missing", REQUIRED_INPUT_COLUMNS)
def test_legacy_transaction_rejects_missing_required_field(missing: str) -> None:
    transactions = _legacy_transactions()
    del transactions[0][missing]

    with pytest.raises(ValidationError, match=missing):
        make_transaction_frame(transactions)


@pytest.mark.parametrize(
    "empty",
    [
        pytest.param([], id="flat-list"),
        pytest.param(pd.DataFrame(columns=REQUIRED_INPUT_COLUMNS), id="canonical-frame"),
        pytest.param(pd.Series(dtype=object, name="transactions"), id="zipline-series"),
    ],
)
def test_empty_transaction_inputs_return_the_fixed_empty_schema(empty: Any) -> None:
    """The fixed empty schema is an intentional enhanced-normalizer contract.

    Pinned pyfolio 0.9.6 raises while building its empty DataFrame.  Task 5
    deliberately makes all accepted input protocols return the canonical
    schema so downstream code does not need an empty-input special case.
    """

    result = make_transaction_frame(empty)

    assert list(result.columns) == EXPECTED_COLUMNS
    assert result.empty
    assert isinstance(result.index, pd.DatetimeIndex)
