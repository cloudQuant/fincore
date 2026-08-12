from __future__ import annotations

import importlib
from decimal import Decimal
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas.testing import assert_series_equal

matplotlib.use("Agg", force=True)

from fincore.constants import CAP_BUCKETS, SECTORS
from fincore.contracts.portfolio import ExposureBundle
from fincore.empyrical import Empyrical
from fincore.exceptions import DataAlignmentError, ValidationError
from fincore.metrics.positions import (
    compute_cap_exposures,
    compute_sector_exposures,
    compute_style_factor_exposures,
    compute_volume_exposures,
    get_long_short_notional,
)
from fincore.metrics.transactions import make_transaction_frame
from fincore.pyfolio import Pyfolio


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    yield
    plt.close("all")


def _partial_metadata_inputs() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    index = pd.date_range("2024-08-01", periods=1, tz="UTC")
    positions = pd.DataFrame(
        {"AAA": [50.0], "BBB": [50.0], "cash": [0.0]},
        index=index,
    )
    return positions, index


def test_style_partial_metadata_keeps_all_position_assets_in_denominator() -> None:
    positions, index = _partial_metadata_inputs()
    factors = pd.DataFrame({"AAA": [1.0], "EXTRA": [99.0]}, index=index)

    result = compute_style_factor_exposures(positions, factors)

    assert_series_equal(result, pd.Series([0.5], index=index))


def test_sector_partial_metadata_keeps_all_position_assets_in_denominator() -> None:
    positions, index = _partial_metadata_inputs()
    sectors = pd.DataFrame({"AAA": [311], "EXTRA": [309]}, index=index)

    result = compute_sector_exposures(positions, sectors)

    assert result.long.loc[index[0], "Technology"] == pytest.approx(0.5)
    assert result.gross.loc[index[0], "Technology"] == pytest.approx(0.5)
    assert result.gross.loc[index[0]].sum() == pytest.approx(0.5)


def test_cap_partial_metadata_keeps_all_position_assets_in_denominator() -> None:
    positions, index = _partial_metadata_inputs()
    caps = pd.DataFrame({"AAA": [1.0e9], "EXTRA": [5.0e9]}, index=index)

    result = compute_cap_exposures(positions, caps)

    assert result.long.loc[index[0], "Small"] == pytest.approx(0.5)
    assert result.gross.loc[index[0], "Small"] == pytest.approx(0.5)
    assert result.gross.loc[index[0]].sum() == pytest.approx(0.5)


def _transaction(**overrides: Any) -> dict[str, Any]:
    transaction = {
        "dt": pd.Timestamp("2024-08-01 14:00", tz="UTC"),
        "sid": {"sid": 1, "symbol": "AAA"},
        "amount": 2.0,
        "price": 5.0,
        "order_id": "order-1",
        "commission": 0.25,
    }
    transaction.update(overrides)
    return transaction


def test_canonical_transaction_frame_rejects_duplicate_columns() -> None:
    frame = pd.DataFrame([_transaction()])
    frame = pd.concat([frame, frame[["price"]]], axis="columns", sort=False)
    assert not frame.columns.is_unique

    with pytest.raises(ValidationError, match="duplicate.*column"):
        make_transaction_frame(frame)


@pytest.mark.parametrize("field", ["amount", "price"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_transaction_amount_and_price_must_be_finite(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match=field):
        make_transaction_frame([_transaction(**{field: value})])


@pytest.mark.parametrize("field", ["amount", "price"])
def test_transaction_decimal_policy_is_explicitly_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        make_transaction_frame([_transaction(**{field: Decimal("1.25")})])


def test_transaction_commission_is_preserved_without_numeric_validation() -> None:
    result = make_transaction_frame([_transaction(commission=None)])

    assert result.loc[result.index[0], "commission"] is None


def test_long_short_notional_is_on_class_surfaces_but_not_strict_module_api() -> None:
    positions, _ = _partial_metadata_inputs()
    expected = get_long_short_notional(positions)
    strict_module = importlib.import_module("fincore.empyrical")

    assert not hasattr(strict_module, "get_long_short_notional")
    for surface in (Empyrical, Pyfolio):
        actual = surface.get_long_short_notional(positions)
        assert_series_equal(actual[0], expected[0])
        assert_series_equal(actual[1], expected[1])


@pytest.mark.parametrize("surface", ["typed", "facade"])
def test_duplicate_custom_sector_display_names_are_rejected(surface: str) -> None:
    index = pd.date_range("2024-08-01", periods=1, tz="UTC")
    positions = pd.DataFrame({"AAA": [50.0], "BBB": [50.0], "cash": [0.0]}, index=index)
    sectors = pd.DataFrame({"AAA": [1], "BBB": [2]}, index=index)
    sector_dict = {1: "Duplicate", 2: "Duplicate"}

    with pytest.raises(ValidationError, match="duplicate.*sector.*name"):
        if surface == "typed":
            compute_sector_exposures(positions, sectors, sector_dict=sector_dict)
        else:
            Pyfolio().compute_sector_exposures(positions, sectors, sector_dict=sector_dict)


@pytest.mark.parametrize("case", ["zero_shares", "no_assets"])
def test_volume_zero_share_and_no_asset_rows_are_finite_zero(case: str) -> None:
    index = pd.date_range("2024-08-01", periods=2, tz="UTC")
    if case == "zero_shares":
        shares = pd.DataFrame({"AAA": [0.0, 0.0]}, index=index)
        volumes = pd.DataFrame({"AAA": [100.0, 200.0]}, index=index)
    else:
        shares = pd.DataFrame(index=index)
        volumes = pd.DataFrame(index=index)

    result = compute_volume_exposures(shares, volumes, 0.5)

    for component in (result.long, result.short, result.gross):
        assert np.isfinite(component.to_numpy(dtype=float)).all()
        assert_series_equal(component, pd.Series([0.0, 0.0], index=index), check_names=False)


def test_risk_sheet_ignores_unused_disjoint_shares_panel(pyfolio_risk_inputs: Any) -> None:
    unused_shares = pyfolio_risk_inputs.shares_held.copy()
    unused_shares.index = unused_shares.index + pd.DateOffset(years=10)

    fig = Pyfolio().create_risk_tear_sheet(
        positions=pyfolio_risk_inputs.positions,
        sectors=pyfolio_risk_inputs.sectors,
        shares_held=unused_shares,
        volumes=None,
        estimate_intraday=False,
        run_flask_app=True,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3


def test_risk_sheet_active_mixed_timezones_raise_alignment_error(pyfolio_risk_inputs: Any) -> None:
    naive_sectors = pyfolio_risk_inputs.sectors.copy()
    naive_sectors.index = naive_sectors.index.tz_localize(None)

    with pytest.raises(DataAlignmentError, match="timezone mismatch"):
        Pyfolio().create_risk_tear_sheet(
            positions=pyfolio_risk_inputs.positions,
            sectors=naive_sectors,
            estimate_intraday=False,
            run_flask_app=True,
        )


def test_risk_sheet_active_panel_no_overlap_warns_and_returns_none(pyfolio_risk_inputs: Any) -> None:
    disjoint_sectors = pyfolio_risk_inputs.sectors.copy()
    disjoint_sectors.index = disjoint_sectors.index + pd.DateOffset(years=10)

    with pytest.warns(UserWarning, match="No overlapping index"):
        result = Pyfolio().create_risk_tear_sheet(
            positions=pyfolio_risk_inputs.positions,
            sectors=disjoint_sectors,
            estimate_intraday=False,
            run_flask_app=True,
        )

    assert result is None


def test_cap_bucket_endpoints_are_inclusive_and_double_count_like_pinned() -> None:
    index = pd.date_range("2024-08-01", periods=1, tz="UTC")
    positions = pd.DataFrame({"AAA": [100.0], "cash": [0.0]}, index=index)
    caps = pd.DataFrame({"AAA": [300_000_000.0]}, index=index)

    result = compute_cap_exposures(positions, caps)

    assert result.gross.loc[index[0], "Micro"] == pytest.approx(1.0)
    assert result.gross.loc[index[0], "Small"] == pytest.approx(1.0)


def test_zipline_mapping_key_is_ignored_in_favor_of_embedded_transaction_dt() -> None:
    transaction = _transaction()
    misleading_key = pd.Timestamp("1999-01-01", tz="UTC")

    result = make_transaction_frame({misleading_key: [transaction]})

    assert result.index[0] == transaction["dt"]
    assert result.loc[result.index[0], "dt"] == transaction["dt"]


@pytest.mark.parametrize("branch", ["missing", "unexpected", "duplicate"])
def test_exposure_bundle_projection_rejects_invalid_category_order(branch: str) -> None:
    index = pd.date_range("2024-08-01", periods=1, tz="UTC")
    columns = ["One", "Two"]
    frame = pd.DataFrame([[0.5, 0.5]], index=index, columns=columns)
    bundle = ExposureBundle(long=frame, short=-frame, gross=frame, net=frame)
    if branch == "missing":
        order = [*columns, "Missing"]
    elif branch == "unexpected":
        order = columns[:-1]
    else:
        order = [columns[0], columns[0], columns[1]]

    with pytest.raises(ValidationError, match=branch):
        bundle.as_legacy_tuple(order)


def test_contract_constants_keep_pinned_cap_and_sector_order() -> None:
    assert list(SECTORS.values())[-1] == "Technology"
    assert list(CAP_BUCKETS) == ["Micro", "Small", "Mid", "Large", "Mega"]
