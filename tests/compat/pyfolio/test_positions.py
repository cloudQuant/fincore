from __future__ import annotations

import importlib
import json
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from fincore.constants import CAP_BUCKETS, SECTORS
from fincore.exceptions import DataAlignmentError, ValidationError
from fincore.metrics import positions as positions_metrics
from fincore.pyfolio import Pyfolio

PORTFOLIO_CONTRACT_FIXTURE = Path(__file__).parents[1] / "fixtures" / "pyfolio-0.9.6-portfolio-contracts.json"


def _assert_exposure_bundle(
    result: Any,
    *,
    index: pd.DatetimeIndex,
    columns: list[str],
) -> None:
    # Check the representation before importing the not-yet-implemented
    # contract module.  On the RED implementation this reports the actual
    # DataFrame-vs-bundle mismatch instead of becoming a collection error.
    assert type(result).__name__ == "ExposureBundle"
    contract_module = importlib.import_module("fincore.contracts.portfolio")
    assert isinstance(result, contract_module.ExposureBundle)
    assert is_dataclass(result)
    assert result.__dataclass_params__.frozen is True

    for field in ("long", "short", "gross", "net"):
        frame = getattr(result, field)
        assert isinstance(frame, pd.DataFrame)
        assert_index_equal(frame.index, index)
        assert list(frame.columns) == columns
        assert frame.columns.is_unique
        assert not np.isinf(frame.to_numpy(dtype=float)).any()

    with pytest.raises(FrozenInstanceError):
        result.long = result.long.copy()


def _assert_volume_bundle(result: Any, *, index: pd.DatetimeIndex) -> None:
    assert type(result).__name__ == "VolumeExposureBundle"
    contract_module = importlib.import_module("fincore.contracts.portfolio")
    assert isinstance(result, contract_module.VolumeExposureBundle)
    assert is_dataclass(result)
    assert result.__dataclass_params__.frozen is True

    for field in ("long", "short", "gross"):
        series = getattr(result, field)
        assert isinstance(series, pd.Series)
        assert_index_equal(series.index, index)
        assert not np.isinf(series.to_numpy(dtype=float)).any()

    with pytest.raises(FrozenInstanceError):
        result.long = result.long.copy()


def _assert_legacy_projection(
    legacy: Any,
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    category_order: list[str],
) -> None:
    assert isinstance(legacy, tuple)
    assert len(legacy) == 4
    for component, frame in zip(legacy, frames, strict=True):
        assert len(component) == len(category_order)
        for actual, category in zip(component, category_order, strict=True):
            assert_series_equal(actual, frame[category], check_names=False)


PANEL_COMPUTATIONS = [
    pytest.param(positions_metrics.compute_style_factor_exposures, "positions", "sectors", id="style"),
    pytest.param(positions_metrics.compute_sector_exposures, "positions", "sectors", id="sector"),
    pytest.param(positions_metrics.compute_cap_exposures, "positions", "caps", id="cap"),
    pytest.param(positions_metrics.compute_volume_exposures, "shares_held", "volumes", id="volume"),
]


def _call_panel_computation(compute: Any, left: pd.DataFrame, right: pd.DataFrame) -> Any:
    if compute is positions_metrics.compute_volume_exposures:
        return compute(left, right, 0.5)
    return compute(left, right)


@pytest.mark.parametrize(("compute", "left_name", "right_name"), PANEL_COMPUTATIONS)
@pytest.mark.parametrize("side", ["left", "right"])
def test_portfolio_computations_reject_duplicate_asset_columns(
    pyfolio_risk_inputs: Any,
    compute: Any,
    left_name: str,
    right_name: str,
    side: str,
) -> None:
    left = getattr(pyfolio_risk_inputs, left_name)
    right = getattr(pyfolio_risk_inputs, right_name)
    if side == "left":
        left = pd.concat([left, left.iloc[:, :1]], axis="columns", sort=False)
    else:
        right = pd.concat([right, right.iloc[:, :1]], axis="columns", sort=False)

    with pytest.raises(ValidationError, match="duplicate"):
        _call_panel_computation(compute, left, right)


@pytest.mark.parametrize(("compute", "left_name", "right_name"), PANEL_COMPUTATIONS)
@pytest.mark.parametrize("side", ["left", "right"])
def test_portfolio_computations_reject_duplicate_dates(
    pyfolio_risk_inputs: Any,
    compute: Any,
    left_name: str,
    right_name: str,
    side: str,
) -> None:
    left = getattr(pyfolio_risk_inputs, left_name)
    right = getattr(pyfolio_risk_inputs, right_name)
    if side == "left":
        left = pd.concat([left, left.iloc[:1]], axis="index", sort=False)
    else:
        right = pd.concat([right, right.iloc[:1]], axis="index", sort=False)

    with pytest.raises(DataAlignmentError, match="duplicate"):
        _call_panel_computation(compute, left, right)


def test_sector_compute_returns_named_bundle_despite_exactly_four_asset_columns(
    pyfolio_risk_inputs: Any,
) -> None:
    assert pyfolio_risk_inputs.sectors.shape[1] == 4

    result = positions_metrics.compute_sector_exposures(
        pyfolio_risk_inputs.positions,
        pyfolio_risk_inputs.sectors,
    )

    _assert_exposure_bundle(
        result,
        index=pyfolio_risk_inputs.positions.index,
        columns=list(SECTORS.values()),
    )


def test_sector_pyfolio_compatibility_projection_uses_frozen_category_order(
    pyfolio_risk_inputs: Any,
) -> None:
    bundle = positions_metrics.compute_sector_exposures(
        pyfolio_risk_inputs.positions,
        pyfolio_risk_inputs.sectors,
    )
    legacy = Pyfolio().compute_sector_exposures(
        pyfolio_risk_inputs.positions,
        pyfolio_risk_inputs.sectors,
    )

    _assert_legacy_projection(
        legacy,
        (bundle.long, bundle.short, bundle.gross, bundle.net),
        list(SECTORS.values()),
    )


def test_cap_compute_returns_named_bundle_despite_exactly_four_asset_columns(
    pyfolio_risk_inputs: Any,
) -> None:
    assert pyfolio_risk_inputs.caps.shape[1] == 4

    result = positions_metrics.compute_cap_exposures(
        pyfolio_risk_inputs.positions,
        pyfolio_risk_inputs.caps,
    )

    _assert_exposure_bundle(
        result,
        index=pyfolio_risk_inputs.positions.index,
        columns=list(CAP_BUCKETS),
    )


def test_cap_pyfolio_compatibility_projection_uses_frozen_category_order(
    pyfolio_risk_inputs: Any,
) -> None:
    bundle = positions_metrics.compute_cap_exposures(
        pyfolio_risk_inputs.positions,
        pyfolio_risk_inputs.caps,
    )
    legacy = Pyfolio().compute_cap_exposures(
        pyfolio_risk_inputs.positions,
        pyfolio_risk_inputs.caps,
    )

    _assert_legacy_projection(
        legacy,
        (bundle.long, bundle.short, bundle.gross, bundle.net),
        list(CAP_BUCKETS),
    )


def test_volume_compute_returns_named_bundle_despite_exactly_three_dates(
    pyfolio_risk_inputs: Any,
) -> None:
    assert len(pyfolio_risk_inputs.shares_held.index) == 3

    result = positions_metrics.compute_volume_exposures(
        pyfolio_risk_inputs.shares_held,
        pyfolio_risk_inputs.volumes,
        pyfolio_risk_inputs.percentile,
    )

    _assert_volume_bundle(result, index=pyfolio_risk_inputs.positions.index)


def test_volume_pyfolio_compatibility_projection_matches_named_bundle(
    pyfolio_risk_inputs: Any,
) -> None:
    bundle = positions_metrics.compute_volume_exposures(
        pyfolio_risk_inputs.shares_held,
        pyfolio_risk_inputs.volumes,
        pyfolio_risk_inputs.percentile,
    )
    legacy = Pyfolio().compute_volume_exposures(
        pyfolio_risk_inputs.shares_held,
        pyfolio_risk_inputs.volumes,
        pyfolio_risk_inputs.percentile,
    )

    assert isinstance(legacy, tuple)
    assert len(legacy) == 3
    for actual, expected in zip(
        legacy,
        (bundle.long, bundle.short, bundle.gross),
        strict=True,
    ):
        assert_series_equal(actual, expected, check_names=False)


def test_style_factor_exposure_excludes_cash_aligns_labels_and_normalizes_by_gross() -> None:
    positions_index = pd.date_range("2024-02-01", periods=3, freq="B", tz="UTC")
    factor_index = pd.DatetimeIndex(
        [positions_index[2], positions_index[1], positions_index[2] + pd.offsets.BDay()],
    )
    positions = pd.DataFrame(
        {
            "AAA": [10.0, 30.0, -20.0],
            "BBB": [-10.0, -10.0, 20.0],
            "cash": [100.0, 80.0, 1000.0],
        },
        index=positions_index,
    )
    factors = pd.DataFrame(
        {
            "cash": [999.0, 999.0, 999.0],
            "BBB": [-1.0, 2.0, 7.0],
            "AAA": [1.0, 0.5, 8.0],
        },
        index=factor_index,
    )
    positions_before = positions.copy(deep=True)
    factors_before = factors.copy(deep=True)

    result = positions_metrics.compute_style_factor_exposures(positions, factors)

    expected = pd.Series(
        [-0.125, -1.0],
        index=pd.DatetimeIndex([positions_index[1], positions_index[2]]),
    )
    assert_series_equal(result, expected)
    assert_frame_equal(positions, positions_before)
    assert_frame_equal(factors, factors_before)


@pytest.mark.parametrize("case", ["empty", "zero", "all_cash"])
def test_style_factor_boundaries_are_finite_and_preserve_index(case: str) -> None:
    if case == "empty":
        index = pd.DatetimeIndex([], tz="UTC")
        positions = pd.DataFrame(columns=["AAA", "cash"], index=index, dtype=float)
        factors = pd.DataFrame(columns=["AAA"], index=index, dtype=float)
    elif case == "zero":
        index = pd.date_range("2024-03-01", periods=1, tz="UTC")
        positions = pd.DataFrame({"AAA": [0.0], "cash": [100.0]}, index=index)
        factors = pd.DataFrame({"AAA": [2.0]}, index=index)
    else:
        index = pd.date_range("2024-03-01", periods=1, tz="UTC")
        positions = pd.DataFrame({"cash": [100.0]}, index=index)
        factors = pd.DataFrame(index=index)

    result = positions_metrics.compute_style_factor_exposures(positions, factors)

    assert_index_equal(result.index, index)
    assert not np.isinf(result.to_numpy(dtype=float)).any()
    if len(index):
        assert result.iloc[0] == 0.0


def test_volume_compute_aligns_dates_and_columns_without_mutating_inputs() -> None:
    dates = pd.date_range("2024-04-01", periods=3, freq="B", tz="UTC")
    shares = pd.DataFrame(
        {"AAA": [100.0, 20.0, 40.0], "BBB": [-10.0, -50.0, -20.0]},
        index=dates,
    )
    volumes = pd.DataFrame(
        {"BBB": [100.0, 200.0], "AAA": [1000.0, 400.0]},
        index=pd.DatetimeIndex([dates[2], dates[1]]),
    )
    shares_before = shares.copy(deep=True)
    volumes_before = volumes.copy(deep=True)

    result = positions_metrics.compute_volume_exposures(shares, volumes, 0.5)

    _assert_volume_bundle(result, index=pd.DatetimeIndex([dates[1], dates[2]]))
    assert_frame_equal(shares, shares_before)
    assert_frame_equal(volumes, volumes_before)


@pytest.mark.parametrize("kind", ["sector", "cap", "volume"])
def test_empty_exposure_inputs_return_typed_empty_bundles(kind: str) -> None:
    index = pd.DatetimeIndex([], tz="UTC")
    positions = pd.DataFrame(columns=["AAA", "cash"], index=index, dtype=float)
    panel = pd.DataFrame(columns=["AAA"], index=index, dtype=float)

    if kind == "sector":
        result = positions_metrics.compute_sector_exposures(positions, panel)
        _assert_exposure_bundle(result, index=index, columns=list(SECTORS.values()))
    elif kind == "cap":
        result = positions_metrics.compute_cap_exposures(positions, panel)
        _assert_exposure_bundle(result, index=index, columns=list(CAP_BUCKETS))
    else:
        result = positions_metrics.compute_volume_exposures(panel, panel, 0.5)
        _assert_volume_bundle(result, index=index)


def test_get_long_short_pos_returns_normalized_compatibility_frame() -> None:
    index = pd.date_range("2024-05-01", periods=3, freq="B", tz="UTC")
    positions = pd.DataFrame(
        {
            "AAA": [60.0, -30.0, 0.0],
            "BBB": [-20.0, 10.0, 0.0],
            "cash": [60.0, 120.0, 100.0],
        },
        index=index,
    )
    before = positions.copy(deep=True)

    result = positions_metrics.get_long_short_pos(positions)

    expected = pd.DataFrame(
        {
            "long": [0.6, 0.1, 0.0],
            "short": [-0.2, -0.3, 0.0],
            "net exposure": [0.4, -0.2, 0.0],
        },
        index=index,
    )
    assert_frame_equal(result, expected)
    assert_frame_equal(positions, before)


def test_get_long_short_notional_preserves_the_old_amount_summary() -> None:
    index = pd.date_range("2024-05-01", periods=2, freq="B", tz="UTC")
    positions = pd.DataFrame(
        {"AAA": [60.0, -30.0], "BBB": [-20.0, 10.0], "cash": [60.0, 120.0]},
        index=index,
    )

    longs, shorts = positions_metrics.get_long_short_notional(positions)

    assert_series_equal(longs, pd.Series([60.0, 10.0], index=index))
    assert_series_equal(shorts, pd.Series([20.0, 30.0], index=index))


def test_portfolio_contract_fixture_is_generator_backed_and_pins_source_goldens() -> None:
    assert PORTFOLIO_CONTRACT_FIXTURE.is_file(), (
        "Generate the pinned portfolio contract fixture with "
        "scripts/generate_compat_manifest.py; do not hand-author source hashes or goldens."
    )
    data = json.loads(PORTFOLIO_CONTRACT_FIXTURE.read_text(encoding="utf-8"))

    assert data["schema_version"] == 1
    assert data["project"] == "pyfolio"
    assert data["version"] == "0.9.6"
    assert data["commit"] == "724bbd7dbed9a88bb47e1057f2ca29b3409d8e7a"
    assert data["fixture_source"]["generator"] == "scripts/generate_compat_manifest.py"
    assert data["fixture_source"]["mode"]
    assert data["oracle_verification"] == {"status": "not_run", "reviewed": False}
    sources = {entry["path"]: entry["sha256"] for entry in data["source_files"]}
    assert set(sources) == {"pos.py", "risk.py", "txn.py"}
    assert all(len(digest) == 64 for digest in sources.values())
    assert data["category_order"]["sectors"] == list(SECTORS.values())
    assert data["category_order"]["cap_buckets"] == list(CAP_BUCKETS)
    assert data["constants"]["SECTORS"] == [[key, value] for key, value in SECTORS.items()]
    assert data["constants"]["CAP_BUCKETS"][:-1] == [
        [key, list(value)] for key, value in list(CAP_BUCKETS.items())[:-1]
    ]
    assert data["constants"]["CAP_BUCKETS"][-1] == ["Mega", [200000000000, "Infinity"]]
    assert {
        "style_factor",
        "sector_exposures",
        "cap_exposures",
        "volume_exposures",
        "long_short_positions",
        "transactions",
    } <= set(data["golden_cases"])
    assert all(case["reviewed"] is False for case in data["golden_cases"].values())
