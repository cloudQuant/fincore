from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.metrics import positions as pm


def test_get_percent_alloc_replaces_inf_and_neg_inf_with_nan():
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    values = pd.DataFrame({"A": [1.0, 1.0], "B": [-1.0, 0.0], "cash": [0.0, 0.0]}, index=idx)
    alloc = pm.get_percent_alloc(values)
    assert np.isnan(alloc.loc[idx[0], "A"])
    assert np.isnan(alloc.loc[idx[0], "B"])


def test_get_top_long_short_abs_smoke():
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    positions = pd.DataFrame({"A": [10, 11], "B": [-5, -4], "C": [2, 1], "cash": [100, 100]}, index=idx)
    top_long, top_short, top_abs = pm.get_top_long_short_abs(positions, top=1)
    assert list(top_long.index) == ["A"]
    assert list(top_short.index) == ["B"]
    assert list(top_abs.index) == ["A"]


def test_get_max_median_position_concentration_smoke():
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    positions = pd.DataFrame({"A": [10, 0], "B": [-5, -5], "cash": [95, 105]}, index=idx)
    out = pm.get_max_median_position_concentration(positions)
    assert set(out.columns) == {"max_long", "median_long", "median_short", "max_short"}


def test_extract_pos_pivots_values_and_joins_cash():
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    positions = pd.DataFrame(
        {
            "sid": ["A", "B", "A"],
            "amount": [1.0, 2.0, 1.0],
            "last_sale_price": [10.0, 20.0, 11.0],
        },
        index=[idx[0], idx[0], idx[1]],
    )
    cash = pd.Series([100.0, 101.0], index=idx)
    out = pm.extract_pos(positions, cash)
    assert out.columns.name == "sid"
    assert "cash" in out.columns
    assert out.loc[idx[0], "A"] == 10.0
    assert out.loc[idx[0], "B"] == 40.0
    assert out.loc[idx[1], "A"] == 11.0


def test_get_long_short_pos_returns_normalized_long_short_and_net_exposure():
    idx = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
    positions = pd.DataFrame(
        {
            "A": [60.0, -30.0, 0.0],
            "B": [-20.0, 10.0, 0.0],
            "cash": [60.0, 120.0, 100.0],
        },
        index=idx,
    )
    result = pm.get_long_short_pos(positions)
    expected = pd.DataFrame(
        {
            "long": [0.6, 0.1, 0.0],
            "short": [-0.2, -0.3, 0.0],
            "net exposure": [0.4, -0.2, 0.0],
        },
        index=idx,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_get_long_short_notional_keeps_the_previous_amount_summary():
    idx = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
    positions = pd.DataFrame({"A": [10, -5, 0], "B": [0, -2, 3], "cash": [100, 100, 100]}, index=idx)

    longs, shorts = pm.get_long_short_notional(positions)

    pd.testing.assert_series_equal(longs, pd.Series([10, 0, 3], index=idx))
    pd.testing.assert_series_equal(shorts, pd.Series([0, 7, 0], index=idx))


def test_compute_style_factor_exposures_aligns_and_normalizes_by_gross():
    idx_p = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
    idx_r = pd.date_range("2024-01-02", periods=2, freq="B", tz="UTC")
    positions = pd.DataFrame({"A": [10, 10, 10], "B": [0, 5, 0]}, index=idx_p)
    risk = pd.DataFrame({"A": [0.1, 0.2], "B": [1.0, 1.0]}, index=idx_r)

    out = pm.compute_style_factor_exposures(positions, risk)
    assert out.index.equals(idx_r)
    assert np.allclose(out.values, [(10 * 0.1 + 5 * 1.0) / 15, (10 * 0.2 + 0 * 1.0) / 10])


def test_compute_sector_exposures_returns_named_normalized_bundle():
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    positions = pd.DataFrame({"A": [10, 11], "B": [-1, -2], "C": [0, 3], "cash": [5, 5]}, index=idx)
    sectors = pd.DataFrame({"A": [1, 1], "B": [2, 2], "C": [1, 1]}, index=idx)
    out = pm.compute_sector_exposures(positions, sectors, sector_dict={1: "tech", 2: "fin"})

    assert list(out.long.columns) == ["tech", "fin"]
    assert np.allclose(out.long["tech"].values, 1.0)
    assert np.allclose(out.short["fin"].values, -1.0)
    assert np.allclose(out.gross["tech"].values, [10 / 11, 14 / 16])
    assert np.allclose(out.net["fin"].values, 1.0)


def test_compute_cap_exposures_returns_frozen_bucket_order():
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    positions = pd.DataFrame({"A": [10, 11], "B": [-1, -2], "C": [0, 3], "cash": [5, 5]}, index=idx)
    caps = pd.DataFrame({"A": [1e11, 1e11], "B": [1e9, 1e9], "C": [5e9, 5e9]}, index=idx)
    out = pm.compute_cap_exposures(positions, caps)
    assert list(out.long.columns) == list(pm.CAP_BUCKETS)
    assert np.allclose(out.long["Large"].values, [1.0, 11 / 14])
    assert np.allclose(out.short["Small"].values, -1.0)
    assert np.allclose(out.gross["Mid"].values, [0.0, 3 / 16])


def test_compute_volume_exposures_returns_named_percentile_series():
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    shares = pd.DataFrame({"A": [100, 10], "B": [0, 50]}, index=idx)
    vols = pd.DataFrame({"A": [10, 10], "B": [100, 10]}, index=idx)
    out = pm.compute_volume_exposures(shares, vols, percentile=0.5)
    assert np.allclose(out.long.values, [1000.0, 300.0])
    assert out.short.isna().all()
    assert np.allclose(out.gross.values, [1000.0, 300.0])


def test_get_sector_exposures_warns_on_unmapped_symbols_and_keeps_cash():
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    positions = pd.DataFrame({"A": [10, 10], "B": [1, 1], "cash": [100, 100]}, index=idx)
    symbol_sector_map = {"A": "tech"}

    with pytest.warns(UserWarning, match="no sector mapping"):
        out = pm.get_sector_exposures(positions, symbol_sector_map)
    assert "cash" in out.columns
    assert "tech" in out.columns


def test_gross_lev_replaces_inf_with_nan():
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    positions = pd.DataFrame({"A": [10, 10], "cash": [-10, 0]}, index=idx)
    out = pm.gross_lev(positions)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 1.0


def test_stack_positions_drops_cash_and_sets_index_names():
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    positions = pd.DataFrame({"A": [10, 11], "cash": [100, 100]}, index=idx)
    stacked = pm.stack_positions(positions)
    assert stacked.index.names == ["dt", "ticker"]
    assert "cash" not in stacked.index.get_level_values("ticker")
