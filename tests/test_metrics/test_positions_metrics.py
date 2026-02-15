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


def test_get_long_short_pos_drops_cash_and_computes_sums():
    idx = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
    positions = pd.DataFrame({"A": [10, -5, 0], "B": [0, -2, 3], "cash": [100, 100, 100]}, index=idx)
    longs, shorts = pm.get_long_short_pos(positions)
    assert list(longs) == [10, 0, 3]
    assert list(shorts) == [0, 7, 0]


def test_compute_style_factor_exposures_aligns_on_index_and_sums():
    idx_p = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
    idx_r = pd.date_range("2024-01-02", periods=2, freq="B", tz="UTC")
    positions = pd.DataFrame({"A": [10, 10, 10], "B": [0, 5, 0]}, index=idx_p)
    risk = pd.DataFrame({"A": [0.1, 0.2], "B": [1.0, 1.0]}, index=idx_r)

    out = pm.compute_style_factor_exposures(positions, risk)
    assert out.index.equals(idx_r)
    assert np.allclose(out.values, [10 * 0.1 + 5 * 1.0, 10 * 0.2 + 0 * 1.0])


def test_compute_sector_exposures_with_and_without_sector_dict():
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    positions = pd.DataFrame({"A": [10, 11], "B": [1, 2], "C": [0, 3]}, index=idx)
    sectors = ["tech", "fin"]
    sector_dict = {"A": "tech", "B": "fin"}

    out1 = pm.compute_sector_exposures(positions, sectors, sector_dict=None)
    assert list(out1.columns) == sectors
    assert np.allclose(out1.values, 0.0)

    out2 = pm.compute_sector_exposures(positions, sectors, sector_dict=sector_dict)
    assert list(out2.columns) == sectors
    assert np.allclose(out2["tech"].values, [10, 11])
    assert np.allclose(out2["fin"].values, [1, 2])


def test_compute_cap_exposures_sums_by_bucket():
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    positions = pd.DataFrame({"A": [10, 11], "B": [1, 2], "C": [0, 3]}, index=idx)
    caps = {"large": ["A"], "small": ["B", "C"]}
    out = pm.compute_cap_exposures(positions, caps)
    assert set(out.columns) == {"large", "small"}
    assert np.allclose(out["large"].values, [10, 11])
    assert np.allclose(out["small"].values, [1, 5])


def test_compute_volume_exposures_counts_days_over_threshold():
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    shares = pd.DataFrame({"A": [100, 10], "B": [0, 50]}, index=idx)
    vols = pd.DataFrame({"A": [10, 10], "B": [100, 10]}, index=idx)
    out = pm.compute_volume_exposures(shares, vols, percentile=5.0)
    assert list(out) == [1, 0]


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
