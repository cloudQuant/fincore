from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import fincore.metrics.perf_attrib as pam


def _make_core_inputs(periods: int = 10, n_tickers: int = 10, n_factors: int = 2):
    dts = pd.date_range("2020-01-01", periods=periods, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    factors = [f"F{i:02d}" for i in range(n_factors)]

    returns = pd.Series(np.linspace(0.001, -0.001, len(dts)), index=dts, name="r")

    positions_df = pd.DataFrame(index=dts, columns=tickers + ["cash"], data=1.0)
    positions_df["cash"] = 0.0

    index = pd.MultiIndex.from_product([dts, tickers], names=["dt", "ticker"])
    factor_returns = pd.DataFrame(index=dts, columns=factors, data=0.001)
    factor_loadings = pd.DataFrame(index=index, columns=factors, data=0.5)
    return returns, positions_df, factor_returns, factor_loadings


def test_perf_attrib_core_requires_inputs():
    returns, positions_df, factor_returns, factor_loadings = _make_core_inputs()
    with pytest.raises(ValueError, match="Either provide positions"):
        pam.perf_attrib_core(returns, None, factor_returns, factor_loadings)
    with pytest.raises(ValueError, match="Either provide factor_returns"):
        pam.perf_attrib_core(returns, positions_df, None, factor_loadings)
    with pytest.raises(ValueError, match="Either provide factor_loadings"):
        pam.perf_attrib_core(returns, positions_df, factor_returns, None)


def test_compute_exposures_internal_requires_inputs():
    _returns, positions_df, _factor_returns, factor_loadings = _make_core_inputs()
    with pytest.raises(ValueError, match="Either provide positions"):
        pam.compute_exposures_internal(None, factor_loadings)
    with pytest.raises(ValueError, match="Either provide factor_loadings"):
        pam.compute_exposures_internal(positions_df.stack(), None)


def test_perf_attrib_requires_positions_factor_returns_factor_loadings():
    returns, positions_df, factor_returns, factor_loadings = _make_core_inputs()
    with pytest.raises(ValueError, match="positions, factor_returns, and factor_loadings are required"):
        pam.perf_attrib(returns, positions=None, factor_returns=factor_returns, factor_loadings=factor_loadings)
    with pytest.raises(ValueError, match="positions, factor_returns, and factor_loadings are required"):
        pam.perf_attrib(returns, positions=positions_df, factor_returns=None, factor_loadings=factor_loadings)
    with pytest.raises(ValueError, match="positions, factor_returns, and factor_loadings are required"):
        pam.perf_attrib(returns, positions=positions_df, factor_returns=factor_returns, factor_loadings=None)


def test_compute_exposures_wrapper_delegates():
    _returns, positions_df, _factor_returns, factor_loadings = _make_core_inputs()
    stacked = pam.normalize_and_stack_positions(positions_df, pos_in_dollars=False)
    out = pam.compute_exposures(stacked, factor_loadings)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == len(positions_df.index)


def test_perf_attrib_core_tilt_returns_non_dataframe_branch_via_custom_dataframe_subclass():
    # Cover the "tilt_returns_raw is not DataFrame" defensive branch without changing runtime behavior.
    returns, positions_df, factor_returns, factor_loadings = _make_core_inputs(periods=6, n_tickers=3, n_factors=2)
    stacked = pam.normalize_and_stack_positions(positions_df, pos_in_dollars=False)

    class WeirdFactorReturns(pd.DataFrame):
        @property
        def _constructor(self):
            return WeirdFactorReturns

        def multiply(self, other, axis="columns", level=None, fill_value=None):  # noqa: D401
            out = super().multiply(other, axis=axis, level=level, fill_value=fill_value)
            return out.sum(axis="columns")

    weird = WeirdFactorReturns(factor_returns)
    exposures, attrib = pam.perf_attrib_core(returns, stacked, weird, factor_loadings)
    assert isinstance(exposures, pd.DataFrame)
    assert isinstance(attrib, pd.DataFrame)
    assert "tilt_returns" in attrib.columns


def test_align_and_warn_positions_series_triggers_missing_assets_gt5_and_missing_dates_gt5(monkeypatch):
    returns, positions_df, factor_returns, factor_loadings = _make_core_inputs(periods=15, n_tickers=12, n_factors=2)

    # Use stacked positions (Series) to cover Series-specific branches.
    stacked = pam.normalize_and_stack_positions(positions_df, pos_in_dollars=False)

    # Make > 5 tickers missing in factor_loadings by dropping loadings for a subset.
    missing_tickers = [f"T{i:02d}" for i in range(6)]
    kept = factor_loadings.index.get_level_values("ticker").isin(missing_tickers)
    factor_loadings_missing_assets = factor_loadings.loc[~kept]

    # Make > 5 dates missing in factor_loadings by keeping only a short date subset.
    keep_dates = set(returns.index[:5])
    keep_idx = factor_loadings_missing_assets.index.get_level_values("dt").isin(list(keep_dates))
    factor_loadings_missing_both = factor_loadings_missing_assets.loc[keep_idx]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out_returns, out_positions, out_factor_returns, _out_factor_loadings = pam.align_and_warn(
            returns=returns,
            positions=stacked,
            factor_returns=factor_returns,
            factor_loadings=factor_loadings_missing_both,
            transactions=None,
            pos_in_dollars=False,
        )

    # Should warn at least for missing assets and missing dates.
    msgs = "\n".join(str(x.message) for x in w)
    assert "assets were missing factor loadings" in msgs
    assert "Truncating date range" in msgs

    # Missing tickers should be removed from positions.
    assert not set(out_positions.index.get_level_values(1)).intersection(set(missing_tickers))

    # Dates should be truncated to what factor_loadings provides.
    assert out_returns.index.min() >= returns.index.min()
    assert len(out_returns) <= len(returns)
    assert out_factor_returns.index.equals(out_returns.index)
