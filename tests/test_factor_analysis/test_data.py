"""Focused validation and edge-case coverage for the enhanced factor-data kernel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _small_inputs() -> tuple[pd.Series, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=5, name="date")
    assets = ("A", "B", "C", "D")
    index = pd.MultiIndex.from_product((dates[:3], assets), names=("date", "asset"))
    factor = pd.Series(np.arange(1, len(index) + 1, dtype=float), index=index, name="factor")
    prices = pd.DataFrame(
        np.arange(1, len(dates) * len(assets) + 1, dtype=float).reshape(len(dates), len(assets)),
        index=dates,
        columns=assets,
    )
    return factor, prices


def test_quantize_factor_handles_edges_groups_and_zero_aware() -> None:
    """Direct enhanced quantization supports each documented binning mode."""

    from fincore.factor_analysis.data import quantize_factor

    factor, _ = _small_inputs()
    factor_data = pd.DataFrame({"factor": factor, "group": ["x", "x", "y", "y"] * 3})
    quantized = quantize_factor(factor_data, quantiles=2, by_group=True)
    assert quantized.name == "factor_quantile"
    assert quantized.between(1, 2).all()

    zero_data = factor_data.copy()
    zero_data["factor"] = [-2, -1, 0, 1] * 3
    zero_quantized = quantize_factor(zero_data, quantiles=None, bins=2, zero_aware=True)
    assert zero_quantized.between(1, 2).all()

    edge_quantized = quantize_factor(factor_data, quantiles=None, bins=(0, 4, 20))
    assert edge_quantized.between(1, 2).all()


def test_data_validation_rejects_duplicate_index_missing_assets_and_ambiguous_bins() -> None:
    """Unsafe alignment and unstable discretization errors are explicit."""

    from fincore.factor_analysis.data import prepare_factor_data, quantize_factor

    factor, prices = _small_inputs()
    duplicate = pd.concat((factor, factor.iloc[:1]))
    with pytest.raises(ValueError, match="unique"):
        prepare_factor_data(duplicate, prices, periods=(1,))

    missing_asset = factor.rename(index={"A": "MISSING"}, level="asset")
    with pytest.raises(ValueError, match="assets"):
        prepare_factor_data(missing_asset, prices, periods=(1,))

    data = pd.DataFrame({"factor": [1.0, 1.0]}, index=factor.index[:2])
    with pytest.raises(ValueError, match="Bin edges must be unique"):
        quantize_factor(data, quantiles=2)


def test_prepare_factor_data_handles_group_mapping_labels_and_all_nan_factor() -> None:
    """Group joins preserve categorical labels and all-NaN input fails clearly."""

    from fincore.factor_analysis.data import prepare_factor_data

    factor, prices = _small_inputs()
    groups = pd.Series({"A": "energy", "B": "banks", "C": "energy", "D": "banks"}, name="group")
    result = prepare_factor_data(
        factor,
        prices,
        groupby=groups,
        groupby_labels={"energy": "Energy", "banks": "Banks"},
        periods=(1,),
        max_loss=1,
    )
    assert result.data["group"].dtype.name == "category"
    assert set(result.data["group"].cat.categories) == {"Banks", "Energy"}

    with pytest.raises(ValueError, match="finite"):
        prepare_factor_data(factor * np.nan, prices, periods=(1,))
