"""Branch-completion tests for the strict Alphalens utils legacy helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.alphalens.utils import (
    _raise_legacy_bin_edge_error,
    _strict_all_nan_factor,
    _strict_all_nan_groupby,
    _strict_empty_factor_projection,
    _strict_prices_for_factor,
    print_table,
    rethrow,
)


def test_raise_legacy_bin_edge_error_non_bin_edges() -> None:
    with pytest.raises(ValueError, match="something else"):
        _raise_legacy_bin_edge_error(ValueError("something else"))


def test_rethrow_with_empty_args() -> None:
    exc = ValueError()
    with pytest.raises(ValueError) as exc_info:
        rethrow(exc, " extra context")
    assert exc_info.value.args == (" extra context",)


def test_strict_all_nan_factor_non_numeric() -> None:
    idx = pd.MultiIndex.from_product([["2024-01-01"], ["A"]], names=("date", "asset"))
    factor = pd.Series(["abc"], index=idx)
    assert _strict_all_nan_factor(factor) is False


def test_strict_all_nan_groupby_rejects_bad_type() -> None:
    idx = pd.MultiIndex.from_product([["2024-01-01"], ["A"]], names=("date", "asset"))
    with pytest.raises(TypeError, match="mapping, Series, or None"):
        _strict_all_nan_groupby(42, None, idx)  # type: ignore[arg-type]


def test_strict_empty_factor_projection_not_all_nan() -> None:
    idx = pd.MultiIndex.from_product([["2024-01-01"], ["A"]], names=("date", "asset"))
    factor = pd.Series([1.0], index=idx)
    forward = pd.DataFrame({"1D": [0.01]}, index=idx)
    assert _strict_empty_factor_projection(factor, forward, groupby=None, groupby_labels=None, max_loss=0.35) is None


def test_strict_empty_factor_projection_non_dataframe_forward() -> None:
    idx = pd.MultiIndex.from_product([["2024-01-01"], ["A"]], names=("date", "asset"))
    factor = pd.Series([np.nan], index=idx)
    assert (
        _strict_empty_factor_projection(
            factor,
            pd.Series([0.01]),
            groupby=None,
            groupby_labels=None,
            max_loss=0.35,  # type: ignore[arg-type]
        )
        is None
    )


def test_strict_empty_factor_projection_bad_max_loss() -> None:
    idx = pd.MultiIndex.from_product([["2024-01-01"], ["A"]], names=("date", "asset"))
    factor = pd.Series([np.nan], index=idx)
    forward = pd.DataFrame({"1D": [np.nan]}, index=idx)
    assert _strict_empty_factor_projection(factor, forward, groupby=None, groupby_labels=None, max_loss=1.5) is None


def test_strict_prices_for_factor_wrong_levels() -> None:
    idx = pd.MultiIndex.from_product([["2024-01-01"], ["A"], ["x"]], names=("date", "asset", "extra"))
    factor = pd.Series([1.0], index=idx)
    prices = pd.DataFrame({"A": [100.0]}, index=pd.DatetimeIndex(["2024-01-01"]))
    assert _strict_prices_for_factor(factor, prices) is prices


def test_print_table_requires_ipython(monkeypatch) -> None:
    import importlib

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "IPython.display":
            raise ModuleNotFoundError("no IPython")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    from fincore.exceptions import DependencyError

    with pytest.raises(DependencyError, match="IPython"):
        print_table(pd.Series([1.0, 2.0]), name="x")
