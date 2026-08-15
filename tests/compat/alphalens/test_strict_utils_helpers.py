"""Strict helper behavior that must not fall back to C0/C1-only stubs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fincore.alphalens import utils


def test_demean_forward_returns_copies_input_and_demeans_each_date() -> None:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["a", "b"]], names=["date", "asset"]
    )
    source = pd.DataFrame(
        {"1D": [0.01, 0.03, 0.02, 0.06], "5D": [0.10, 0.20, 0.30, 0.50], "factor": [1, 2, 3, 4]},
        index=index,
    )
    before = source.copy(deep=True)

    actual = utils.demean_forward_returns(source)

    assert actual is not source
    pd.testing.assert_frame_equal(
        actual[["1D", "5D"]],
        pd.DataFrame({"1D": [-0.01, 0.01, -0.02, 0.02], "5D": [-0.05, 0.05, -0.10, 0.10]}, index=index),
    )
    pd.testing.assert_series_equal(actual["factor"], source["factor"])
    pd.testing.assert_frame_equal(source, before)


def test_demean_forward_returns_honors_explicit_grouping() -> None:
    index = pd.MultiIndex.from_product([pd.to_datetime(["2024-01-02"]), ["a", "b", "c", "d"]], names=["date", "asset"])
    source = pd.DataFrame({"1D": [0.01, 0.03, 0.10, 0.16]}, index=index)
    groups = ["left", "left", "right", "right"]

    actual = utils.demean_forward_returns(source, grouper=groups)

    np.testing.assert_allclose(actual["1D"].to_numpy(), [-0.01, 0.01, -0.03, 0.03])


def test_rate_and_standard_deviation_conversion_follow_named_period() -> None:
    returns = pd.Series([0.10, -0.10], name="5D")
    standard_error = pd.Series([0.20, 0.40], name="5D")

    converted_returns = utils.rate_of_return(returns, "1D")
    converted_standard_error = utils.std_conversion(standard_error, "1D")

    np.testing.assert_allclose(converted_returns.to_numpy(), np.array([1.10, 0.90]) ** (1 / 5) - 1)
    np.testing.assert_allclose(converted_standard_error.to_numpy(), standard_error.to_numpy() / math.sqrt(5))
    assert converted_returns.name == "5D"
    assert converted_standard_error.name == "5D"


def test_non_unique_bin_edges_decorator_augments_only_the_source_value_error() -> None:
    @utils.non_unique_bin_edges_error
    def duplicate_edges() -> None:
        raise ValueError("Bin edges must be unique: array([1.0, 1.0])")

    @utils.non_unique_bin_edges_error
    def unrelated_value_error() -> None:
        raise ValueError("different error")

    with pytest.raises(ValueError, match="Decrease the number of quantiles"):
        duplicate_edges()
    with pytest.raises(ValueError, match="^different error$"):
        unrelated_value_error()


def test_rethrow_appends_context_to_the_original_exception() -> None:
    error = ValueError("base")

    with pytest.raises(ValueError, match="^base context$") as captured:
        utils.rethrow(error, " context")

    assert captured.value is error
    assert error.args == ("base context",)


def test_print_table_uses_lazy_ipython_display_and_restores_format(monkeypatch: pytest.MonkeyPatch) -> None:
    import IPython.display

    displayed: list[pd.DataFrame] = []
    source = pd.DataFrame({"value": [1.23456]})
    original_format = pd.get_option("display.float_format")
    monkeypatch.setattr(IPython.display, "display", displayed.append)

    result = utils.print_table(source, name="Example", fmt="{:.2f}")

    assert result is None
    assert len(displayed) == 1
    assert displayed[0] is source
    assert source.columns.name == "Example"
    assert pd.get_option("display.float_format") is original_format
