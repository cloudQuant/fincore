"""Enhanced factor-data contracts that are intentionally independent of the facade."""

from __future__ import annotations

import inspect
import typing

import numpy as np
import pandas as pd
import pytest


def test_prepare_factor_data_returns_structured_loss_report_without_stdout(
    raw_factor: pd.Series, prices: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    """The enhanced entry point returns structured diagnostics and stays silent."""

    from fincore.factor_analysis.data import FactorLossReport, PreparedFactorData, prepare_factor_data

    result = prepare_factor_data(raw_factor, prices, periods=(1, 5), max_loss=1)

    assert isinstance(result, PreparedFactorData)
    assert isinstance(result.loss_report, FactorLossReport)
    assert result.loss_report.total_loss >= 0
    assert result.data.index.names == ["date", "asset"]
    assert capsys.readouterr().out == ""
    assert "profile" not in inspect.signature(prepare_factor_data).parameters


def test_max_loss_boundary_uses_structured_report_and_legacy_projection(
    raw_factor: pd.Series, prices: pd.DataFrame
) -> None:
    """The facade keeps its legacy exception while enhanced callers retain loss detail."""

    from fincore.alphalens import utils as legacy_utils
    from fincore.factor_analysis.data import FactorLossExceededError, prepare_factor_data

    with pytest.raises(legacy_utils.MaxLossExceededError, match=r"max_loss .* exceeded"):
        legacy_utils.get_clean_factor_and_forward_returns(raw_factor, prices, periods=(1, 5), max_loss=0)

    with pytest.raises(FactorLossExceededError) as enhanced_error:
        prepare_factor_data(raw_factor, prices, periods=(1, 5), max_loss=0)

    assert enhanced_error.value.report.total_loss > 0


def test_prepared_factor_data_copies_public_inputs(raw_factor: pd.Series, prices: pd.DataFrame) -> None:
    """Caller-owned factor and price data remain unchanged after kernel execution."""

    from fincore.factor_analysis.data import prepare_factor_data

    factor_before = raw_factor.copy(deep=True)
    prices_before = prices.copy(deep=True)
    prepare_factor_data(raw_factor, prices, periods=(1,), max_loss=1)

    pd.testing.assert_series_equal(raw_factor, factor_before)
    pd.testing.assert_frame_equal(prices, prices_before)


def test_clean_factor_fixture_is_real_and_returns_mutation_isolated(clean_factor_data: pd.DataFrame) -> None:
    """The shared fixture now uses the kernel and never leaks cached mutable rows."""

    assert clean_factor_data.index.names == ["date", "asset"]
    assert {"factor", "factor_quantile", "group"} <= set(clean_factor_data.columns)
    original = clean_factor_data.iloc[0, 0]
    clean_factor_data.iloc[0, 0] = -999.0
    from tests.compat.alphalens.conftest import _shared_clean_factor_data

    assert _shared_clean_factor_data().iloc[0, 0] == original


def test_strict_utils_runtime_type_hints_resolve_public_collection_annotations() -> None:
    """Postponed facade annotations remain resolvable by runtime contract tooling."""

    from fincore.alphalens import utils

    for function in (utils.compute_forward_returns, utils.get_clean_factor, utils.get_clean_factor_and_forward_returns):
        hints = typing.get_type_hints(function)
        assert "factor" in hints
        assert "return" in hints


def _all_nan_factor_inputs() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Frozen all-NaN case observed from pinned commit 3fa17ad."""

    index = pd.MultiIndex.from_product(
        (pd.date_range("2024-01-02", periods=2, name="date"), ("A", "B")), names=("date", "asset")
    )
    factor = pd.Series(np.nan, index=index, name="factor")
    forward_returns = pd.DataFrame({"1D": (0.1, 0.2, 0.3, 0.4)}, index=index)
    prices = pd.DataFrame(
        {"A": (10.0, 11.0, 12.0, 13.0), "B": (20.0, 19.0, 18.0, 17.0)},
        index=pd.date_range("2024-01-02", periods=4, name="date"),
    )
    return factor, forward_returns, prices


_PINNED_ALL_NAN_STDOUT = (
    "Dropped 100.0% entries from factor data: 100.0% in forward returns computation and 0.0% in binning phase "
    "(set max_loss=0 to see potentially suppressed Exceptions).\n"
    "max_loss is 100.0%, not exceeded: OK!\n"
)


def test_strict_get_clean_factor_projects_pinned_all_nan_empty_frame(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The strict adapter preserves the pinned empty-frame projection, not enhanced validation."""

    from fincore.alphalens import utils

    factor, forward_returns, _ = _all_nan_factor_inputs()
    result = utils.get_clean_factor(factor, forward_returns, max_loss=1)

    expected = pd.DataFrame(index=forward_returns.index[:0], columns=("1D", "factor", "factor_quantile"), dtype=float)
    pd.testing.assert_frame_equal(result, expected)
    assert capsys.readouterr().out == _PINNED_ALL_NAN_STDOUT


def test_strict_get_clean_factor_and_forward_returns_projects_pinned_all_nan_empty_frame(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The price-derived strict path has the same pinned all-NaN behavior."""

    from fincore.alphalens import utils

    factor, _, prices = _all_nan_factor_inputs()
    result = utils.get_clean_factor_and_forward_returns(factor, prices, periods=(1,), max_loss=1)

    expected = pd.DataFrame(index=factor.index[:0], columns=("1D", "factor", "factor_quantile"), dtype=float)
    pd.testing.assert_frame_equal(result, expected)
    assert capsys.readouterr().out == _PINNED_ALL_NAN_STDOUT


def _all_nan_groupby(kind: str, index: pd.MultiIndex) -> pd.Series | dict[str, str]:
    """Return a fresh pinned-groupby input for each strict projection case."""

    values = ("one", "two", "one", "two")
    if kind == "series":
        return pd.Series(values, index=index, name="group")
    if kind == "categorical":
        return pd.Series(
            pd.Categorical(values, categories=("two", "one", "unused"), ordered=True), index=index, name="group"
        )
    if kind == "dict":
        return {"A": "one", "B": "two"}
    raise AssertionError(f"unknown all-NaN groupby kind: {kind}")


def _pinned_empty_group_frame(index: pd.MultiIndex, categories: tuple[str, ...], *, ordered: bool) -> pd.DataFrame:
    """Frozen DataFrame projection observed from pinned 3fa17ad source bytes."""

    expected = pd.DataFrame(index=index[:0], columns=("1D", "factor"), dtype=float)
    category_index = pd.Index(categories, dtype="str")
    expected["group"] = pd.Series(pd.Categorical([], categories=category_index, ordered=ordered), index=expected.index)
    expected["factor_quantile"] = pd.Series(index=expected.index, dtype=float)
    return expected


@pytest.mark.parametrize(
    ("groupby_kind", "groupby_labels", "categories", "ordered"),
    (
        pytest.param("series", None, ("one", "two"), False, id="series"),
        pytest.param("categorical", None, ("two", "one", "unused"), True, id="categorical-series"),
        pytest.param("dict", None, (), False, id="dict"),
        pytest.param("series", {"one": "One", "two": "Two"}, ("One", "Two"), False, id="series-labels"),
        pytest.param("categorical", {"one": "One", "two": "Two"}, ("One", "Two"), False, id="categorical-labels"),
        pytest.param("dict", {"one": "One", "two": "Two"}, (), False, id="dict-labels"),
    ),
)
@pytest.mark.parametrize("entrypoint", ("direct", "prices"))
def test_strict_all_nan_groupby_projection_matches_frozen_pinned_frame(
    entrypoint: str,
    groupby_kind: str,
    groupby_labels: dict[str, str] | None,
    categories: tuple[str, ...],
    ordered: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Series/category/dict metadata survives the adapter-only empty projection."""

    from fincore.alphalens import utils

    factor, forward_returns, prices = _all_nan_factor_inputs()
    groupby = _all_nan_groupby(groupby_kind, factor.index)
    if entrypoint == "direct":
        result = utils.get_clean_factor(
            factor, forward_returns, groupby=groupby, groupby_labels=groupby_labels, max_loss=1
        )
        expected_index = forward_returns.index
    else:
        result = utils.get_clean_factor_and_forward_returns(
            factor, prices, groupby=groupby, groupby_labels=groupby_labels, periods=(1,), max_loss=1
        )
        expected_index = factor.index

    expected = _pinned_empty_group_frame(expected_index, categories, ordered=ordered)
    pd.testing.assert_frame_equal(result, expected)
    assert result["group"].dtype.name == "category"
    assert tuple(result["group"].cat.categories) == categories
    assert result["group"].cat.ordered is ordered
    assert capsys.readouterr().out == _PINNED_ALL_NAN_STDOUT


@pytest.mark.parametrize("entrypoint", ("direct", "prices"))
def test_strict_all_nan_groupby_labels_validate_before_stdout_or_empty_success(
    entrypoint: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pinned group-label validation precedes the all-NaN facade short circuit."""

    from fincore.alphalens import utils

    factor, forward_returns, prices = _all_nan_factor_inputs()
    groupby = _all_nan_groupby("series", factor.index)
    with pytest.raises(KeyError, match=r"groups \['two'\] not in passed group names"):
        if entrypoint == "direct":
            utils.get_clean_factor(
                factor,
                forward_returns,
                groupby=groupby,
                groupby_labels={"one": "One"},
                max_loss=1,
            )
        else:
            utils.get_clean_factor_and_forward_returns(
                factor,
                prices,
                groupby=groupby,
                groupby_labels={"one": "One"},
                periods=(1,),
                max_loss=1,
            )
    assert capsys.readouterr().out == ""


def test_strict_non_unique_bin_edges_projection_is_identical_for_each_entrypoint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """All strict entrypoints apply the pinned decorator's complete error text."""

    from fincore.alphalens import utils

    index = pd.MultiIndex.from_product(
        (pd.date_range("2024-01-02", periods=2, name="date"), ("A", "B")), names=("date", "asset")
    )
    factor = pd.Series(1.0, index=index, name="factor")
    factor_data = pd.DataFrame({"factor": factor})
    forward_returns = pd.DataFrame({"1D": (0.1, 0.2, 0.3, 0.4)}, index=index)
    prices = pd.DataFrame(
        {"A": (10.0, 11.0, 12.0), "B": (20.0, 19.0, 18.0)},
        index=pd.date_range("2024-01-02", periods=3, name="date"),
    )

    with pytest.raises(ValueError) as direct_error:
        utils.quantize_factor(factor_data, quantiles=2)
    with pytest.raises(ValueError) as clean_error:
        utils.get_clean_factor(factor, forward_returns, quantiles=2, max_loss=0)
    with pytest.raises(ValueError) as prices_error:
        utils.get_clean_factor_and_forward_returns(factor, prices, periods=(1,), quantiles=2, max_loss=0)

    expected = str(direct_error.value)
    assert expected.startswith("Bin edges must be unique")
    assert "An error occurred while computing bins/quantiles" in expected
    assert str(clean_error.value) == expected
    assert str(prices_error.value) == expected
    assert capsys.readouterr().out == ""


def test_strict_exception_identities_do_not_inherit_value_error() -> None:
    """Pinned public exception classes are direct Exception identities at the facade."""

    from fincore.alphalens import utils

    assert issubclass(utils.MaxLossExceededError, Exception)
    assert issubclass(utils.NonMatchingTimezoneError, Exception)
    assert not issubclass(utils.MaxLossExceededError, ValueError)
    assert not issubclass(utils.NonMatchingTimezoneError, ValueError)
