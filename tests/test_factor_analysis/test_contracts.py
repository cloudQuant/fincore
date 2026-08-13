"""Enhanced factor-data contracts that are intentionally independent of the facade."""

from __future__ import annotations

import inspect

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
