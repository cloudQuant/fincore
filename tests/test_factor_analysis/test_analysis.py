"""Public enhanced entry-point contracts for factor-analysis models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import pandas as pd


def test_analyze_factor_is_owned_by_its_analysis_module(clean_factor_data: pd.DataFrame) -> None:
    """The factor root is a namespace; the leaf entry point stays direct."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.models import FactorAnalysisModel

    model = analyze_factor(clean_factor_data, periods=("1D",), turnover_periods=(1,), include_portfolio_inputs=False)

    assert isinstance(model, FactorAnalysisModel)


def test_analyze_factor_accepts_clean_data_without_recleaning(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw factor cleaning remains the Task 3 boundary and is not repeated here."""

    import fincore.factor_analysis.data as data
    from fincore.factor_analysis.analysis import analyze_factor

    def forbidden_cleaning(*args: object, **kwargs: object) -> object:
        raise AssertionError("analyze_factor must receive pre-cleaned factor data")

    monkeypatch.setattr(data, "prepare_factor_data", forbidden_cleaning)
    model = analyze_factor(clean_factor_data, periods=("1D",), turnover_periods=(1,), include_portfolio_inputs=False)

    assert model.factor_data.shape == clean_factor_data.shape


def test_analyze_factor_rejects_unknown_or_empty_forward_period_selection(clean_factor_data: pd.DataFrame) -> None:
    """The immutable config cannot describe results not present in the snapshot."""

    from fincore.factor_analysis.analysis import analyze_factor

    with pytest.raises(ValueError, match="unknown forward periods"):
        analyze_factor(clean_factor_data, periods=("missing",), include_portfolio_inputs=False)
    with pytest.raises(ValueError, match="at least one forward period"):
        analyze_factor(clean_factor_data, periods=(), include_portfolio_inputs=False)


def test_analyze_factor_rejects_integer_capital_that_cannot_survive_float_math(
    clean_factor_data: pd.DataFrame,
) -> None:
    """Bridge capital is rejected before a nonrepresentable integer is rounded."""

    from fincore.factor_analysis.analysis import analyze_factor

    with pytest.raises(ValueError, match="exactly"):
        analyze_factor(
            clean_factor_data,
            periods=("1D",),
            include_portfolio_inputs=True,
            portfolio_capital=2**53 + 1,
        )


def test_incomplete_event_input_remains_optional(clean_factor_data: pd.DataFrame, prices: pd.DataFrame) -> None:
    """Event sections only exist when returns and a complete window are supplied."""

    from fincore.factor_analysis.analysis import analyze_factor

    event_returns = prices.pct_change(fill_method=None).fillna(0.0)
    without_window = analyze_factor(
        clean_factor_data,
        periods=("1D",),
        include_portfolio_inputs=False,
        event_returns=event_returns,
    )
    without_returns = analyze_factor(
        clean_factor_data,
        periods=("1D",),
        include_portfolio_inputs=False,
        event_before=1,
        event_after=2,
    )

    assert without_window.event_returns is None
    assert without_returns.event_returns is None
