"""Enhanced analysis input contract tests."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from fincore.contracts.analysis import AnalysisInput, PortfolioSemantics, SeriesSemantics
from fincore.exceptions import (
    AlignmentError,
    FincoreError,
    InputContractError,
    NumericalConvergenceError,
    ResourceLifecycleError,
    ResultContractError,
)


def test_analysis_input_is_copy_on_ingest() -> None:
    returns = pd.Series([0.01, -0.02, 0.03])
    ai = AnalysisInput.from_returns(returns)
    assert ai.returns is not returns
    assert list(ai.returns) == list(returns)
    assert ai.config_digest


def test_analysis_input_rejects_non_series() -> None:
    with pytest.raises(TypeError, match="Series"):
        AnalysisInput(returns=[0.01, -0.02])  # type: ignore[arg-type]


def test_semantics_defaults() -> None:
    semantics = SeriesSemantics()
    assert semantics.frequency == "daily"
    assert semantics.return_type == "simple"


def test_new_errors_subclass_fincore_error() -> None:
    for exc in (
        InputContractError,
        AlignmentError,
        NumericalConvergenceError,
        ResultContractError,
        ResourceLifecycleError,
    ):
        assert issubclass(exc, FincoreError)


def test_contextual_error_carries_operation_context() -> None:
    exc = InputContractError(
        "bad returns",
        operation_id="sharpe_ratio",
        parameter="returns",
        path="fincore.metrics.ratios.sharpe_ratio",
        profile="enhanced_v1",
    )
    assert exc.operation_id == "sharpe_ratio"
    assert exc.parameter == "returns"
    assert exc.path == "fincore.metrics.ratios.sharpe_ratio"
    assert exc.profile == "enhanced_v1"
    assert "sharpe_ratio" in str(exc)


def test_enhanced_errors_catchable_as_fincore_error() -> None:
    with pytest.raises(FincoreError):
        raise InputContractError("bad", operation_id="x")


def test_analysis_input_to_dict() -> None:
    returns = pd.Series([0.01, -0.02, 0.03])
    ai = AnalysisInput.from_returns(returns)
    d = ai.to_dict()
    assert d["profile"] == "enhanced_v1"
    assert d["frequency"] == "daily"
    assert "config_digest" in d


def test_portfolio_semantics_defaults() -> None:
    ps = PortfolioSemantics()
    assert ps.weight_timestamp_convention == "as_of"
    assert ps.gross_net == "net"


def test_fama_macbeth_preserves_the_iid_default_and_exposes_newey_west_keywords() -> None:
    from fincore.factor_analysis import fama_macbeth

    parameters = inspect.signature(fama_macbeth).parameters

    assert list(parameters) == ["returns", "exposures", "covariance", "newey_west_lags"]
    assert parameters["covariance"].default == "iid"
    assert parameters["newey_west_lags"].default == 1
    assert parameters["covariance"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["newey_west_lags"].kind is inspect.Parameter.KEYWORD_ONLY
