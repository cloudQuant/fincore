"""Numerical and workflow tests for enhanced factor-model IC inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from fincore.factor_analysis.analysis import analyze_factor
from fincore.factor_analysis.data import prepare_factor_data_from_forward_returns
from fincore.factor_analysis.inference import factor_model_inference, information_coefficient_inference


def _information_coefficients() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "1D": [-0.03, 0.02, 0.04, 0.01, -0.01],
            "5D": [0.01, 0.02, 0.03, 0.02, 0.04],
            "10D": [0.02, np.nan, np.nan, np.nan, np.nan],
        },
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
    )


def test_information_coefficient_inference_matches_scipy_and_statsmodels_oracles() -> None:
    information = _information_coefficients()

    result = information_coefficient_inference(information, alpha=0.10)

    expected_p_values = pd.Series(
        {period: stats.ttest_1samp(information[period].dropna(), popmean=0.0).pvalue for period in ("1D", "5D")},
        name="p_value",
    )
    expected_rejected, expected_adjusted, _, _ = multipletests(
        expected_p_values.to_numpy(),
        alpha=0.10,
        method="fdr_bh",
    )

    assert result.alpha == 0.10
    assert result.method == "two-sided-student-t+benjamini-hochberg"
    assert result.n_hypotheses == 3
    assert result.n_tested == 2
    assert result.hypotheses["n_observations"].to_dict() == {"1D": 5, "5D": 5, "10D": 1}
    assert result.hypotheses["testable"].to_dict() == {"1D": True, "5D": True, "10D": False}
    np.testing.assert_allclose(
        result.hypotheses.loc[expected_p_values.index, "p_value"],
        expected_p_values,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.hypotheses.loc[expected_p_values.index, "adjusted_p_value"],
        expected_adjusted,
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.hypotheses.loc[expected_p_values.index, "rejected"].to_list() == expected_rejected.tolist()
    assert np.isnan(result.hypotheses.loc["10D", "p_value"])
    assert np.isnan(result.hypotheses.loc["10D", "adjusted_p_value"])
    assert not result.hypotheses.loc["10D", "rejected"]


def _prepared_factor_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, tz="UTC")
    assets = ("A", "B", "C", "D")
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    factor = pd.Series(np.tile([-1.0, -0.25, 0.25, 1.0], len(dates)), index=index, name="factor")
    forward_returns = pd.DataFrame(
        {
            "1D": np.tile([-0.02, -0.01, 0.01, 0.02], len(dates)),
            "5D": np.tile([-0.01, -0.005, 0.005, 0.01], len(dates)),
        },
        index=index,
    )
    return prepare_factor_data_from_forward_returns(
        factor,
        forward_returns,
        quantiles=2,
        max_loss=1.0,
    ).data


def test_factor_model_inference_consumes_the_aggregate_enhanced_ic_snapshot() -> None:
    model = analyze_factor(_prepared_factor_data(), periods=("1D", "5D"), include_portfolio_inputs=False)

    from_model = factor_model_inference(model)
    direct = information_coefficient_inference(model.aggregate_information_coefficient)

    pd.testing.assert_frame_equal(from_model.hypotheses, direct.hypotheses)
    assert from_model.alpha == direct.alpha
    assert from_model.n_tested == direct.n_tested


def test_information_coefficient_inference_fails_closed_for_invalid_or_ambiguous_inputs() -> None:
    duplicate_columns = pd.DataFrame([[0.01, 0.02], [0.02, 0.03]], columns=["1D", "1D"])

    with pytest.raises(ValueError, match="duplicate"):
        information_coefficient_inference(duplicate_columns)
    with pytest.raises(ValueError, match="infinite"):
        information_coefficient_inference(pd.DataFrame({"1D": [0.01, np.inf]}))
    with pytest.raises(ValueError, match="alpha"):
        information_coefficient_inference(_information_coefficients(), alpha=0.0)
    with pytest.raises(TypeError, match="FactorAnalysisModel"):
        factor_model_inference(object())
