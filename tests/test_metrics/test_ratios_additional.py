import numpy as np
import pandas as pd
import pytest

from fincore.metrics import ratios as ratios_mod


def test_sortino_ratio_uses_precomputed_downside_risk_branch() -> None:
    r = np.array([0.01, -0.02, 0.03, -0.01])
    out = ratios_mod.sortino_ratio(r, _downside_risk=0.5, period="daily", annualization=252)
    assert np.isfinite(out)


def test_adjusted_sharpe_ratio_nan_and_skew_kurt_fallbacks() -> None:
    # Force the early sharpe-is-nan return.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ratios_mod, "sharpe_ratio", lambda *_args, **_kwargs: np.nan)
        assert np.isnan(ratios_mod.adjusted_sharpe_ratio(np.array([0.01, 0.02, 0.03, 0.04])))

    # Force skew/kurtosis to NaN so the internal fallback-to-0 branches are taken.
    import fincore.metrics.stats as stats_mod

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ratios_mod, "sharpe_ratio", lambda *_args, **_kwargs: 1.0)
        mp.setattr(stats_mod, "skewness", lambda *_args, **_kwargs: np.nan)
        mp.setattr(stats_mod, "kurtosis", lambda *_args, **_kwargs: np.nan)

        out = ratios_mod.adjusted_sharpe_ratio(np.random.default_rng(0).normal(0.0, 0.01, 30))
        assert np.isfinite(out)

        out2 = ratios_mod.adjusted_sharpe_ratio(np.random.default_rng(1).normal(0.0, 0.01, 10))
        assert np.isfinite(out2)


def test_calmar_ratio_returns_nan_when_temp_infinite(monkeypatch) -> None:
    import fincore.metrics.drawdown as dd_mod
    import fincore.metrics.yearly as yearly_mod

    min_pos = float(np.nextafter(0.0, 1.0))
    monkeypatch.setattr(dd_mod, "max_drawdown", lambda *_args, **_kwargs: -min_pos)
    monkeypatch.setattr(yearly_mod, "annual_return", lambda *_args, **_kwargs: 1.0)

    r = np.array([0.01, -0.01, 0.02, -0.02])
    assert np.isnan(ratios_mod.calmar_ratio(r))


def test_mar_ratio_nan_when_all_nan_and_when_infinite(monkeypatch) -> None:
    assert np.isnan(ratios_mod.mar_ratio(np.array([np.nan, np.nan, np.nan, np.nan])))

    import fincore.metrics.drawdown as dd_mod

    min_pos = float(np.nextafter(0.0, 1.0))
    monkeypatch.setattr(dd_mod, "max_drawdown", lambda *_args, **_kwargs: -min_pos)
    r = np.array([0.01, 0.02, 0.03, 0.04])
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        assert np.isnan(ratios_mod.mar_ratio(r))


def test_mar_ratio_nan_when_no_clean_returns() -> None:
    """Test mar_ratio returns NaN when returns_clean has < 1 element (line 417)."""
    # mar_ratio filters NaNs, so if all are NaN, returns_clean is empty
    assert np.isnan(ratios_mod.mar_ratio(np.array([np.nan, np.nan, np.nan, np.nan, np.nan])))


def test_omega_ratio_annualization_one_and_required_return_invalid() -> None:
    r = np.array([0.02, -0.01, 0.03, -0.02])
    out = ratios_mod.omega_ratio(r, risk_free=0.0, required_return=0.01, annualization=1)
    assert np.isfinite(out)

    assert np.isnan(ratios_mod.omega_ratio(r, required_return=-1.0, annualization=252))


def test_cal_treynor_ratio_dataframe_beta_mask_and_series_output(monkeypatch) -> None:
    import fincore.metrics.alpha_beta as ab_mod
    import fincore.metrics.yearly as yearly_mod

    idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
    returns = pd.DataFrame({"A": [0.01, 0.0, 0.01, 0.0, 0.01], "B": [0.02, 0.0, 0.0, 0.0, 0.0]}, index=idx)
    factor = returns * 0.5

    monkeypatch.setattr(ab_mod, "beta_aligned", lambda *_args, **_kwargs: pd.Series({"A": 0.0, "B": 2.0}))
    monkeypatch.setattr(yearly_mod, "annual_return", lambda *_args, **_kwargs: pd.Series({"A": 0.1, "B": 0.2}))

    out = ratios_mod.cal_treynor_ratio(returns, factor, risk_free=0.0, period="daily", annualization=252)
    assert isinstance(out, pd.Series)
    assert np.isnan(out["A"])
    assert np.isfinite(out["B"])


def test_cal_treynor_ratio_2d_array_beta_else_branch(monkeypatch) -> None:
    """Test cal_treynor_ratio when b is array but ann_excess_return is not Series/DataFrame (line 603)."""
    import fincore.metrics.alpha_beta as ab_mod
    import fincore.metrics.yearly as yearly_mod

    returns = np.array([[0.01, 0.02], [0.0, 0.01], [0.01, 0.0], [0.0, 0.01], [0.01, 0.0]])
    factor = np.array([0.005, 0.0, 0.005, 0.0, 0.005])

    # Mock beta_aligned to return an array
    monkeypatch.setattr(ab_mod, "beta_aligned", lambda *_args, **_kwargs: np.array([1.5, 2.0]))
    # Mock annual_return to return scalar (not Series/DataFrame)
    monkeypatch.setattr(yearly_mod, "annual_return", lambda *_args, **_kwargs: 0.1)

    out = ratios_mod.cal_treynor_ratio(returns, factor, risk_free=0.0, period="daily", annualization=252)
    # Should return an array
    assert isinstance(out, np.ndarray)
    assert len(out) == 2


def test_cal_treynor_ratio_1d_scalar_beta_else_branch(monkeypatch) -> None:
    """Test cal_treynor_ratio when b is scalar in else branch (lines 606-607)."""
    import fincore.metrics.alpha_beta as ab_mod
    import fincore.metrics.yearly as yearly_mod

    returns = np.array([0.01, 0.0, 0.01, 0.0, 0.01])
    factor = np.array([0.005, 0.0, 0.005, 0.0, 0.005])

    # Mock beta_aligned to return a positive scalar (not Series/ndarray)
    # This is unusual but the code handles it
    monkeypatch.setattr(ab_mod, "beta_aligned", lambda *_args, **_kwargs: 1.5)
    monkeypatch.setattr(yearly_mod, "annual_return", lambda *_args, **_kwargs: 0.1)

    out = ratios_mod.cal_treynor_ratio(returns, factor, risk_free=0.0, period="daily", annualization=252)
    # Should return a scalar
    assert isinstance(out, (float, np.floating))


def test_m_squared_returns_nan_when_volatility_zero() -> None:
    r = np.zeros(50)
    f = np.random.default_rng(0).normal(0.0, 0.01, 50)
    assert np.isnan(ratios_mod.m_squared(r, f))


def test_sterling_ratio_and_burke_ratio_fallback_branches(monkeypatch) -> None:
    import fincore.metrics.drawdown as dd_mod

    # No drawdowns -> fallback to downside returns mean (line with downside mean).
    monkeypatch.setattr(dd_mod, "get_all_drawdowns", lambda *_args, **_kwargs: [])
    r = np.array([0.01, -0.02, 0.03, -0.01, 0.0, 0.01, -0.01, 0.02])
    assert np.isfinite(ratios_mod.sterling_ratio(r))
    assert np.isfinite(ratios_mod.burke_ratio(r))

    # Tiny drawdown risk -> inf/nan branch.
    monkeypatch.setattr(dd_mod, "get_all_drawdowns", lambda *_args, **_kwargs: [-1e-12])
    r_pos = np.array([0.01] * 50)
    assert np.isinf(ratios_mod.sterling_ratio(r_pos))
    assert np.isinf(ratios_mod.burke_ratio(r_pos))


def test_kappa_three_ratio_nan_when_not_enough_clean_obs() -> None:
    assert np.isnan(ratios_mod.kappa_three_ratio(np.array([np.nan, 0.01])))


def test_deflated_sharpe_ratio_logN_tiny_branch_and_denom_nonpositive() -> None:
    # Cover log_N < 1e-10 branch using a float very close to 1.
    r = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
    out = ratios_mod.deflated_sharpe_ratio(r, num_trials=1.0 + 1e-12)
    assert 0.0 <= out <= 1.0

    # denom_sq <= 0 branch
    r2 = np.array([10.0, -10.0, 10.0, -10.0, 10.0, -10.0])
    out2 = ratios_mod.deflated_sharpe_ratio(r2, num_trials=1000)
    assert 0.0 <= out2 <= 1.0


def test_sample_skewness_and_kurtosis_edge_cases() -> None:
    assert ratios_mod._sample_skewness([1.0, 2.0]) == 0.0
    assert ratios_mod._sample_skewness([1.0, 1.0, 1.0]) == 0.0

    assert ratios_mod._sample_excess_kurtosis([1.0, 2.0, 3.0]) == 0.0
    assert ratios_mod._sample_excess_kurtosis([1.0, 1.0, 1.0, 1.0]) == 0.0


def test_common_sense_ratio_and_stability_and_capture_edge_cases(monkeypatch) -> None:
    assert np.isnan(ratios_mod.common_sense_ratio([0.01]))

    assert np.isnan(ratios_mod.stability_of_timeseries([np.nan, 0.01]))

    assert np.isnan(ratios_mod._capture_aligned([], [], period="daily"))

    import fincore.metrics.yearly as yearly_mod

    monkeypatch.setattr(yearly_mod, "annual_return", lambda *_args, **_kwargs: 0.0)
    assert np.isnan(ratios_mod._capture_aligned([0.01, 0.02], [0.01, 0.02], period="daily"))


def test_up_down_capture_returns_nan_when_down_cap_zero(monkeypatch) -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    r = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01, 0.01], index=idx)
    f = pd.Series([0.01, 0.01, -0.01, -0.01, 0.01, -0.01], index=idx)

    monkeypatch.setattr(ratios_mod, "_capture_aligned", lambda *_args, **_kwargs: 0.0)
    assert np.isnan(ratios_mod.up_down_capture(r, f))


def test_cal_treynor_ratio_scalar_beta_2d_returns() -> None:
    """Test cal_treynor_ratio when returns is 2D but beta is scalar (lines 606-607)."""
    import fincore.metrics.alpha_beta as ab_mod
    import fincore.metrics.yearly as yearly_mod

    # 2D returns (multi-column)
    returns = np.array([[0.01, 0.02], [0.0, 0.01], [0.01, 0.0], [0.0, 0.01], [0.01, 0.0]])
    factor = np.array([0.005, 0.0, 0.005, 0.0, 0.005])

    # Mock beta_aligned to return a scalar (not Series/ndarray)
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(ab_mod, "beta_aligned", lambda *_args, **_kwargs: 1.5)
        monkeypatch.setattr(yearly_mod, "annual_return", lambda *_args, **_kwargs: 0.1)

        out = ratios_mod.cal_treynor_ratio(returns, factor, risk_free=0.0, period="daily", annualization=252)
        # Should return an array with both elements set
        assert isinstance(out, np.ndarray)
        assert len(out) == 2
        # Both elements should be finite since annual_return is 0.1 and beta is 1.5
        assert np.isfinite(out[0])
        assert np.isfinite(out[1])
    finally:
        monkeypatch.undo()


def test_calmar_ratio_with_all_nan_returns() -> None:
    """Test calmar_ratio returns NaN when all returns are NaN (line 417)."""
    returns = np.array([np.nan, np.nan, np.nan])
    factor_returns = np.array([0.01, 0.02, 0.015])

    result = ratios_mod.calmar_ratio(returns, factor_returns)

    # Should return NaN when max_drawdown is positive (no drawdown) or returns are all NaN
    assert np.isnan(result)


def test_calmar_ratio_with_empty_returns_after_nan_removal() -> None:
    """Test calmar_ratio when returns_clean is empty (line 417)."""
    # Create returns that become empty after NaN removal
    returns = pd.Series([np.nan, np.nan, np.nan])

    result = ratios_mod.calmar_ratio(returns)

    # Should return NaN when len(returns_clean) < 1
    assert np.isnan(result)
