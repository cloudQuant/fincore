from __future__ import annotations

import numpy as np
import pandas as pd

from fincore.metrics import risk as rm


def test_value_at_risk_empty_returns_nan():
    assert np.isnan(rm.value_at_risk(pd.Series([], dtype=float)))


def test_tail_ratio_all_nan_returns_nan():
    assert np.isnan(rm.tail_ratio(pd.Series([np.nan, np.nan])))


def test_residual_risk_returns_nan_when_aligned_observations_less_than_two():
    # Two input series with disjoint indices should align to length 0.
    r = pd.Series([0.01, -0.02], index=pd.date_range("2020-01-01", periods=2, freq="B"))
    f = pd.Series([0.01, -0.02], index=pd.date_range("2020-02-01", periods=2, freq="B"))
    assert np.isnan(rm.residual_risk(r, f))


def test_residual_risk_ndarray_returns_nan_when_fewer_than_two_finite_pairs():
    r = np.array([0.01, np.nan, 0.02], dtype=float)
    f = np.array([0.01, 0.02, np.nan], dtype=float)
    assert np.isnan(rm.residual_risk(r, f))


def test_residual_risk_ndarray_computes_for_perfect_correlation():
    r = np.array([0.01, 0.02, -0.01, 0.03], dtype=float)
    f = r.copy()
    out = rm.residual_risk(r, f)
    assert isinstance(out, (float, np.floating))
    assert out >= 0
    assert out < 1e-12


def test_var_excess_return_returns_nan_when_var_is_zero():
    # Flat returns => VaR == 0, should produce NaN by definition.
    r = pd.Series([0.0] * 252, index=pd.date_range("2020-01-01", periods=252, freq="B"))
    assert np.isnan(rm.var_excess_return(r))


def test_trading_value_at_risk_handles_short_and_normal_inputs():
    assert np.isnan(rm.trading_value_at_risk([0.01]))
    out = rm.trading_value_at_risk([0.01, -0.02, 0.03], sigma=1.0)
    # mean - sigma*std(ddof=1)
    expected = np.mean([0.01, -0.02, 0.03]) - np.std([0.01, -0.02, 0.03], ddof=1)
    assert np.isclose(out, expected)


def test_gpd_risk_estimates_covers_attributeerror_fallback_and_scale_only_path(monkeypatch):
    # Force the AttributeError fallback for `.to_numpy()` to cover that branch.
    monkeypatch.setattr(pd.Series, "to_numpy", lambda _self: (_ for _ in ()).throw(AttributeError()))

    calls = {"n": 0}

    def fake_minimize(fun, x0, method):  # noqa: ARG001
        # Make sure the scale-only loglikelihood branch runs at least once.
        fun(np.array([1.0, 0.0]))
        calls["n"] += 1
        if calls["n"] == 1:
            # Cover the exception handler in the optimization loop.
            raise ValueError("boom")

        class _Result:
            success = True
            x = np.array([1.0, 0.5])

        # Also exercise the scale+shape path.
        fun(_Result.x)
        return _Result()

    # Patch inside the function's local import (`from scipy import optimize`).
    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "minimize", fake_minimize)

    returns = pd.Series([-0.30, 0.10, -0.25, 0.05, -0.10, 0.02])
    out = rm.gpd_risk_estimates(returns, var_p=0.01)
    assert isinstance(out, pd.Series)
    assert len(out) == 5


def test_beta_fragility_heuristic_returns_nan_when_alignment_too_short():
    # Both inputs have >= 3 points, but the *overlap after alignment* has only 2 observations.
    idx_r = pd.date_range("2020-01-01", periods=4, freq="B")
    r = pd.Series([0.01, 0.02, -0.01, 0.03], index=idx_r)

    # Use a 1-col DataFrame to trigger the Series-vs-DataFrame alignment code path
    # in fincore.metrics.basic.aligned_series (which intersects indices).
    idx_f = idx_r[-2:].append(pd.date_range("2020-02-01", periods=2, freq="B"))
    f = pd.DataFrame({"mkt": [0.01, -0.02, 0.005, -0.005]}, index=idx_f)

    assert np.isnan(rm.beta_fragility_heuristic(r, f))
