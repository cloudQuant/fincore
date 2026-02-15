import numpy as np
import pandas as pd
import pytest

from fincore.simulation.base import SimResult, annualize, estimate_parameters, validate_returns


def test_validate_returns_errors_and_success() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_returns(np.array([]))

    with pytest.raises(ValueError, match="NaN"):
        validate_returns(np.array([0.0, np.nan]))

    out = validate_returns([0.01, -0.02])
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)


def test_annualize_supported_periods_and_default() -> None:
    assert annualize(1.0, period="daily") == pytest.approx(np.sqrt(252))
    assert annualize(1.0, period="weekly") == pytest.approx(np.sqrt(52))
    assert annualize(1.0, period="monthly") == pytest.approx(np.sqrt(12))
    assert annualize(1.0, period="yearly") == pytest.approx(1.0)
    # Unknown period falls back to daily.
    assert annualize(1.0, period="unknown") == pytest.approx(np.sqrt(252))


def test_estimate_parameters_drops_nan_and_errors_if_all_nan() -> None:
    with pytest.raises(ValueError, match="No valid returns"):
        estimate_parameters(np.array([np.nan, np.nan]))

    drift, vol = estimate_parameters(np.array([0.01, np.nan, -0.01]), frequency=252)
    assert isinstance(drift, float)
    assert isinstance(vol, float)
    assert vol >= 0.0


def test_sim_result_properties_var_cvar_and_to_dataframe() -> None:
    paths = np.array([[1.0, 0.9], [1.0, 1.1], [1.0, 0.8]])
    sr = SimResult(paths)
    assert sr.n_paths == 3
    assert sr.horizon == 2
    assert sr.var(alpha=0.5) == pytest.approx(0.9)
    assert sr.cvar(alpha=0.5) <= sr.var(alpha=0.5)
    df = sr.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == paths.shape

    sr1 = SimResult(np.array([1.0, 2.0, 3.0]))
    assert sr1.n_paths == 1
    assert sr1.horizon == 3

