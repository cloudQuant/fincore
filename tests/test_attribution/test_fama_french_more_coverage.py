from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.attribution import fama_french
from fincore.attribution.fama_french import FamaFrenchModel


@pytest.fixture(autouse=True)
def _reset_ff_provider() -> None:
    # Keep tests isolated: provider and cache are module-level globals.
    fama_french.set_ff_provider(None)
    yield
    fama_french.set_ff_provider(None)


def _make_factor_data(index: pd.Index, *, include_mom: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    out = pd.DataFrame(
        {
            "MKT": rng.normal(0, 0.01, len(index)),
            "SMB": rng.normal(0, 0.01, len(index)),
            "HML": rng.normal(0, 0.01, len(index)),
            "RMW": rng.normal(0, 0.01, len(index)),
            "CMA": rng.normal(0, 0.01, len(index)),
        },
        index=index,
    )
    if include_mom:
        out["MOM"] = rng.normal(0, 0.01, len(index))
    return out


def test_model_type_4factor_mom_and_unknown_model_type_raises() -> None:
    m = FamaFrenchModel(model_type="4factor_mom")
    assert m.factors == ["MKT", "SMB", "HML", "MOM"]

    with pytest.raises(ValueError, match="Unknown model_type"):
        FamaFrenchModel(model_type="nope")


def test_fit_accepts_single_column_dataframe_and_unknown_method_raises() -> None:
    idx = pd.RangeIndex(30)
    returns = pd.DataFrame({"r": np.linspace(-0.01, 0.02, len(idx))}, index=idx)
    factor_data = _make_factor_data(idx)

    model = FamaFrenchModel(model_type="5factor")
    out = model.fit(returns, factor_data, method="ols")
    assert isinstance(out["alpha"], float)

    with pytest.raises(ValueError, match="Unknown method"):
        model.fit(returns, factor_data, method="gls")


def test_fit_newey_west_handles_lags_longer_than_sample() -> None:
    idx = pd.RangeIndex(3)
    returns = pd.Series([0.01, -0.02, 0.03], index=idx)
    factor_data = _make_factor_data(idx)

    model = FamaFrenchModel(model_type="5factor")
    # newey_west_lags is intentionally larger than the sample so the
    # "else: acorr_vals.append(0.0)" path is exercised.
    out = model.fit(returns, factor_data, newey_west_lags=10)
    assert out["std_errors"].shape[0] == 6


def test_fit_simple_ols_std_errors_path_when_newey_west_disabled() -> None:
    idx = pd.RangeIndex(60)
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    factor_data = _make_factor_data(idx)

    model = FamaFrenchModel(model_type="5factor")
    out = model.fit(returns, factor_data, newey_west_lags=0)
    assert out["std_errors"].shape[0] == 6


def test_predict_raises_before_fit() -> None:
    idx = pd.RangeIndex(5)
    model = FamaFrenchModel()
    factor_data = _make_factor_data(idx)
    with pytest.raises(RuntimeError, match="fit before prediction"):
        model.predict(factor_data)


def test_get_factor_exposures_with_rolling_window_exercises_window_bounds() -> None:
    idx = pd.RangeIndex(12)
    rng = np.random.default_rng(11)
    returns = pd.DataFrame({"asset": rng.normal(0, 0.02, len(idx))}, index=idx)
    factor_data = _make_factor_data(idx)

    model = FamaFrenchModel(model_type="5factor")
    exposures = model.get_factor_exposures(returns, factor_data, rolling_window=10)

    assert exposures.index.equals(returns.index)
    assert list(exposures.columns) == ["alpha", "MKT", "SMB", "HML", "RMW", "CMA"]
    # At least one window should have enough observations to fit.
    assert exposures.dropna(how="all").shape[0] > 0


def test_attribution_decomposition_returns_expected_keys() -> None:
    idx = pd.RangeIndex(80)
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    factor_data = _make_factor_data(idx)

    model = FamaFrenchModel(model_type="5factor", risk_free_rate=0.001)
    out = model.attribution_decomposition(returns, factor_data)

    assert "alpha" in out
    assert "specific_return" in out
    assert "common_return" in out
    assert "unexplained" in out
    assert "MKT_attribution" in out


def test_fetch_ff_factors_with_provider_argument_and_copy_flag() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")

    def provider(_start: str, _end: str, _library: str) -> pd.DataFrame:
        return pd.DataFrame({"MKT": [0.01, 0.02], "SMB": [0.0, 0.0], "HML": [0.0, 0.0]}, index=idx)

    df_copy = fama_french.fetch_ff_factors("2020-01-01", "2020-01-31", provider=provider, copy=True)
    df_nocopy = fama_french.fetch_ff_factors("2020-01-01", "2020-01-31", provider=provider, copy=False)

    assert df_copy is not df_nocopy
    assert df_copy.equals(df_nocopy)


def test_fetch_ff_factors_provider_must_return_dataframe() -> None:
    def bad_provider(_start: str, _end: str, _library: str):  # type: ignore[no-untyped-def]
        return {"not": "a dataframe"}

    with pytest.raises(TypeError, match="must return a pandas DataFrame"):
        fama_french.fetch_ff_factors("2020-01-01", "2020-01-31", provider=bad_provider)


def test_cached_fetch_type_error_and_cache_clear() -> None:
    calls: dict[str, int] = {"n": 0}

    def provider(_start: str, _end: str, _library: str):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return ["not a df"]

    fama_french.set_ff_provider(provider)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must return a pandas DataFrame"):
        fama_french.fetch_ff_factors("2020-01-01", "2020-01-31")

    # Clearing cache should be a safe no-op even after an exception path.
    fama_french.clear_ff_factor_cache()
    assert calls["n"] == 1

