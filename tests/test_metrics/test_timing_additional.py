import numpy as np
import pandas as pd

from fincore.metrics import timing as timing_mod


def test_treynor_mazuy_and_henriksson_early_return_nan() -> None:
    # Need at least 10 aligned observations.
    r = pd.Series([0.01] * 9)
    f = pd.Series([0.02] * 9)
    assert np.isnan(timing_mod.treynor_mazuy_timing(r, f))
    assert np.isnan(timing_mod.henriksson_merton_timing(r, f))


def test_treynor_mazuy_and_henriksson_exception_branch(monkeypatch) -> None:
    r = pd.Series([0.01] * 12)
    f = pd.Series([0.02] * 12)

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(np.linalg, "lstsq", boom)
    assert np.isnan(timing_mod.treynor_mazuy_timing(r, f))
    assert np.isnan(timing_mod.henriksson_merton_timing(r, f))


def test_market_timing_return_nan_when_gamma_nan() -> None:
    r = pd.Series([0.01] * 9)
    f = pd.Series([0.02] * 9)
    assert np.isnan(timing_mod.market_timing_return(r, f))


def test_cornell_timing_guard_rails_and_exception(monkeypatch) -> None:
    r_short = pd.Series([0.01] * 9)
    f_short = pd.Series([0.02] * 9)
    assert np.isnan(timing_mod.cornell_timing(r_short, f_short))

    # Enough length, but after NaN filtering we have < 10 valid obs.
    r = pd.Series([np.nan] * 11 + [0.01])
    f = pd.Series([np.nan] * 11 + [0.02])
    assert np.isnan(timing_mod.cornell_timing(r, f))

    # Exception branch inside regression.
    r2 = pd.Series([0.01] * 12)
    f2 = pd.Series([0.02] * 12)

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(np.linalg, "lstsq", boom)
    assert np.isnan(timing_mod.cornell_timing(r2, f2))


def test_extract_interesting_date_ranges_continue_and_exception(monkeypatch) -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B", tz="UTC")
    returns = pd.Series([0.01, -0.01, 0.02, 0.0, 0.01], index=idx)

    # One empty slice + one invalid slice (exception) should return empty.
    monkeypatch.setattr(
        timing_mod,
        "PERIODS",
        {
            "Empty": ("1900-01-01", "1900-01-02"),
            "Bad": ("not-a-date", object()),
        },
    )
    out = timing_mod.extract_interesting_date_ranges(returns)
    assert out == {}


def test_henriksson_alignment_can_reduce_length_below_threshold() -> None:
    idx_r = pd.date_range("2024-01-01", periods=12, freq="B", tz="UTC")
    idx_f = pd.date_range("2024-02-01", periods=5, freq="B", tz="UTC")
    r = pd.Series([0.01] * len(idx_r), index=idx_r)
    f = pd.Series([0.02] * len(idx_f), index=idx_f)
    # After aligning on intersection, length is < 10 => NaN.
    assert np.isnan(timing_mod.henriksson_merton_timing(r, f))


def test_extract_interesting_date_ranges_adds_non_empty_range(monkeypatch) -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    returns = pd.Series(np.linspace(0.01, -0.01, len(idx)), index=idx)
    monkeypatch.setattr(timing_mod, "PERIODS", {"Here": (idx[1], idx[3])})
    out = timing_mod.extract_interesting_date_ranges(returns)
    assert "Here" in out
