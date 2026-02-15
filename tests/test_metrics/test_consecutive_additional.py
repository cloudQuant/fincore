import numpy as np
import pandas as pd

from fincore.metrics import consecutive as cons_mod


def test_max_consecutive_run_no_match_returns_zero() -> None:
    s = pd.Series([0.0, 0.0, 0.0])
    assert cons_mod._max_consecutive_run(s, lambda x: x > 0) == 0


def test_consecutive_stats_empty_and_basic() -> None:
    empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    out = cons_mod.consecutive_stats(empty)
    assert np.isnan(out["max_consecutive_up_days"])
    assert np.isnan(out["max_consecutive_down_months"])

    idx = pd.date_range("2024-01-01", periods=40, freq="B", tz="UTC")
    returns = pd.Series([0.01] * 3 + [-0.01] * 2 + [0.0] * (len(idx) - 5), index=idx)
    out2 = cons_mod.consecutive_stats(returns)
    assert out2["max_consecutive_up_days"] >= 3
    assert out2["max_consecutive_down_days"] >= 2


def test_gain_loss_and_week_month_guards_and_dates() -> None:
    empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    assert np.isnan(cons_mod.max_consecutive_up_weeks(empty))
    assert np.isnan(cons_mod.max_consecutive_down_weeks(empty))
    assert np.isnan(cons_mod.max_consecutive_up_months(empty))
    assert np.isnan(cons_mod.max_consecutive_down_months(empty))
    assert np.isnan(cons_mod.max_single_day_gain(empty))
    assert np.isnan(cons_mod.max_single_day_loss(empty))
    assert cons_mod.max_single_day_gain_date(empty) is None
    assert cons_mod.max_single_day_loss_date(empty) is None
    assert cons_mod.max_consecutive_up_start_date(empty) is None
    assert cons_mod.max_consecutive_up_end_date(empty) is None
    assert cons_mod.max_consecutive_down_start_date(empty) is None
    assert cons_mod.max_consecutive_down_end_date(empty) is None

    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    # No positive days.
    r0 = pd.Series([0.0, 0.0, -0.01, 0.0, -0.02, 0.0], index=idx)
    assert np.isnan(cons_mod.max_consecutive_gain(r0))
    assert cons_mod.max_consecutive_up_start_date(r0) is None
    assert cons_mod.max_consecutive_up_end_date(r0) is None

    # No negative days.
    r1 = pd.Series([0.0, 0.01, 0.0, 0.02, 0.0, 0.03], index=idx)
    assert np.isnan(cons_mod.max_consecutive_loss(r1))
    assert cons_mod.max_consecutive_down_start_date(r1) is None
    assert cons_mod.max_consecutive_down_end_date(r1) is None

