from __future__ import annotations

import numpy as np
import pandas as pd


def test_perf_stats_bootstrap_resolves_callable_and_skips_none_entries(monkeypatch) -> None:
    """Cover perf_stats_bootstrap's internal stat resolution branches.

    This test patches the stat function lists to include:
    - a callable entry (callable branch)
    - a None entry (fallthrough + skip branch)
    and replaces calc_bootstrap with a tiny deterministic stub to keep the test fast.
    """
    import fincore.constants.style as style_const
    import fincore.metrics.perf_stats as perf_mod

    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    returns = pd.Series([0.01, 0.0, -0.01, 0.02, 0.0], index=idx)

    def my_stat(x: pd.Series) -> float:
        return float(np.nanmean(np.asarray(x)))

    monkeypatch.setattr(style_const, "SIMPLE_STAT_FUNCS", [my_stat, None], raising=True)
    monkeypatch.setattr(style_const, "FACTOR_STAT_FUNCS", [None], raising=True)
    monkeypatch.setattr(style_const, "STAT_FUNC_NAMES", {}, raising=True)

    calls: list[object] = []

    def fake_calc_bootstrap(func, _returns, *args, **kwargs):
        calls.append(func)
        return np.array([0.0, 1.0], dtype=float)

    monkeypatch.setattr(perf_mod, "calc_bootstrap", fake_calc_bootstrap, raising=True)

    out = perf_mod.perf_stats_bootstrap(returns, factor_returns=returns, return_stats=True)

    # Only the callable entry should produce a column; None entries are skipped.
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == 1
    assert any(call is my_stat for call in calls)
