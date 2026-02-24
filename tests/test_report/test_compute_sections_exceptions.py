import pandas as pd

import fincore
from fincore.report.compute import compute_sections


def test_compute_sections_handles_turnover_and_leverage_exceptions(monkeypatch) -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="B", tz="UTC")
    returns = pd.Series([0.001 if (i % 2 == 0) else -0.0008 for i in range(len(idx))], index=idx, name="strategy")

    positions = pd.DataFrame({"AAA": 100.0, "cash": 10.0}, index=idx)
    tx_idx = pd.to_datetime(["2024-02-01 10:00:00", "2024-02-02 11:00:00"], utc=True)
    transactions = pd.DataFrame({"amount": [10, -5], "price": [10.0, 11.0], "symbol": ["AAA", "AAA"]}, index=tx_idx)

    def _boom(*_args, **_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(fincore.Empyrical, "get_turnover", staticmethod(_boom))
    monkeypatch.setattr(fincore.Empyrical, "gross_lev", staticmethod(_boom))

    s = compute_sections(
        returns=returns,
        benchmark_rets=None,
        positions=positions,
        transactions=transactions,
        trades=None,
        rolling_window=20,
    )

    assert "perf_stats" in s
    assert "extended_stats" in s
    assert s["has_positions"] is True
    assert s["has_transactions"] is True


def test_compute_sections_summary_text_handles_nan_drawdown(monkeypatch) -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="B", tz="UTC")
    returns = pd.Series([0.001 if (i % 2 == 0) else -0.0008 for i in range(len(idx))], index=idx, name="strategy")

    monkeypatch.setattr(fincore.Empyrical, "max_drawdown", staticmethod(lambda _r: float("nan")))

    s = compute_sections(
        returns=returns,
        benchmark_rets=None,
        positions=None,
        transactions=None,
        trades=None,
        rolling_window=20,
    )

    assert "summary_text" in s
    assert "Max drawdown" in s["summary_text"]
