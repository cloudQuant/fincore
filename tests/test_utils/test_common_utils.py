"""Tests for fincore.utils.common_utils.

Target the pure helpers and the most error-prone branches.
"""

from __future__ import annotations

import contextlib
import io
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fincore.utils import common_utils as cu


def test_customize_decorator_set_context_false_calls_function_directly():
    calls = []

    @cu.customize
    def f(x):
        calls.append(x)
        return x + 1

    assert f(1, set_context=False) == 2
    assert calls == [1]


def test_customize_decorator_fallback_when_no_plotting_helpers_present():
    @cu.customize
    def f(x):
        return x + 1

    assert f(1) == 2


def test_customize_decorator_uses_plotting_context_and_axes_style():
    calls = []

    class Dummy:
        @contextlib.contextmanager
        def plotting_context(self):
            calls.append("plotting_context")
            yield

        @contextlib.contextmanager
        def axes_style(self):
            calls.append("axes_style")
            yield

        @cu.customize
        def f(self, x):
            calls.append("f")
            return x * 2

    d = Dummy()
    assert d.f(3) == 6
    assert calls == ["plotting_context", "axes_style", "f"]


def test_vectorize_decorator_series_and_dataframe():
    @cu.vectorize
    def plus_one(s):
        return s + 1

    s = pd.Series([1, 2, 3])
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    out_s = plus_one(s)
    out_df = plus_one(df)

    pd.testing.assert_series_equal(out_s, pd.Series([2, 3, 4]))
    pd.testing.assert_frame_equal(out_df, pd.DataFrame({"a": [2, 3], "b": [4, 5]}))


def test_clip_returns_to_benchmark_clips_when_out_of_range():
    idx_rets = pd.date_range("2020-01-01", periods=5, freq="D")
    idx_b = pd.date_range("2020-01-02", periods=3, freq="D")
    rets = pd.Series(range(5), index=idx_rets, dtype=float)
    bench = pd.Series(range(3), index=idx_b, dtype=float)

    out = cu.clip_returns_to_benchmark(rets, bench)
    assert out.index.equals(idx_b)


def test_clip_returns_to_benchmark_noop_when_already_aligned():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    rets = pd.Series(range(3), index=idx, dtype=float)
    bench = pd.Series(range(3), index=idx, dtype=float)
    out = cu.clip_returns_to_benchmark(rets, bench)
    assert out is rets


def test_to_utc_localize_and_convert_branches():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    df = pd.DataFrame({"x": [1, 2]}, index=idx)
    out1 = cu.to_utc(df.copy())
    assert str(out1.index.tz) == "UTC"

    df2 = pd.DataFrame({"x": [1, 2]}, index=idx.tz_localize("US/Eastern"))
    out2 = cu.to_utc(df2.copy())
    assert str(out2.index.tz) == "UTC"


def test_get_month_end_freq_both_branches_via_monkeypatch(monkeypatch):
    monkeypatch.setattr(cu.pd, "__version__", "2.1.9", raising=False)
    assert cu.get_month_end_freq() == "M"

    monkeypatch.setattr(cu.pd, "__version__", "2.2.0", raising=False)
    assert cu.get_month_end_freq() == "ME"


def test_make_timezone_aware_all_branches():
    ts_naive = pd.Timestamp("2020-01-01")
    ts_aware = pd.Timestamp("2020-01-01", tz="UTC")

    assert cu.make_timezone_aware(ts_naive, "UTC").tz is not None
    assert str(cu.make_timezone_aware(ts_aware, "US/Eastern").tz) == "US/Eastern"
    assert cu.make_timezone_aware(ts_aware, None).tz is None
    assert cu.make_timezone_aware(ts_naive, None).tz is None


def test_fallback_html_returns_input():
    assert cu._fallback_html("<b>x</b>") == "<b>x</b>"


def test_format_asset_returns_input_when_zipline_missing():
    assert cu.format_asset("AAPL") == "AAPL"


def test_register_return_func_and_get_symbol_rets_roundtrip():
    calls = []

    def f(symbol, start=None, end=None):
        calls.append((symbol, start, end))
        return pd.Series([1.0, 2.0])

    old = cu.SETTINGS["returns_func"]
    try:
        cu.register_return_func(f)
        out = cu.get_symbol_rets("AAPL", start="2020-01-01", end="2020-01-31")
        assert isinstance(out, pd.Series)
        assert calls == [("AAPL", "2020-01-01", "2020-01-31")]
    finally:
        cu.SETTINGS["returns_func"] = old


def test_default_returns_func_imports_empyrical_and_returns_float():
    rets = pd.Series([0.01, -0.005, 0.002], index=pd.date_range("2020-01-01", periods=3))
    out = cu._default_returns_func(rets)
    assert isinstance(out, float)


def test_get_utc_timestamp_localize_and_convert_branches():
    out1 = cu.get_utc_timestamp("2020-01-01")
    assert str(out1.tz) == "UTC"

    out2 = cu.get_utc_timestamp(pd.Timestamp("2020-01-01", tz="US/Eastern"))
    assert str(out2.tz) == "UTC"


def test_restride_rolling_window_happy_path_and_errors():
    a = np.arange(25).reshape(5, 5)
    out = cu.rolling_window(a, 2, mutable=True)
    assert out.shape == (4, 2, 5)

    out[0, 0, 0] = 999
    assert a[0, 0] == 999

    with pytest.raises(ValueError, match="0-length"):
        cu.rolling_window(a, 0)
    with pytest.raises(IndexError, match="scalar"):
        cu.rolling_window(np.array(1.0), 1)
    with pytest.raises(IndexError, match="window length"):
        cu.rolling_window(np.arange(3), 5)


def test_sample_colormap_returns_n_colors():
    colors = cu.sample_colormap("viridis", 3)
    assert len(colors) == 3


def test_sample_colormap_fallback_path_when_modern_registry_missing(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "colormaps", {}, raising=False)
    colors = cu.sample_colormap("viridis", 2)
    assert len(colors) == 2


def test_configure_legend_smoke(tmp_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="a")
    ax.plot([0, 1], [0, 2], label="b")
    ax.legend()

    cu.configure_legend(ax, change_colors=True, autofmt_xdate=True, rotation=10, ha="left")
    assert ax.get_legend() is not None


def test_1_bday_ago_returns_timestamp():
    ts = cu._1_bday_ago()
    assert isinstance(ts, pd.Timestamp)


def test_default_returns_func_is_passthrough():
    rets = pd.Series([1.0, 2.0])
    assert cu.default_returns_func(rets) is rets


def test_fallback_display_prints_to_stdout():
    buf = io.StringIO()
    old = cu.display
    try:
        cu.display = cu._fallback_display
        with contextlib.redirect_stdout(buf):
            cu.display("x", 1)
    finally:
        cu.display = old
    assert "x 1" in buf.getvalue()


def test_analyze_dataframe_differences_prints_for_identical_and_different(capsys):
    df1 = pd.DataFrame({"a": [1, 2]}, index=pd.date_range("2024-01-01", periods=2, freq="D"))
    df2 = df1.copy()
    cu.analyze_dataframe_differences(df1, df2)
    out = capsys.readouterr().out
    assert "The DataFrames are identical." in out

    df3 = pd.DataFrame({"a": [1, 999]}, index=df1.index)
    cu.analyze_dataframe_differences(df1, df3)
    out = capsys.readouterr().out
    assert "The DataFrames are not identical" in out


def test_analyze_dataframe_differences_prints_index_columns_dtype_and_metadata_differences(capsys):
    idx1 = pd.date_range("2024-01-01", periods=2, freq="D")
    idx2 = pd.date_range("2024-01-02", periods=2, freq="D")
    df1 = pd.DataFrame({"a": [1, 2]}, index=idx1)
    df2 = pd.DataFrame({"b": [1.0, 2.0]}, index=idx2)

    cu.analyze_dataframe_differences(df1, df2)
    out = capsys.readouterr().out
    assert "Indices are different" in out
    assert "Columns are different" in out
    assert "Dtypes are different" in out


def test_analyze_series_differences_prints_for_identical_and_different(capsys):
    s1 = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, freq="D"))
    s2 = s1.copy()
    cu.analyze_series_differences(s1, s2)
    out = capsys.readouterr().out
    assert "The Series are identical." in out

    s3 = pd.Series([1.0, 999.0], index=s1.index)
    cu.analyze_series_differences(s1, s3)
    out = capsys.readouterr().out
    assert "The Series are not identical" in out


def test_analyze_series_differences_prints_index_dtype_and_metadata_differences(capsys):
    idx1 = pd.date_range("2024-01-01", periods=2, freq="D")
    idx2 = pd.date_range("2024-01-02", periods=2, freq="D")
    s1 = pd.Series([1, 2], index=idx1, dtype="int64")
    s2 = pd.Series([1.0, 2.0], index=idx2, dtype="float64")
    cu.analyze_series_differences(s1, s2)
    out = capsys.readouterr().out
    assert "Indices are different" in out
    assert "Dtypes are different" in out
    assert "Index frequencies are identical" in out or "Index frequencies are different" in out


def test_format_asset_with_zipline_asset_class_via_stub_module(monkeypatch):
    class Asset:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

    zipline_assets_mod = SimpleNamespace(Asset=Asset)
    zipline_mod = SimpleNamespace(assets=zipline_assets_mod)
    monkeypatch.setitem(sys.modules, "zipline", zipline_mod)
    monkeypatch.setitem(sys.modules, "zipline.assets", zipline_assets_mod)

    assert cu.format_asset(Asset("AAPL")) == "AAPL"


def test_standardize_data_center_and_scale():
    x = np.array([1.0, 2.0, 3.0])
    z = cu.standardize_data(x)
    assert abs(float(np.mean(z))) < 1e-12
    assert abs(float(np.std(z)) - 1.0) < 1e-12


def test_print_table_injects_header_rows_and_calls_display(monkeypatch):
    captured = {}

    def fake_display(obj):
        captured["obj"] = obj

    monkeypatch.setattr(cu, "display", fake_display)
    monkeypatch.setattr(cu, "HTML", lambda s: s)

    df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
    cu.print_table(df, name="T", header_rows={"H": "V"})
    html = captured["obj"]
    assert "<thead>" in html
    assert "H" in html and "V" in html


def test_print_table_run_flask_app_uses_temp_static_dir_without_writing_xlsx(monkeypatch, tmp_path):
    captured = {"excel_path": None}

    def fake_to_excel(self, path, index=True):  # noqa: ARG001
        captured["excel_path"] = str(path)

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)
    monkeypatch.setattr(cu, "__file__", str(tmp_path / "common_utils.py"), raising=False)
    monkeypatch.setattr(cu, "display", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cu, "HTML", lambda s: s)

    df = pd.DataFrame({"a": [1, 2]})
    cu.print_table(df, name="X", run_flask_app=True)
    assert captured["excel_path"] is not None
    assert str(tmp_path / "static") in captured["excel_path"]


def test_print_table_run_flask_app_logs_warning_when_to_excel_fails(monkeypatch, tmp_path):
    def fake_to_excel(self, path, index=True):  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel, raising=True)
    monkeypatch.setattr(cu, "__file__", str(tmp_path / "common_utils.py"), raising=False)
    monkeypatch.setattr(cu, "display", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cu, "HTML", lambda s: s)

    df = pd.DataFrame({"a": [1, 2]})
    cu.print_table(df, name="X", run_flask_app=True)
    # We don't assert on log output here; the goal is to cover the exception
    # handling path without requiring a specific logging backend.


def test_detect_intraday_and_check_intraday_branches(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
    positions = pd.DataFrame({"A": [0, 0, 0], "cash": [100, 100, 100]}, index=idx)
    txn_idx = pd.to_datetime(["2024-01-01 10:00", "2024-01-02 10:00"]).tz_localize("UTC")
    txns = pd.DataFrame({"symbol": ["A", "A"], "amount": [1, -1], "price": [10.0, 10.0]}, index=txn_idx)

    assert bool(cu.detect_intraday(positions, txns)) is True

    monkeypatch.setattr(cu, "detect_intraday", lambda *_a, **_k: True)
    monkeypatch.setattr(cu, "estimate_intraday", lambda *_a, **_k: "EST")
    rets = pd.Series([0.0, 0.0, 0.0], index=idx)
    with pytest.warns(UserWarning, match="Detected intraday strategy"):
        out = cu.check_intraday("infer", rets, positions, txns)
    assert out == "EST"

    with pytest.raises(ValueError, match="Positions and txns needed"):
        cu.check_intraday(True, rets, None, txns)


def test_estimate_intraday_smoke_and_divisor_zero_branch():
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    returns = pd.Series([-1.0, 0.0], index=idx)
    positions = pd.DataFrame({"A": [50.0, 0.0], "cash": [50.0, 100.0]}, index=idx)

    txn_idx = pd.to_datetime(
        ["2024-01-01 09:30", "2024-01-01 15:30", "2024-01-02 10:00"],
        utc=True,
    )
    transactions = pd.DataFrame(
        {"symbol": ["A", "A", "A"], "amount": [1, -1, 1], "price": [10.0, 11.0, 12.0]},
        index=txn_idx,
    )

    corrected = cu.estimate_intraday(returns, positions, transactions)
    assert isinstance(corrected, pd.DataFrame)
    assert "cash" in corrected.columns
    assert corrected.index.name == "period_close"


def test_extract_rets_pos_txn_from_zipline_smoke(monkeypatch):
    # Patch Empyrical helpers so we can run without zipline.
    from fincore.empyrical import Empyrical

    def fake_extract_pos(positions_df, ending_cash):
        # Minimal extract: pivot values by sid and join cash.
        df = positions_df.copy()
        df["values"] = df["amount"] * df["last_sale_price"]
        out = df.reset_index().pivot_table(index="index", columns="sid", values="values").fillna(0)
        cash = ending_cash.copy()
        cash.name = "cash"
        out = out.join(cash).fillna(0)
        out.columns.name = "sid"
        return out

    def fake_make_transaction_frame(txn):
        return pd.DataFrame(
            {"amount": [1], "price": [10.0], "symbol": ["A"]},
            index=pd.to_datetime(["2024-01-02 10:00"]),
        )

    monkeypatch.setattr(Empyrical, "extract_pos", staticmethod(fake_extract_pos))
    monkeypatch.setattr(Empyrical, "make_transaction_frame", staticmethod(fake_make_transaction_frame))

    class Backtest:
        def __init__(self) -> None:
            self.index = pd.date_range("2024-01-01", periods=2, freq="B")
            self.returns = pd.Series([0.0, 0.01], index=self.index)
            self.ending_cash = pd.Series([100.0, 101.0], index=self.index)
            self.positions = {
                self.index[0]: [{"sid": "A", "amount": 1.0, "last_sale_price": 10.0}],
                self.index[1]: [{"sid": "A", "amount": 2.0, "last_sale_price": 11.0}],
            }
            self.transactions = []

    bt = Backtest()
    rets, pos, txn = cu.extract_rets_pos_txn_from_zipline(bt)
    assert isinstance(rets, pd.Series)
    assert isinstance(pos, pd.DataFrame)
    assert isinstance(txn, pd.DataFrame)
    assert str(txn.index.tz) in {"UTC", "utc"}


def test_extract_rets_pos_txn_from_zipline_raises_when_no_positions():
    class Backtest:
        def __init__(self) -> None:
            self.index = pd.date_range("2024-01-01", periods=2, freq="B")
            self.returns = pd.Series([0.0, 0.01], index=self.index)
            self.ending_cash = pd.Series([100.0, 101.0], index=self.index)
            self.positions = {}
            self.transactions = []

    with pytest.raises(ValueError, match="does not have any positions"):
        cu.extract_rets_pos_txn_from_zipline(Backtest())


def test_rolling_window_fallback_when_writeable_kwarg_not_supported(monkeypatch):
    import numpy as _np
    from numpy.lib.stride_tricks import as_strided as _real_as_strided

    def _as_strided_no_writeable(array, shape, strides, writeable=None):  # noqa: ARG001
        raise TypeError("no writeable")

    a = _np.arange(9).reshape(3, 3)
    monkeypatch.setattr(cu, "as_strided", _as_strided_no_writeable)
    # Use the real as_strided for the fallback call inside the TypeError branch.
    monkeypatch.setattr(cu, "as_strided", _as_strided_no_writeable)
    monkeypatch.setattr(
        cu,
        "as_strided",
        lambda array, shape, strides, writeable=None: (
            _real_as_strided(array, shape, strides)
            if writeable is None
            else (_as_strided_no_writeable(array, shape, strides, writeable))
        ),  # noqa: E501
    )

    with pytest.raises(ValueError, match="Cannot create a writable rolling window view"):
        cu.rolling_window(a, 2, mutable=True)
