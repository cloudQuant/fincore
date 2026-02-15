import numpy as np
import pandas as pd

from fincore.report import format as report_format
from fincore.report.format import css_cls, fmt, html_df, html_table, safe_list


def test_fmt_basic() -> None:
    assert fmt(10) == "10"
    assert fmt(10000) == "10,000"
    assert fmt(1.234567) == "1.2346"
    assert fmt(1.234567, pct=True) == "123.46%"
    assert fmt(2_500_000.12) == "2,500,000"
    assert fmt(np.nan) == "N/A"


def test_css_cls_sign() -> None:
    assert css_cls(1.0) == "pos"
    assert css_cls(-1.0) == "neg"
    assert css_cls(0.0) == ""
    assert css_cls(np.nan) == ""
    assert css_cls("x") == ""


def test_css_cls_handles_isnan_typeerror(monkeypatch) -> None:
    # Cover the defensive exception path around np.isnan().
    def _boom(_v):
        raise TypeError("boom")

    monkeypatch.setattr(report_format.np, "isnan", _boom)
    assert css_cls(1.0) == "pos"


def test_safe_list_handles_nan_and_inf() -> None:
    arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.5])
    assert safe_list(arr) == [1.0, None, None, None, 2.5]
    assert safe_list(arr, pct=True, decimals=2) == [100.0, None, None, None, 250.0]


def test_html_table_renders_rows() -> None:
    d = {"A": 0.1, "B": -0.2}
    html = html_table(d, pct_keys={"A", "B"})
    assert "<table>" in html
    assert "A" in html and "B" in html
    assert "%" in html


def test_safe_list_accepts_pandas_series() -> None:
    s = pd.Series([0.0, 0.1, -0.2])
    assert safe_list(s, decimals=4, pct=True) == [0.0, 10.0, -20.0]


def test_html_df_renders_nan_as_empty_cell() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan]})
    html = html_df(df, float_format=".2f")
    assert "<td></td>" in html
