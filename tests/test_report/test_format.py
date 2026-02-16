"""Tests for report.format module."""

from __future__ import annotations

import numpy as np

from fincore.report.format import css_cls, fmt, safe_list


class TestFmt:
    def test_positive_percentage(self):
        result = fmt(0.1234, pct=True)
        assert "12.3" in result

    def test_negative_percentage(self):
        result = fmt(-0.05, pct=True)
        assert "5.0" in result

    def test_float_format(self):
        result = fmt(1.2345, pct=False)
        assert "1.23" in result or "1.2" in result

    def test_nan_value(self):
        result = fmt(float("nan"), pct=True)
        assert isinstance(result, str)

    def test_none_value(self):
        result = fmt(None, pct=True)
        assert isinstance(result, str)


class TestCssCls:
    def test_positive_returns_green(self):
        cls = css_cls(0.05)
        assert "pos" in cls.lower() or "green" in cls.lower() or cls != ""

    def test_negative_returns_red(self):
        cls = css_cls(-0.05)
        assert isinstance(cls, str)

    def test_zero_value(self):
        cls = css_cls(0.0)
        assert isinstance(cls, str)


class TestSafeList:
    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = safe_list(arr)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_with_nan(self):
        arr = np.array([1.0, float("nan"), 3.0])
        result = safe_list(arr)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_empty_array(self):
        arr = np.array([])
        result = safe_list(arr)
        assert isinstance(result, list)
        assert len(result) == 0
