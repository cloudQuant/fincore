"""Tests for decorator utilities in fincore.utils.common_utils."""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

from fincore.utils import common_utils as cu


def test_customize_decorator_set_context_false_calls_function_directly():
    """Test @customize decorator with set_context=False calls function directly."""
    calls = []

    @cu.customize
    def f(x):
        calls.append(x)
        return x + 1

    assert f(1, set_context=False) == 2
    assert calls == [1]


def test_customize_decorator_fallback_when_no_plotting_helpers_present():
    """Test @customize decorator fallback when no plotting helpers present."""
    @cu.customize
    def f(x):
        return x + 1

    assert f(1) == 2


def test_customize_decorator_uses_plotting_context_and_axes_style():
    """Test @customize decorator uses plotting_context and axes_style when available."""
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


def test_default_returns_func_is_passthrough():
    """Test default_returns_func returns input unchanged."""
    rets = pd.Series([1.0, 2.0])
    assert cu.default_returns_func(rets) is rets


def test_default_returns_func_imports_empyrical_and_returns_float():
    """Test _default_returns_func returns float from Series."""
    rets = pd.Series([0.01, -0.005, 0.002], index=pd.date_range("2020-01-01", periods=3))
    out = cu._default_returns_func(rets)
    assert isinstance(out, float)


def test_register_return_func_and_get_symbol_rets_roundtrip():
    """Test register_return_func and get_symbol_rets roundtrip."""
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
