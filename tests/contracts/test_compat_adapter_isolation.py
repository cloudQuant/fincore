"""Strict adapter isolation tests: strict calls must not enter enhanced state."""

from __future__ import annotations

import sys

import pandas as pd

from fincore import empyrical


def test_strict_empyrical_call_does_not_construct_enhanced_stateful_class() -> None:
    constructed: list[str] = []

    def _trace(frame: object, event: str, arg: object) -> object:
        if event == "call":
            name = frame.f_code.co_name  # type: ignore[attr-defined]
            if name in {"__init__", "analyze"}:
                mod = frame.f_globals.get("__name__", "")  # type: ignore[attr-defined]
                if mod.startswith("fincore.core") or mod == "fincore.core.context":
                    constructed.append(f"{mod}.{name}")
        return _trace

    returns = pd.Series([0.01, -0.005, 0.002, 0.004])
    sys.settrace(_trace)
    try:
        empyrical.sharpe_ratio(returns)
    finally:
        sys.settrace(None)

    assert constructed == [], f"strict call constructed enhanced state: {constructed}"


def test_strict_empyrical_does_not_import_context() -> None:
    returns = pd.Series([0.01, -0.005, 0.002])
    empyrical.sharpe_ratio(returns)
    assert "fincore.core.context" not in sys.modules


def test_strict_surface_returns_scalar() -> None:
    returns = pd.Series([0.01, -0.005, 0.002, 0.004])
    value = empyrical.sharpe_ratio(returns)
    assert isinstance(value, float)
