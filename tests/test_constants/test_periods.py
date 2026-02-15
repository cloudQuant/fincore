"""Tests for fincore.constants.periods."""

from __future__ import annotations

import importlib

import pandas as pd


def test_period_to_freq_legacy_pandas_branch_via_reload(monkeypatch):
    import fincore.constants.periods as periods

    original = pd.__version__
    try:
        monkeypatch.setattr(pd, "__version__", "2.1.9", raising=False)
        periods2 = importlib.reload(periods)
        assert periods2.PERIOD_TO_FREQ["monthly"] == "M"
        assert periods2.PERIOD_TO_FREQ["quarterly"] == "Q"
        assert periods2.PERIOD_TO_FREQ["yearly"] == "A"
    finally:
        monkeypatch.setattr(pd, "__version__", original, raising=False)
        importlib.reload(periods)
