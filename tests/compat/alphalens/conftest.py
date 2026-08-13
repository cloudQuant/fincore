"""Shared Alphalens compatibility fixtures and pinned-manifest helpers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_PATH = _REPOSITORY_ROOT / "tests/compat/fixtures/alphalens-0.4.0-cloudquant-api.json"
_ASSETS = tuple(f"asset_{ordinal:02d}" for ordinal in range(10))


@lru_cache(maxsize=1)
def load_pinned_manifest() -> dict[str, Any]:
    """Load the development-only pinned API manifest for compatibility assertions."""

    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def manifest_entries() -> tuple[dict[str, Any], ...]:
    """Return the pinned C0 definition entries in stable manifest order."""

    return tuple(load_pinned_manifest()["entries"])


def callable_entries_with_signature() -> tuple[dict[str, Any], ...]:
    """Return entries whose manifest freezes a source or introspection signature."""

    return tuple(
        entry for entry in manifest_entries() if entry["kind"] == "function" or entry["symbol"] == "GridFigure"
    )


def accepted_call_cases() -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
    """Return every manifest-declared accepted call grammar row."""

    return tuple((entry, case) for entry in manifest_entries() for case in entry["accepted_call_cases"])


@lru_cache(maxsize=1)
def _shared_inputs() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build the immutable-source synthetic data contract shared by later tasks."""

    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=120)
    factor_index = pd.MultiIndex.from_product((dates, _ASSETS), names=("date", "asset"))
    raw_factor = pd.Series(rng.normal(0, 1, size=len(factor_index)), index=factor_index, name="factor")

    price_assets = (*_ASSETS, "asset_10", "asset_11")
    price_changes = rng.normal(0, 0.01, size=(len(dates), len(price_assets))).cumsum(axis=0)
    prices = pd.DataFrame(100 + price_changes, index=dates, columns=price_assets)
    prices.index.name = "date"
    tz_aware_prices = prices.copy()
    tz_aware_prices.index = tz_aware_prices.index.tz_localize("UTC")
    groups = pd.Series(
        ["sector_a" if ordinal % 2 == 0 else "sector_b" for ordinal in range(len(_ASSETS))],
        index=pd.Index(_ASSETS, name="asset"),
        name="group",
    )
    return raw_factor, prices, tz_aware_prices, groups


@pytest.fixture
def raw_factor() -> pd.Series:
    """A fresh factor series; callers may safely make local mutations."""

    return _shared_inputs()[0].copy()


@pytest.fixture
def prices() -> pd.DataFrame:
    """A fresh naive price frame with two non-factor assets."""

    return _shared_inputs()[1].copy()


@pytest.fixture
def tz_aware_prices() -> pd.DataFrame:
    """A fresh UTC version of :func:`prices`."""

    return _shared_inputs()[2].copy()


@pytest.fixture
def groups() -> pd.Series:
    """A fresh alternating-sector mapping for the ten factor assets."""

    return _shared_inputs()[3].copy()


@pytest.fixture(scope="session")
def clean_factor_data() -> pd.DataFrame:
    """Reserve the real cleaned-data fixture for Task 3 rather than fabricate it."""

    raise RuntimeError(
        "clean_factor_data is deferred until Task 3 provides prepare_factor_data; "
        "Task 2 deliberately does not fabricate cleaned factor output."
    )
