"""Canonical frequency and annualization semantics for metric kernels."""

from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType

import pandas as pd
from packaging.version import Version

DAILY = "daily"
WEEKLY = "weekly"
MONTHLY = "monthly"
QUARTERLY = "quarterly"
YEARLY = "yearly"

APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
QTRS_PER_YEAR = 4

ANNUALIZATION_FACTORS = MappingProxyType(
    {
        DAILY: APPROX_BDAYS_PER_YEAR,
        WEEKLY: WEEKS_PER_YEAR,
        MONTHLY: MONTHS_PER_YEAR,
        QUARTERLY: QTRS_PER_YEAR,
        YEARLY: 1,
    }
)
_PANDAS_FREQUENCIES = MappingProxyType(
    {
        DAILY: "D",
        WEEKLY: "W",
        MONTHLY: "ME" if Version(pd.__version__) >= Version("2.2.0") else "M",
        QUARTERLY: "QE" if Version(pd.__version__) >= Version("2.2.0") else "Q",
        YEARLY: "YE" if Version(pd.__version__) >= Version("2.2.0") else "A",
    }
)


@lru_cache(maxsize=32)
def annualization_factor(period: str, annualization: float | None = None) -> float:
    """Return an explicit annualization override or the canonical frequency factor."""
    if annualization is not None:
        return float(annualization)
    try:
        return float(ANNUALIZATION_FACTORS[period])
    except KeyError as error:
        raise ValueError(
            f"Period cannot be {period!r}: unknown frequency; expected one of {tuple(ANNUALIZATION_FACTORS)!r}"
        ) from error


def pandas_frequency(period: str) -> str:
    """Return the pandas resampling alias for one canonical frequency name."""
    try:
        return _PANDAS_FREQUENCIES[period]
    except KeyError as error:
        raise ValueError(
            f"Period cannot be {period!r}: unknown frequency; expected one of {tuple(_PANDAS_FREQUENCIES)!r}"
        ) from error
