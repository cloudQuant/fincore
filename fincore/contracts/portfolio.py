"""Temporary migration bridge for portfolio result models.

Canonical ownership moved to :mod:`fincore.portfolio.models`.  This module is
kept only while the pre-0.5 contract package remains in the source tree and
must not be used by new domain code.
"""

from __future__ import annotations

from typing import TypeAlias

import pandas as pd

from fincore.portfolio.models import ExposureBundle, VolumeExposureBundle

ExposureTuple: TypeAlias = tuple[
    list[pd.Series],
    list[pd.Series],
    list[pd.Series],
    list[pd.Series],
]
VolumeExposureTuple: TypeAlias = tuple[pd.Series, pd.Series, pd.Series]

__all__ = [
    "ExposureBundle",
    "ExposureTuple",
    "VolumeExposureBundle",
    "VolumeExposureTuple",
]
