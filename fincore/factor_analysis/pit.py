"""Point-in-time (PIT) factor data contracts.

Enhanced factor inputs must declare ``as_of``, ``known_at`` and
``effective_from`` so look-ahead bias can be detected: a factor value observed
at ``as_of`` may only be used from ``known_at`` onward, and a perturbation of
data after ``as_of`` must never change results computed at or before ``as_of``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["PITPoint", "validate_pit_alignment"]


@dataclass(frozen=True)
class PITPoint:
    """One point-in-time factor observation."""

    as_of: pd.Timestamp
    known_at: pd.Timestamp
    effective_from: pd.Timestamp
    value: float

    def __post_init__(self) -> None:
        if self.known_at < self.as_of:
            raise ValueError(f"look-ahead bias: known_at {self.known_at} precedes as_of {self.as_of}")


def validate_pit_alignment(points: list[PITPoint]) -> None:
    """Raise if any PIT point is known before it is observed (look-ahead)."""
    for point in points:
        if point.known_at < point.as_of:
            raise ValueError(f"look-ahead bias: {point.known_at} < {point.as_of}")
