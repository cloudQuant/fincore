"""Typed inputs and result models owned by the portfolio domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from fincore.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "ExposureBundle",
    "PortfolioInputs",
    "VolumeExposureBundle",
]


def _validate_frame(name: str, frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValidationError(
            "exposure component must be a pandas DataFrame",
            param_name=name,
            value=type(frame).__name__,
        )
    if not frame.columns.is_unique:
        raise ValidationError(
            "exposure category columns must be unique",
            param_name=name,
            value=list(frame.columns),
        )


def _validate_series(name: str, series: pd.Series) -> None:
    if not isinstance(series, pd.Series):
        raise ValidationError(
            "volume exposure component must be a pandas Series",
            param_name=name,
            value=type(series).__name__,
        )


@dataclass(frozen=True, slots=True)
class PortfolioInputs:
    """Detached, optional input tables for a portfolio domain workflow.

    The model intentionally carries only concrete data, not methods, caches,
    renderers, or legacy façade state.  Inputs are copied at ingestion and
    copied again when materialized so callers cannot mutate a queued workflow.
    """

    returns: pd.Series | None = None
    positions: pd.DataFrame | None = None
    transactions: pd.DataFrame | None = None
    benchmark_returns: pd.Series | None = None

    def __post_init__(self) -> None:
        for name in ("returns", "benchmark_returns"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, pd.Series):
                raise TypeError(f"{name} must be a pandas Series or None")
            if value is not None:
                object.__setattr__(self, name, value.copy(deep=True))
        for name in ("positions", "transactions"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame or None")
            if value is not None:
                object.__setattr__(self, name, value.copy(deep=True))

    def materialize(self) -> Mapping[str, pd.Series | pd.DataFrame]:
        """Return a fresh mapping of every supplied input table."""
        return {
            name: value.copy(deep=True)
            for name in ("returns", "positions", "transactions", "benchmark_returns")
            if (value := getattr(self, name)) is not None
        }


@dataclass(frozen=True, slots=True)
class ExposureBundle:
    """Named long, short, gross, and net exposure tables."""

    long: pd.DataFrame
    short: pd.DataFrame
    gross: pd.DataFrame
    net: pd.DataFrame

    def __post_init__(self) -> None:
        components = {"long": self.long, "short": self.short, "gross": self.gross, "net": self.net}
        for name, frame in components.items():
            _validate_frame(name, frame)

        reference_index = self.long.index
        reference_columns = self.long.columns
        for name, frame in tuple(components.items())[1:]:
            if not frame.index.equals(reference_index):
                raise ValidationError("exposure components must use the same index", param_name=name)
            if not frame.columns.equals(reference_columns):
                raise ValidationError("exposure components must use the same category columns", param_name=name)


@dataclass(frozen=True, slots=True)
class VolumeExposureBundle:
    """Named long, short, and gross volume-exposure series."""

    long: pd.Series
    short: pd.Series
    gross: pd.Series

    def __post_init__(self) -> None:
        components = {"long": self.long, "short": self.short, "gross": self.gross}
        for name, series in components.items():
            _validate_series(name, series)
        reference_index = self.long.index
        for name, series in tuple(components.items())[1:]:
            if not series.index.equals(reference_index):
                raise ValidationError("volume exposure components must use the same index", param_name=name)
