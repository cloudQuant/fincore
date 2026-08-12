"""Typed portfolio exposure results and compatibility projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from fincore.exceptions import ValidationError

ExposureTuple = tuple[
    list[pd.Series],
    list[pd.Series],
    list[pd.Series],
    list[pd.Series],
]
VolumeExposureTuple = tuple[pd.Series, pd.Series, pd.Series]

__all__ = [
    "ExposureBundle",
    "ExposureTuple",
    "VolumeExposureBundle",
    "VolumeExposureTuple",
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


@dataclass(frozen=True)
class ExposureBundle:
    """Named long, short, gross, and net exposure tables."""

    long: pd.DataFrame
    short: pd.DataFrame
    gross: pd.DataFrame
    net: pd.DataFrame

    def __post_init__(self) -> None:
        components = {
            "long": self.long,
            "short": self.short,
            "gross": self.gross,
            "net": self.net,
        }
        for name, frame in components.items():
            _validate_frame(name, frame)

        reference_index = self.long.index
        reference_columns = self.long.columns
        for name, frame in tuple(components.items())[1:]:
            if not frame.index.equals(reference_index):
                raise ValidationError(
                    "exposure components must use the same index",
                    param_name=name,
                )
            if not frame.columns.equals(reference_columns):
                raise ValidationError(
                    "exposure components must use the same category columns",
                    param_name=name,
                )

    def as_legacy_tuple(self, category_order: Iterable[str]) -> ExposureTuple:
        """Project named frames to pyfolio's ordered 4-tuple of Series lists."""

        order = list(category_order)
        if len(order) != len(set(order)):
            raise ValidationError(
                "legacy category order contains duplicate columns",
                param_name="category_order",
                value=order,
            )
        actual = list(self.long.columns)
        missing = [category for category in order if category not in self.long.columns]
        unexpected = [category for category in actual if category not in order]
        if missing or unexpected:
            raise ValidationError(
                f"legacy exposure projection has missing={missing!r}, unexpected={unexpected!r}",
                param_name="category_order",
                value=actual,
            )

        return tuple([frame[category] for category in order] for frame in (self.long, self.short, self.gross, self.net))  # type: ignore[return-value]


@dataclass(frozen=True)
class VolumeExposureBundle:
    """Named long, short, and gross volume exposure series."""

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
                raise ValidationError(
                    "volume exposure components must use the same index",
                    param_name=name,
                )

    def as_legacy_tuple(self) -> VolumeExposureTuple:
        """Project the named result to pyfolio's ordered 3-tuple."""

        return self.long, self.short, self.gross
