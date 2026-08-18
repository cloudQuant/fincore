"""Content-addressed data snapshots for reproducible external-data analysis.

A :class:`DataSnapshot` freezes the identity of a fetched frame (source, request
interval, as-of timestamp, price-adjustment convention, timezone) together with
a defensive copy of the data and a stable SHA256 of that data.  The manifest it
produces never contains secret configuration such as API keys or tokens.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

SCHEMA_VERSION = 1


def _sha256_dataframe(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=True, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DataSnapshot:
    """An immutable, content-addressed snapshot of fetched market data."""

    provider: str
    requested_start: str
    requested_end: str
    as_of: str
    schema_version: int = SCHEMA_VERSION
    timezone: str = "UTC"
    price_adjustment: str = "raw"
    content_sha256: str = ""
    _data: pd.DataFrame = field(repr=False, compare=False, default_factory=pd.DataFrame)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        provider: str,
        requested_start: str,
        requested_end: str,
        as_of: str,
        timezone: str = "UTC",
        price_adjustment: str = "raw",
    ) -> DataSnapshot:
        """Build a snapshot from a frame, hashing its deterministic contents."""
        data = frame.copy(deep=True)
        return cls(
            provider=provider,
            requested_start=requested_start,
            requested_end=requested_end,
            as_of=as_of,
            timezone=timezone,
            price_adjustment=price_adjustment,
            content_sha256=_sha256_dataframe(data),
            _data=data,
        )

    @property
    def data(self) -> pd.DataFrame:
        """A defensive copy of the snapshot's frame."""
        return self._data.copy(deep=True)

    def identity_kwargs(self) -> dict[str, Any]:
        """Return the kwargs needed to rebuild an identical snapshot from a frame."""
        return {
            "provider": self.provider,
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "as_of": self.as_of,
            "timezone": self.timezone,
            "price_adjustment": self.price_adjustment,
        }

    def to_manifest(self) -> dict[str, Any]:
        """Return a provenance manifest that never carries secret configuration."""
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "as_of": self.as_of,
            "timezone": self.timezone,
            "price_adjustment": self.price_adjustment,
            "content_sha256": self.content_sha256,
        }


__all__ = ["SCHEMA_VERSION", "DataSnapshot"]
