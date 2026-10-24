"""Regression tests for fingerprint streaming boundaries."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pandas as pd

from fincore.factor_analysis import models


@dataclass
class _DataFrameAlsoMapping(Mapping[str, object]):
    """Exercise dataclass dispatch before the generic Mapping fallback."""

    frame: pd.DataFrame

    def __getitem__(self, key: str) -> object:
        raise AssertionError(f"dataclass fingerprinting must not read mapping key {key!r}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("dataclass fingerprinting must not iterate the mapping")

    def __len__(self) -> int:
        raise AssertionError("dataclass fingerprinting must not inspect mapping length")


class _CaptureWriter:
    def __init__(self) -> None:
        self.parts: list[bytes] = []

    def write(self, value: bytes | bytearray | memoryview) -> None:
        self.parts.append(bytes(value))

    def getvalue(self) -> bytes:
        return b"".join(self.parts)


def test_dataclass_mapping_with_nested_dataframe_streams_as_dataclass() -> None:
    frame = pd.DataFrame(
        {"signal": np.array([1.25, -0.0, np.nan], dtype=np.float64)},
        index=pd.date_range("2025-01-01", periods=3, name="date"),
    )
    frame.attrs = {"metadata": ("nested", 2)}
    value = _DataFrameAlsoMapping(frame=frame)

    expected = models._fingerprint_bytes(value)
    writer = _CaptureWriter()
    encoded_length = models._stream_fingerprint(value, writer)
    actual = writer.getvalue()

    assert encoded_length == len(expected)
    assert actual == expected
    assert models.fingerprint_value(value) == sha256(expected).hexdigest()