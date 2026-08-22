"""Deterministic cross-domain benchmark workloads.

Each factory builds a fixed-size, fixed-seed input and records its expected row
count and a SHA256 input digest, so a profile run can prove it measured the same
workload across commits and platforms.  Sizes are ``small``, ``medium`` and
``large``; digests depend only on the seed and size.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SIZES: dict[str, dict[str, int]] = {
    "small": {"dates": 252, "assets": 100},
    "medium": {"dates": 1_260, "assets": 500},
    "large": {"dates": 2_520, "assets": 1_000},
}


def _pandas_value_digest(value: pd.Series | pd.DataFrame) -> str:
    """Digest values *and* label semantics for a benchmark input.

    A benchmark with the same row count but different returns, labels, dtypes,
    or business calendar is a different financial workload.  ``hash_pandas_object``
    covers indexed values while the explicit JSON header preserves column and
    dtype order, index names, and calendar metadata that row hashes alone do
    not make obvious in an artifact review.
    """
    if isinstance(value, pd.Series):
        header: dict[str, Any] = {
            "kind": "series",
            "name": str(value.name),
            "dtype": str(value.dtype),
        }
    else:
        header = {
            "kind": "dataframe",
            "columns": [str(column) for column in value.columns],
            "dtypes": [str(dtype) for dtype in value.dtypes],
        }

    index = value.index
    header["shape"] = list(value.shape)
    header["index_type"] = type(index).__name__
    header["index_names"] = [str(name) for name in index.names]
    header["index_dtypes"] = (
        [str(level.dtype) for level in index.levels] if isinstance(index, pd.MultiIndex) else [str(index.dtype)]
    )
    frequency = getattr(index, "freqstr", None)
    if frequency is not None:
        header["calendar_frequency"] = frequency

    hasher = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    hashes = pd.util.hash_pandas_object(value, index=True, categorize=True).to_numpy(dtype=np.uint64, copy=False)
    hasher.update(hashes.tobytes())
    return hasher.hexdigest()


def workload_input_digest(
    name: str,
    size: str,
    seed: int,
    *,
    factor: pd.DataFrame | None = None,
    returns: pd.Series | None = None,
    transactions: pd.DataFrame | None = None,
) -> str:
    """Return the semantic digest of every concrete input supplied to a workload."""
    hasher = hashlib.sha256(
        json.dumps({"name": name, "size": size, "seed": seed}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for field_name, value in (
        ("factor", factor),
        ("returns", returns),
        ("transactions", transactions),
    ):
        if value is not None:
            hasher.update(field_name.encode("utf-8"))
            hasher.update(_pandas_value_digest(value).encode("ascii"))
    return hasher.hexdigest()


@dataclass(frozen=True)
class Workload:
    """A deterministic input workload with provenance."""

    name: str
    size: str
    seed: int
    expected_rows: int
    input_digest: str
    factor: pd.DataFrame | None = None
    returns: pd.Series | None = None
    transactions: pd.DataFrame | None = None


def _factor_frame(dates: int, assets: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    date_index = pd.bdate_range("2010-01-04", periods=dates, name="date")
    asset_index = pd.Index([f"A{number:04d}" for number in range(assets)], name="asset")
    index = pd.MultiIndex.from_product((date_index, asset_index), names=("date", "asset"))
    return pd.DataFrame({"factor": rng.standard_normal(len(index))}, index=index)


def factor_panel_workload(size: str = "medium", seed: int = 20260817) -> Workload:
    """A factor panel workload (factor values over a date/asset MultiIndex)."""
    if size not in SIZES:
        raise ValueError(f"size must be one of {tuple(SIZES)}")
    shape = SIZES[size]
    factor = _factor_frame(shape["dates"], shape["assets"], seed)
    expected_rows = shape["dates"] * shape["assets"]
    return Workload(
        name="factor_panel",
        size=size,
        seed=seed,
        expected_rows=expected_rows,
        input_digest=workload_input_digest("factor_panel", size, seed, factor=factor),
        factor=factor,
    )


def single_series_workload(size: str = "medium", seed: int = 20260817) -> Workload:
    """A single return-series workload for scalar metrics."""
    if size not in SIZES:
        raise ValueError(f"size must be one of {tuple(SIZES)}")
    shape = SIZES[size]
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2010-01-04", periods=shape["dates"], name="date")
    returns = pd.Series(rng.normal(0.0, 0.02, shape["dates"]), index=index, name="returns")
    return Workload(
        name="single_series",
        size=size,
        seed=seed,
        expected_rows=shape["dates"],
        input_digest=workload_input_digest("single_series", size, seed, returns=returns),
        returns=returns,
    )


def rolling_returns_workload(size: str = "medium", seed: int = 20260817) -> Workload:
    """A return-series workload for rolling metrics."""
    if size not in SIZES:
        raise ValueError(f"size must be one of {tuple(SIZES)}")
    shape = SIZES[size]
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2010-01-04", periods=shape["dates"], name="date")
    returns = pd.Series(rng.normal(0.0, 0.02, shape["dates"]), index=index, name="returns")
    return Workload(
        name="rolling_returns",
        size=size,
        seed=seed,
        expected_rows=shape["dates"],
        input_digest=workload_input_digest("rolling_returns", size, seed, returns=returns),
        returns=returns,
    )


def transactions_workload(size: str = "medium", seed: int = 20260817) -> Workload:
    """A FIFO transaction workload for round-trip extraction."""
    if size not in SIZES:
        raise ValueError(f"size must be one of {tuple(SIZES)}")
    row_counts = {"small": 500, "medium": 5_000, "large": 50_000}
    rows = row_counts[size]
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-02", periods=rows, name="date")
    transactions = pd.DataFrame(
        {
            "symbol": np.tile(["AAPL", "MSFT", "GOOG", "AMZN"], int(np.ceil(rows / 4)))[:rows],
            "amount": rng.integers(-1000, 1001, rows),
            "price": rng.uniform(50.0, 500.0, rows),
        },
        index=index,
    )
    return Workload(
        name="transactions",
        size=size,
        seed=seed,
        expected_rows=rows,
        input_digest=workload_input_digest("transactions", size, seed, transactions=transactions),
        transactions=transactions,
    )


def report_workload(size: str = "medium", seed: int = 20260817) -> Workload:
    """A return-series workload for report model computation."""
    if size not in SIZES:
        raise ValueError(f"size must be one of {tuple(SIZES)}")
    shape = SIZES[size]
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2010-01-04", periods=shape["dates"], name="date")
    returns = pd.Series(rng.normal(0.0, 0.02, shape["dates"]), index=index, name="returns")
    return Workload(
        name="report",
        size=size,
        seed=seed,
        expected_rows=shape["dates"],
        input_digest=workload_input_digest("report", size, seed, returns=returns),
        returns=returns,
    )


def describe_workload(workload: Workload) -> dict[str, Any]:
    """Return a JSON-compatible description of a workload."""
    return {
        "name": workload.name,
        "size": workload.size,
        "seed": workload.seed,
        "expected_rows": workload.expected_rows,
        "input_digest": workload.input_digest,
    }


__all__ = [
    "SIZES",
    "Workload",
    "describe_workload",
    "factor_panel_workload",
    "report_workload",
    "rolling_returns_workload",
    "single_series_workload",
    "transactions_workload",
    "workload_input_digest",
]
