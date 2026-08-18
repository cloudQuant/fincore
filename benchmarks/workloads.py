"""Deterministic cross-domain benchmark workloads.

Each factory builds a fixed-size, fixed-seed input and records its expected row
count and a SHA256 input digest, so a profile run can prove it measured the same
workload across commits and platforms.  Sizes are ``small``, ``medium`` and
``large``; digests depend only on the seed and size.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SIZES: dict[str, dict[str, int]] = {
    "small": {"dates": 252, "assets": 100},
    "medium": {"dates": 1_260, "assets": 500},
    "large": {"dates": 2_520, "assets": 1_000},
}


def _sha256_payload(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
        input_digest=_sha256_payload(f"factor_panel:{size}:{seed}:{expected_rows}"),
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
        input_digest=_sha256_payload(f"single_series:{size}:{seed}:{shape['dates']}"),
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
        input_digest=_sha256_payload(f"rolling_returns:{size}:{seed}:{shape['dates']}"),
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
        input_digest=_sha256_payload(f"transactions:{size}:{seed}:{rows}"),
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
        input_digest=_sha256_payload(f"report:{size}:{seed}:{shape['dates']}"),
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
]
