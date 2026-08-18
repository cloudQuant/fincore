"""Deterministic workload factory tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from workloads import (
    factor_panel_workload,
    report_workload,
    rolling_returns_workload,
    single_series_workload,
    transactions_workload,
)


def test_factor_workload_has_fixed_shape_seed_and_output_digest() -> None:
    case = factor_panel_workload("medium", seed=20260817)

    assert case.factor.index.nlevels == 2
    assert case.expected_rows == 630000
    assert len(case.input_digest) == 64


def test_factor_workload_is_deterministic_for_a_fixed_seed() -> None:
    a = factor_panel_workload("small", seed=42)
    b = factor_panel_workload("small", seed=42)
    c = factor_panel_workload("small", seed=43)

    assert a.input_digest == b.input_digest
    assert a.input_digest != c.input_digest
    assert a.factor.equals(b.factor)


def test_workload_sizes_cover_three_scales() -> None:
    rows = {size: factor_panel_workload(size).expected_rows for size in ("small", "medium", "large")}

    assert rows["small"] < rows["medium"] < rows["large"]
    assert rows["medium"] == 630000


def test_single_series_and_rolling_workloads_have_bounded_rows() -> None:
    single = single_series_workload("medium")
    rolling = rolling_returns_workload("medium")

    assert len(single.returns) == 1260
    assert len(rolling.returns) == 1260
    assert len(single.input_digest) == 64


def test_transactions_workload_builds_fifo_inputs() -> None:
    case = transactions_workload("small")

    assert case.transactions is not None
    assert {"symbol", "amount", "price"} <= set(case.transactions.columns)
    assert case.expected_rows == len(case.transactions)


def test_report_workload_builds_a_returns_series() -> None:
    case = report_workload("small")

    assert case.returns is not None
    assert len(case.returns) == 252


def test_unknown_size_raises() -> None:
    with pytest.raises(ValueError):
        factor_panel_workload("bogus")
