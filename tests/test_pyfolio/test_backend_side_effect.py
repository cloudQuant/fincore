"""Task 8 side-effect regression: importing pyfolio must not switch backends.

The library must never call ``matplotlib.use()`` (or any other import-time
backend selection) on behalf of the caller.  Backend selection belongs to
applications, CLIs, and tests.
"""

from __future__ import annotations

import importlib

import matplotlib
import pandas as pd


def test_import_pyfolio_preserves_selected_backend() -> None:
    matplotlib.use("svg", force=True)
    before = matplotlib.get_backend()
    try:
        importlib.import_module("fincore.pyfolio")
        assert matplotlib.get_backend() == before
    finally:
        matplotlib.use(before)


def test_import_pyfolio_class_and_impl_preserve_selected_backend() -> None:
    matplotlib.use("svg", force=True)
    before = matplotlib.get_backend()
    try:
        importlib.import_module("fincore.pyfolio")
        # Attribute access is the lazy trigger that pulls in the heavy
        # _pyfolio_impl module (the old home of ``matplotlib.use('Agg')``).
        from fincore.pyfolio import Pyfolio

        assert Pyfolio is not None
        assert matplotlib.get_backend() == before
    finally:
        matplotlib.use(before)


def test_import_tearsheets_preserve_selected_backend() -> None:
    matplotlib.use("svg", force=True)
    before = matplotlib.get_backend()
    try:
        importlib.import_module("fincore.tearsheets.sheets")
        assert matplotlib.get_backend() == before
    finally:
        matplotlib.use(before)


def test_enhanced_full_tear_sheet_return_result_owns_figures() -> None:
    """Enhanced-only ``return_result`` hands figure ownership to the caller."""
    matplotlib.use("Agg", force=True)
    try:
        from fincore.pyfolio import Pyfolio
        from fincore.report.artifacts import ReportArtifacts

        idx = pd.date_range("2024-01-01", periods=180, freq="B", tz="UTC")
        returns = pd.Series([0.001 if i % 2 == 0 else -0.0006 for i in range(len(idx))], index=idx)

        result = Pyfolio().create_full_tear_sheet(returns, return_result=True)

        assert isinstance(result, ReportArtifacts)
        assert result.backend == "matplotlib"
        assert len(result.figures) > 0
        result.close()
        assert result.closed
    finally:
        matplotlib.use("svg")


def test_full_tear_sheet_default_return_is_none() -> None:
    """The default enhanced tear-sheet return stays None (no API drift)."""
    matplotlib.use("Agg", force=True)
    try:
        from fincore.pyfolio import Pyfolio

        idx = pd.date_range("2024-01-01", periods=60, freq="B", tz="UTC")
        returns = pd.Series([0.001 if i % 2 == 0 else -0.0006 for i in range(len(idx))], index=idx)

        result = Pyfolio().create_full_tear_sheet(returns)

        assert result is None
    finally:
        matplotlib.use("svg")
