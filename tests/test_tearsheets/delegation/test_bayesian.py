"""Tests for bayesian tear sheet.

Tests create_bayesian_tear_sheet delegation.
Split from test_sheets_delegation.py for maintainability.
"""

from __future__ import annotations

import pandas as pd
import pytest

import fincore.tearsheets.sheets as sheets


def test_create_bayesian_tear_sheet_requires_live_start_date() -> None:
    """Test that bayesian tear sheet requires live_start_date."""
    with pytest.raises(NotImplementedError):
        sheets.create_bayesian_tear_sheet(object(), pd.Series(dtype=float))
