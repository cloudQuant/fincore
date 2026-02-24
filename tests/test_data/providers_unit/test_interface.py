"""Tests for DataProvider interface and date validation.

Tests for DataProvider abstract base class and date validation edge cases.
Split from test_providers_unit.py for maintainability.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest


class TestDataProviderInterface:
    """Tests for DataProvider abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Test that DataProvider cannot be instantiated directly."""
        from fincore.data.providers import DataProvider

        with pytest.raises(TypeError):
            DataProvider()

    def test_validate_dates_string(self):
        """Test date validation with string dates."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        start, end = provider.validate_dates("2020-01-01", "2020-12-31")
        assert start == pd.to_datetime("2020-01-01")
        assert end == pd.to_datetime("2020-12-31")

    def test_validate_dates_datetime(self):
        """Test date validation with datetime objects."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        dt_start = datetime(2020, 1, 1)
        dt_end = datetime(2020, 12, 31)
        start, end = provider.validate_dates(dt_start, dt_end)
        assert start == pd.to_datetime(dt_start)
        assert end == pd.to_datetime(dt_end)

    def test_validate_dates_invalid_range(self):
        """Test date validation rejects invalid date range."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        with pytest.raises(ValueError):
            provider.validate_dates("2020-12-31", "2020-01-01")


class TestDataProviderDateValidationEdgeCases:
    """Tests for edge cases in date validation."""

    def test_validate_dates_with_nat_values(self):
        """Test date validation with NaT values (line 197)."""
        from fincore.data.providers import YahooFinanceProvider

        provider = YahooFinanceProvider()

        # Test with NaT
        with pytest.raises(ValueError, match="Invalid date values"):
            provider.validate_dates(pd.NaT, "2020-12-31")

        with pytest.raises(ValueError, match="Invalid date values"):
            provider.validate_dates("2020-01-01", pd.NaT)
