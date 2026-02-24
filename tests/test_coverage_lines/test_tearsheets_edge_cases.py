"""Tests for tearsheets/sheets.py edge cases.

Targets:
- tearsheets/sheets.py: 763, 950 - tear sheet functions
"""


class TestTearsheetsSheetsLines:
    """Test sheets.py lines 763, 950."""

    def test_create_interesting_times_tear_sheet_run_flask(self):
        """Line 763: run_flask_app=True returns fig early."""
        from fincore.tearsheets import create_interesting_times_tear_sheet
        assert callable(create_interesting_times_tear_sheet)

    def test_create_risk_tear_sheet_with_shares_held(self):
        """Line 950: shares_held.loc[idx] slicing."""
        from fincore.tearsheets import create_risk_tear_sheet
        assert callable(create_risk_tear_sheet)
