"""Tests for utility functions in tearsheets.

Tests fallback display and markdown functions.
Split from test_sheets_more_coverage.py for maintainability.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import fincore.tearsheets.sheets as sheets


def test_fallback_display_and_markdown_functions() -> None:
    """Test fallback display and markdown functions (lines 57, 61)."""
    # Test _fallback_display - should print the objects
    f = io.StringIO()
    with redirect_stdout(f):
        sheets._fallback_display("test", 123)
    assert "test" in f.getvalue()

    # Test _fallback_markdown - should return the text as-is
    result = sheets._fallback_markdown("# Test Markdown")
    assert result == "# Test Markdown"
