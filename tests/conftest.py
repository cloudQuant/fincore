"""Shared pytest configuration and fixtures.

This file contains:
- Priority markers configuration (p0, p1, p2, p3)
- Shared fixtures for test data
- Custom pytest hooks

Priority Levels:
- P0: Critical - core metrics (sharpe_ratio, max_drawdown, etc.), security, compliance
- P1: High - frequently used features, important edge cases
- P2: Medium - secondary features, admin functions, edge cases
- P3: Low - rarely used, cosmetic, deprecation tests
"""
from __future__ import annotations

import pytest

# ==============================================================================
# Priority Markers - Apply to test classes and methods
# ==============================================================================
# See pyproject.toml [tool.pytest.ini_options].markers for marker definitions
#
# Usage examples:
#
# @pytest.mark.p0  # Critical: core financial metric
# def test_sharpe_ratio():
#     ...
#
# @pytest.mark.p1  # High: important edge case
# def test_sharpe_ratio_with_nan():
#     ...
#
# @pytest.mark.p2  # Medium: nice-to-have validation
# def test_sharpe_ratio_boundary_conditions():
#     ...
#
# Run selective tests:
#   pytest -m p0                    # Only critical tests
#   pytest -m "p0 or p1"            # Critical + high priority
#   pytest -m "not slow"            # Skip slow tests
# ==============================================================================


def pytest_configure(config):
    """Configure custom pytest markers.

    This hook ensures markers are registered even if not in pyproject.toml.
    """
    config.addinivalue_line("markers", "p0: Critical priority tests (core metrics, security)")
    config.addinivalue_line("markers", "p1: High priority tests (frequently used)")
    config.addinivalue_line("markers", "p2: Medium priority tests (secondary features)")
    config.addinivalue_line("markers", "p3: Low priority tests (rarely used)")


# ==============================================================================
# P0: Critical Metrics - These are the core financial metrics
# ==============================================================================
P0_METRICS = [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "annual_return",
    "volatility",
    "alpha",
    "beta",
    "cum_returns",
    "cum_returns_final",
    "value_at_risk",
    "conditional_value_at_risk",
]

P0_FEATURES = [
    "returns_calculation",
    "drawdown_analysis",
    "risk_adjusted_returns",
]


# ==============================================================================
# Test collection hooks for automatic priority assignment
# ==============================================================================

def pytest_collection_modifyitems(items):
    """Automatically assign P0 markers to critical metric tests.

    This hook looks for test functions that test P0 metrics and marks them
    automatically if not already marked.
    """
    for item in items:
        # Skip if already has a priority marker
        if any(marker in item.keywords for marker in ["p0", "p1", "p2", "p3"]):
            continue

        # Check if test name contains a P0 metric
        test_name = item.name.lower()
        for metric in P0_METRICS:
            if metric in test_name:
                item.add_marker(pytest.mark.p0)
                break
