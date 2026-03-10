"""Shared pytest configuration and fixtures.

This file contains:
- Priority markers configuration (p0, p1, p2, p3)
- Shared fixtures for test data
- Custom pytest hooks
- Automatic test isolation and cleanup

Priority Levels:
- P0: Critical - core metrics (sharpe_ratio, max_drawdown, etc.), security, compliance
- P1: High - frequently used features, important edge cases
- P2: Medium - secondary features, admin functions, edge cases
- P3: Low - rarely used, cosmetic, deprecation tests
"""

from __future__ import annotations

import copy

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


# ==============================================================================
# Test Isolation - Automatic cleanup fixtures
# ==============================================================================


@pytest.fixture(autouse=True, scope="function")
def cleanup_module_cache():
    """Automatically cleanup fincore module cache after each test.

    This ensures tests don't affect each other through shared cached modules.
    """
    # Setup: save original state
    try:
        import fincore.empyrical as empyrical

        original_cache = copy.copy(empyrical._MODULE_CACHE)
    except (ImportError, AttributeError):
        # If module not loaded yet, nothing to save
        original_cache = {}

    yield

    # Teardown: restore original state
    try:
        import fincore.empyrical as empyrical

        empyrical._MODULE_CACHE.clear()
        empyrical._MODULE_CACHE.update(original_cache)
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True, scope="function")
def cleanup_empyrical_cache():
    """Automatically cleanup Empyrical class method caches after each test.

    This ensures instance method wrappers don't persist between tests.
    """
    yield

    # Teardown: clear cached bound methods
    try:
        from fincore.empyrical import Empyrical

        # Clear any cached bound methods
        for attr_name in dir(Empyrical):
            if attr_name.startswith("_bound_"):
                try:
                    delattr(Empyrical, attr_name)
                except AttributeError:
                    pass
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True, scope="function")
def reset_numpy_random_seed():
    """Reset numpy random seed before each test for reproducibility.

    Note: Tests that need specific random sequences should set their own seeds.
    """
    import numpy as np

    np.random.seed(None)  # Reset to unpredictable state
    yield
    # No cleanup needed


# ==============================================================================
# Test Isolation - Shared state detection (optional, for debugging)
# ==============================================================================


def pytest_runtest_makereport(item, call):
    """Hook to detect potential test isolation issues.

    This runs after each test phase (setup, call, teardown) and can be used
    to detect shared state pollution.
    """
    if call.when == "call" and call.excinfo is None:
        # Test passed - no need to check
        pass
    elif call.when == "call" and call.excinfo is not None:
        # Test failed - could log for debugging
        # For now, we just let pytest handle failure reporting
        pass


# ==============================================================================
# Performance fixtures for benchmarking
# ==============================================================================


@pytest.fixture
def small_returns():
    """Generate small returns dataset for quick tests (252 points = 1 year)."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    return pd.Series(
        np.random.randn(252) * 0.01,
        index=pd.bdate_range("2020-01-01", periods=252),
    )


@pytest.fixture
def medium_returns():
    """Generate medium returns dataset (2520 points = 10 years)."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    return pd.Series(
        np.random.randn(2520) * 0.01,
        index=pd.bdate_range("2010-01-01", periods=2520),
    )


@pytest.fixture
def large_returns():
    """Generate large returns dataset for performance testing (25200 points = 100 years)."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    return pd.Series(
        np.random.randn(25200) * 0.01,
        index=pd.bdate_range("1920-01-01", periods=25200),
    )


# ==============================================================================
# Edge case fixtures for robustness testing
# ==============================================================================


@pytest.fixture
def empty_returns():
    """Empty returns series for edge case testing."""
    import pandas as pd

    return pd.Series([], dtype=float)


@pytest.fixture
def single_value_returns():
    """Single value returns for edge case testing."""
    import pandas as pd

    return pd.Series([0.01])


@pytest.fixture
def all_nan_returns():
    """All NaN returns for edge case testing."""
    import numpy as np
    import pandas as pd

    return pd.Series([np.nan] * 100)


@pytest.fixture
def zero_volatility_returns():
    """Zero volatility returns (constant) for edge case testing."""
    import pandas as pd

    return pd.Series([0.01] * 100)


@pytest.fixture
def extreme_values_returns():
    """Extreme values for edge case testing."""
    import numpy as np
    import pandas as pd

    return pd.Series([1e10, -1e10, 1e-10, -1e-10, 0.01])
