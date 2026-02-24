"""Shared fixtures for EVT tests."""

import numpy as np
import pytest


@pytest.fixture
def heavy_tailed_data():
    """Create heavy-tailed data for EVT testing."""
    np.random.seed(42)
    # Use t-distribution with low degrees of freedom for heavy tails
    return np.random.standard_t(3, 5000)


@pytest.fixture
def light_tailed_data():
    """Create light-tailed data for EVT testing."""
    np.random.seed(42)
    return np.random.normal(0, 0.01, 5000)
