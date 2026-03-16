"""Import time benchmarks for fincore module.

This module measures and validates import times to ensure fast startup.
Target: import fincore in <0.5 seconds (generous for CI shared runners)
"""

from __future__ import annotations

import time

import pytest


@pytest.mark.p2
@pytest.mark.benchmark(group="import_time")
def test_import_fincore_benchmark(benchmark):
    """Benchmark fincore import (target: <0.1s)."""

    def import_fincore():
        import importlib
        import sys

        # Force reimport for accurate measurement
        if "fincore" in sys.modules:
            del sys.modules["fincore"]
            to_delete = [k for k in list(sys.modules.keys()) if k.startswith("fincore.")]
            for k in to_delete:
                del sys.modules[k]

        import fincore

        return fincore

    result = benchmark(import_fincore)
    assert result is not None
    assert hasattr(result, "sharpe_ratio")
    # benchmark.stats is None when xdist is active (parallel); skip median check then
    if benchmark.stats is not None:
        assert benchmark.stats.stats.median < 0.1  # <100ms


@pytest.mark.p2
def test_import_fincore_direct():
    """Verify fincore imports in reasonable time without benchmark.

    This test validates import time without pytest-benchmark dependency.
    """
    import importlib
    import sys

    # Clear cached imports
    if "fincore" in sys.modules:
        to_delete = [k for k in list(sys.modules.keys()) if k.startswith("fincore.")]
        for k in to_delete:
            del sys.modules[k]

    # Measure import time
    start = time.perf_counter()
    import fincore

    elapsed = time.perf_counter() - start

    # Assert import is fast (500ms allows CI shared-runner variability)
    assert elapsed < 0.5, f"Import time {elapsed:.3f}s exceeds 500ms target"

    # Verify basic functionality
    assert hasattr(fincore, "sharpe_ratio")
    assert hasattr(fincore, "max_drawdown")
    assert hasattr(fincore, "analyze")


@pytest.mark.p2
def test_import_empyrical_fast():
    """Verify Empyrical class import is fast."""
    import importlib
    import sys

    # Clear cached imports
    if "fincore.empyrical" in sys.modules:
        to_delete = [k for k in list(sys.modules.keys()) if "empyrical" in k]
        for k in to_delete:
            del sys.modules[k]

    # Measure import time
    start = time.perf_counter()
    from fincore import Empyrical

    elapsed = time.perf_counter() - start

    # Empyrical import should also be fast (500ms allows CI shared-runner variability)
    assert elapsed < 0.5, f"Empyrical import time {elapsed:.3f}s exceeds 500ms target"


@pytest.mark.p2
def test_import_lazy_module_deferred():
    """Verify that lazy loading defers heavy modules."""
    import sys

    # Note: In test environments, matplotlib may already be loaded
    # This test verifies that fincore doesn't actively load it on import

    # Store current state
    matplotlib_before = "matplotlib" in sys.modules
    pymc_before = "pymc" in sys.modules
    pandas_datareader_before = "pandas_datareader" in sys.modules

    # Import fincore (should NOT import viz, bayesian, etc.)
    import fincore

    # Check that fincore import didn't load new heavy modules
    assert ("matplotlib" in sys.modules) == matplotlib_before, "fincore import should not load matplotlib"
    assert ("pymc" in sys.modules) == pymc_before, "fincore import should not load pymc"
    assert ("pandas_datareader" in sys.modules) == pandas_datareader_before, (
        "fincore import should not load pandas_datareader"
    )

    # Now use Empyrical (should still NOT load viz)
    _ = fincore.Empyrical
    assert ("matplotlib" in sys.modules) == matplotlib_before, "Empyrical access should not load matplotlib"

    # Metrics should be loaded
    assert "fincore.metrics" in sys.modules


@pytest.mark.p2
def test_flat_api_import():
    """Verify flat API functions are accessible at import time."""
    import fincore

    # Common functions should be accessible
    assert hasattr(fincore, "sharpe_ratio")
    assert hasattr(fincore, "max_drawdown")
    assert hasattr(fincore, "cum_returns")
    assert hasattr(fincore, "annual_return")
    assert hasattr(fincore, "analyze")


@pytest.mark.p3
def test_import_all_metrics_individually():
    """Test that individual metric modules can be imported efficiently."""
    import sys
    import time

    metrics = [
        "fincore.metrics.returns",
        "fincore.metrics.drawdown",
        "fincore.metrics.ratios",
        "fincore.metrics.risk",
        "fincore.metrics.rolling",
    ]

    total_time = 0
    for metric in metrics:
        # Clear module cache
        if metric in sys.modules:
            del sys.modules[metric]

        start = time.perf_counter()
        __import__(metric)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        # Each metric should import quickly (500ms allows CI shared-runner variability)
        assert elapsed < 0.5, f"{metric} import took {elapsed:.3f}s (>500ms)"

    # Total import time for all metrics should be fast
    assert total_time < 2.5, f"Total metrics import time {total_time:.3f}s exceeds 2.5s"
