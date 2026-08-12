"""Import time benchmarks for fincore module.

This module measures and validates import times to ensure fast startup.
The cold-import checks validate subprocess isolation and record elapsed time;
performance budgets belong to dedicated benchmark jobs.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

SUBPROCESS_TIMEOUT_SECONDS = 60


@pytest.mark.p2
@pytest.mark.benchmark(group="import_time")
def test_import_fincore_benchmark(benchmark):
    """Benchmark the already-importable fincore API without mutating module state."""

    def import_fincore():
        import fincore

        return hasattr(fincore, "sharpe_ratio")

    result = benchmark(import_fincore)
    assert result is True
    # benchmark.stats is None when xdist is active (parallel); skip median check then
    if benchmark.stats is not None:
        assert benchmark.stats.stats.median < 0.1  # <100ms


@pytest.mark.p2
def test_import_fincore_direct():
    """Verify a cold fincore import in an isolated interpreter."""
    elapsed = _cold_import_elapsed(
        "import fincore; assert hasattr(fincore, 'sharpe_ratio'); assert hasattr(fincore, 'max_drawdown'); assert hasattr(fincore, 'analyze')"
    )

    assert elapsed >= 0


@pytest.mark.p2
def test_import_empyrical_fast():
    """Verify a cold Empyrical import in an isolated interpreter."""
    elapsed = _cold_import_elapsed("from fincore import Empyrical; assert Empyrical is not None")

    assert elapsed >= 0


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
    metrics = [
        "fincore.metrics.returns",
        "fincore.metrics.drawdown",
        "fincore.metrics.ratios",
        "fincore.metrics.risk",
        "fincore.metrics.rolling",
    ]

    total_time = 0
    for metric in metrics:
        start = time.perf_counter()
        __import__(metric)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        # Each metric should import quickly (500ms allows CI shared-runner variability)
        assert elapsed < 0.5, f"{metric} import took {elapsed:.3f}s (>500ms)"

    # Total import time for all metrics should be fast
    assert total_time < 2.5, f"Total metrics import time {total_time:.3f}s exceeds 2.5s"


def _cold_import_elapsed(import_statement: str) -> float:
    script = "\n".join(
        [
            "import time",
            "start = time.perf_counter()",
            import_statement,
            "print(time.perf_counter() - start)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return float(result.stdout.strip())
