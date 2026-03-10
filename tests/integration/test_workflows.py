"""Integration tests for complete analysis workflows.

These tests validate end-to-end workflows from data loading to report generation,
ensuring all components work together correctly.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from fincore import (
    Empyrical,
    analyze,
    create_strategy_report,
    sharpe_ratio,
    max_drawdown,
    annual_return,
    annual_volatility,
    alpha,
    beta,
)
from fincore.core.context import AnalysisContext
from fincore.constants import DAILY


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def strategy_returns():
    """Generate realistic strategy returns for integration testing."""
    np.random.seed(42)
    n_days = 252 * 3  # 3 years of data
    returns = pd.Series(
        np.random.randn(n_days) * 0.01 + 0.0003,  # Slight positive drift
        index=pd.bdate_range("2020-01-01", periods=n_days),
    )
    return returns


@pytest.fixture
def benchmark_returns(strategy_returns):
    """Generate benchmark returns for comparison."""
    np.random.seed(123)
    n_days = len(strategy_returns)
    returns = pd.Series(
        np.random.randn(n_days) * 0.008 + 0.0002,  # Lower volatility and drift
        index=strategy_returns.index,
    )
    return returns


@pytest.fixture
def multi_strategy_returns():
    """Generate multiple strategy returns for portfolio testing."""
    np.random.seed(42)
    n_days = 252 * 2  # 2 years
    dates = pd.bdate_range("2021-01-01", periods=n_days)

    return pd.DataFrame(
        {
            "momentum": np.random.randn(n_days) * 0.012 + 0.0004,
            "value": np.random.randn(n_days) * 0.010 + 0.0003,
            "quality": np.random.randn(n_days) * 0.008 + 0.0002,
        },
        index=dates,
    )


# ==============================================================================
# Complete Workflow Tests
# ==============================================================================


@pytest.mark.integration
@pytest.mark.p1
class TestCompleteAnalysisWorkflow:
    """Test complete analysis workflows from data to insights."""

    def test_basic_workflow_series(self, strategy_returns, benchmark_returns):
        """Test basic workflow with Series input."""
        # 1. Analyze returns
        ctx = analyze(strategy_returns, factor_returns=benchmark_returns)

        # 2. Get performance statistics
        stats = ctx.perf_stats()

        # 3. Validate all metrics are present
        assert "Sharpe ratio" in stats
        assert "Max drawdown" in stats
        assert "Annual return" in stats

        # 4. Validate all metrics are finite
        for metric, value in stats.items():
            assert np.isfinite(value), f"{metric} is not finite: {value}"

        # 5. Validate reasonable ranges
        assert stats["Sharpe ratio"] > -5 and stats["Sharpe ratio"] < 5
        assert stats["Max drawdown"] <= 0
        assert abs(stats["Annual return"]) < 1  # <100% annual return

    def test_workflow_with_empyrical_class(self, strategy_returns, benchmark_returns):
        """Test workflow using Empyrical class."""
        # 1. Create Empyrical instance
        emp = Empyrical(returns=strategy_returns, factor_returns=benchmark_returns)

        # 2. Calculate key metrics using class methods (passing returns explicitly)
        sharpe = emp.sharpe_ratio(strategy_returns)
        sortino = emp.sortino_ratio(strategy_returns)
        max_dd = emp.max_drawdown(strategy_returns)
        annual_ret = emp.annual_return(strategy_returns)
        annual_vol = emp.annual_volatility(strategy_returns)
        alpha_val = emp.alpha(strategy_returns, benchmark_returns)
        beta_val = emp.beta(strategy_returns, benchmark_returns)

        # 3. Validate all metrics are finite
        assert np.isfinite(sharpe)
        assert np.isfinite(sortino)
        assert np.isfinite(max_dd)
        assert np.isfinite(annual_ret)
        assert np.isfinite(annual_vol)
        assert np.isfinite(alpha_val)
        assert np.isfinite(beta_val)

        # 4. Validate reasonable ranges
        assert max_dd <= 0
        assert annual_vol > 0
        assert beta_val > -2 and beta_val < 3

    def test_workflow_dataframe(self, multi_strategy_returns):
        """Test workflow with DataFrame input (multiple strategies)."""
        # 1. Analyze each strategy individually
        results = {}
        for col in multi_strategy_returns.columns:
            ctx = analyze(multi_strategy_returns[col])
            stats = ctx.perf_stats()
            results[col] = stats

        # 2. Validate results for all strategies
        assert len(results) == 3  # 3 strategies

        # 3. Validate each strategy
        for strategy, stats in results.items():
            assert np.isfinite(stats.get("Sharpe ratio", np.nan))
            assert stats.get("Max drawdown", 0) <= 0


@pytest.mark.integration
@pytest.mark.p1
class TestReportGenerationWorkflow:
    """Test report generation workflows."""

    def test_html_report_generation(self, strategy_returns, benchmark_returns, tmp_path):
        """Test HTML report generation."""
        # 1. Generate HTML report
        output_file = tmp_path / "test_report.html"
        report_path = create_strategy_report(
            strategy_returns,
            benchmark_rets=benchmark_returns,
            title="Strategy Performance Report",
            output=str(output_file),
        )

        # 2. Validate report file is generated
        assert report_path is not None
        assert isinstance(report_path, str)
        assert output_file.exists()

        # 3. Read and validate HTML content
        html_content = output_file.read_text(encoding="utf-8")
        assert len(html_content) > 1000  # Should be substantial

        # 4. Validate HTML structure
        assert "<html" in html_content.lower() or "<!doctype html" in html_content.lower()
        assert "</html>" in html_content.lower()

    def test_report_with_benchmark(self, strategy_returns, benchmark_returns, tmp_path):
        """Test report includes benchmark comparison."""
        output_file = tmp_path / "test_report_benchmark.html"
        report_path = create_strategy_report(
            strategy_returns,
            benchmark_rets=benchmark_returns,
            output=str(output_file),
        )

        # Read HTML content
        html_content = output_file.read_text(encoding="utf-8")

        # Should include relative metrics
        assert "alpha" in html_content.lower() or "beta" in html_content.lower()

    def test_report_without_benchmark(self, strategy_returns, tmp_path):
        """Test report without benchmark."""
        output_file = tmp_path / "test_report_no_benchmark.html"
        report_path = create_strategy_report(
            strategy_returns,
            output=str(output_file),
        )

        # Should still generate valid report file
        assert report_path is not None
        assert output_file.exists()
        html_content = output_file.read_text(encoding="utf-8")
        assert len(html_content) > 500


@pytest.mark.integration
@pytest.mark.p2
class TestAnalysisContextWorkflow:
    """Test AnalysisContext workflow methods."""

    def test_to_dict_workflow(self, strategy_returns):
        """Test converting analysis to dictionary."""
        ctx = AnalysisContext(strategy_returns)

        # Export to dict
        result_dict = ctx.to_dict()

        # Validate
        assert isinstance(result_dict, dict)
        assert "Sharpe ratio" in result_dict
        assert "Max drawdown" in result_dict

    def test_to_json_workflow(self, strategy_returns):
        """Test converting analysis to JSON."""
        ctx = AnalysisContext(strategy_returns)

        # Export to JSON
        json_str = ctx.to_json()

        # Validate
        assert isinstance(json_str, str)

        # Parse and validate
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "Sharpe ratio" in parsed

    def test_to_html_workflow(self, strategy_returns):
        """Test converting analysis to HTML."""
        ctx = AnalysisContext(strategy_returns)

        # Export to HTML
        html = ctx.to_html()

        # Validate
        assert isinstance(html, str)
        assert len(html) > 100

    def test_cached_properties(self, strategy_returns):
        """Test that cached properties work correctly."""
        ctx = AnalysisContext(strategy_returns)

        # Access same property twice
        sharpe1 = ctx.sharpe_ratio
        sharpe2 = ctx.sharpe_ratio

        # Should return same object (cached)
        assert sharpe1 is sharpe2


@pytest.mark.integration
@pytest.mark.p2
class TestDataConsistencyWorkflow:
    """Test data consistency across different workflows."""

    def test_empyrical_vs_flat_api(self, strategy_returns):
        """Test that Empyrical class and flat API give same results."""
        # Using Empyrical class (need to pass returns explicitly)
        emp = Empyrical(returns=strategy_returns)
        sharpe_class = emp.sharpe_ratio(strategy_returns)

        # Using flat API
        sharpe_flat = sharpe_ratio(strategy_returns)

        # Should be identical
        assert np.isclose(sharpe_class, sharpe_flat)

    def test_context_vs_empyrical(self, strategy_returns):
        """Test that AnalysisContext and Empyrical give same results."""
        # Using AnalysisContext
        ctx = AnalysisContext(strategy_returns)
        sharpe_ctx = ctx.sharpe_ratio

        # Using Empyrical (need to pass returns explicitly)
        emp = Empyrical(returns=strategy_returns)
        sharpe_emp = emp.sharpe_ratio(strategy_returns)

        # Should be identical or very close
        assert np.isclose(sharpe_ctx, sharpe_emp, rtol=1e-10)

    def test_different_input_types(self):
        """Test that different input types give consistent results."""
        np.random.seed(42)
        data = np.random.randn(100) * 0.01

        # Series input
        series = pd.Series(data)
        sharpe_series = sharpe_ratio(series)

        # Array input (if supported)
        try:
            sharpe_array = sharpe_ratio(data)
            # Should be close
            assert np.isclose(sharpe_series, sharpe_array, rtol=1e-10)
        except (TypeError, AttributeError):
            # Array input may not be supported
            pass


@pytest.mark.integration
@pytest.mark.p2
class TestPerformanceUnderLoad:
    """Test performance with larger datasets."""

    def test_large_dataset_workflow(self):
        """Test workflow with large dataset (10 years)."""
        # Generate 10 years of daily data
        np.random.seed(42)
        n_days = 2520
        returns = pd.Series(
            np.random.randn(n_days) * 0.01,
            index=pd.bdate_range("2010-01-01", periods=n_days),
        )

        # Run analysis
        ctx = analyze(returns)
        stats = ctx.perf_stats()

        # Validate - stats.values is a numpy array property
        assert all(np.isfinite(v) for v in stats.values)

    def test_multiple_strategies_efficiency(self):
        """Test efficiency with many strategies."""
        np.random.seed(42)
        n_strategies = 10
        n_days = 252

        returns_df = pd.DataFrame(
            np.random.randn(n_days, n_strategies) * 0.01,
            index=pd.bdate_range("2022-01-01", periods=n_days),
            columns=[f"strategy_{i}" for i in range(n_strategies)],
        )

        # Analyze each strategy
        results = {}
        for col in returns_df.columns:
            ctx = analyze(returns_df[col])
            stats = ctx.perf_stats()
            results[col] = stats

        # Validate all strategies analyzed
        assert len(results) == n_strategies
        for col, stats in results.items():
            assert np.isfinite(stats.get("Sharpe ratio", np.nan))


# ==============================================================================
# Integration Test Summary
# ==============================================================================

# These integration tests ensure:
# 1. End-to-end workflows work correctly
# 2. Different APIs give consistent results
# 3. Report generation works
# 4. Performance is acceptable
# 5. Data consistency across methods
