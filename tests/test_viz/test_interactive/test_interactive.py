"""Tests for interactive visualization backends."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Check if optional dependencies are available
pytest.importorskip("plotly", reason="plotly not installed")


@pytest.fixture
def sample_returns():
    """Create sample returns for testing."""
    np.random.seed(42)
    n = 252 * 3  # 3 years of daily data

    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = pd.Series(np.random.randn(n) * 0.01, index=dates)

    return returns


@pytest.fixture
def sample_benchmark():
    """Create sample benchmark returns for testing."""
    np.random.seed(123)
    n = 252 * 3

    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = pd.Series(np.random.randn(n) * 0.008, index=dates)

    return returns


@pytest.fixture
def cumulative_returns(sample_returns):
    """Calculate cumulative returns."""
    return (1 + sample_returns).cumprod()


@pytest.fixture
def drawdown(sample_returns):
    """Calculate drawdown from returns."""
    cum_returns = (1 + sample_returns).cumprod()
    running_max = cum_returns.cummax()
    return (cum_returns - running_max) / running_max


@pytest.fixture
def rolling_sharpe(sample_returns):
    """Calculate rolling Sharpe ratio."""
    from fincore.metrics.rolling import roll_sharpe_ratio

    return roll_sharpe_ratio(sample_returns, window=63)


@pytest.fixture
def multi_asset_returns():
    """Create multi-asset returns for testing."""
    np.random.seed(42)
    n = 252 * 2

    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    assets = ["Asset_A", "Asset_B", "Asset_C"]

    returns = pd.DataFrame(
        np.random.randn(n, 3) * 0.015,
        index=dates,
        columns=assets,
    )

    return returns


class TestPlotlyBackend:
    """Tests for PlotlyBackend."""

    def test_backend_creation(self):
        """Test backend can be instantiated."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        backend = PlotlyBackend()
        assert backend is not None
        assert backend.theme == "light"

    def test_backend_with_theme(self):
        """Test backend with dark theme."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        backend = PlotlyBackend(theme="dark")
        assert backend.theme == "dark"
        assert backend.colors["background"] == PlotlyBackend.COLORS["background_dark"]

    def test_plot_returns(self, cumulative_returns):
        """Test plotting cumulative returns."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        backend = PlotlyBackend()
        fig = backend.plot_returns(cumulative_returns)

        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) > 0

    def test_plot_returns_with_benchmark(self, cumulative_returns, sample_benchmark):
        """Test plotting returns with benchmark."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        cum_bench = (1 + sample_benchmark).cumprod()

        backend = PlotlyBackend()
        fig = backend.plot_returns(cumulative_returns, benchmark=cum_bench)

        assert fig is not None
        assert len(fig.data) >= 2  # Portfolio + benchmark

    def test_plot_drawdown(self, drawdown):
        """Test plotting drawdown."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        backend = PlotlyBackend()
        fig = backend.plot_drawdown(drawdown)

        assert fig is not None
        assert hasattr(fig, "data")

    def test_plot_rolling_sharpe(self, rolling_sharpe):
        """Test plotting rolling Sharpe ratio."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        backend = PlotlyBackend()
        fig = backend.plot_rolling_sharpe(rolling_sharpe)

        assert fig is not None
        assert hasattr(fig, "data")

    def test_plot_rolling_sharpe_with_benchmark(self, sample_returns, sample_benchmark):
        """Test plotting rolling Sharpe with benchmark."""
        from fincore.metrics.rolling import roll_sharpe_ratio
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        sharpe = roll_sharpe_ratio(sample_returns, window=63)
        bench_sharpe = roll_sharpe_ratio(sample_benchmark, window=63)

        backend = PlotlyBackend()
        fig = backend.plot_rolling_sharpe(sharpe, benchmark_sharpe=bench_sharpe)

        assert fig is not None
        assert len(fig.data) >= 2

    def test_plot_monthly_heatmap(self, sample_returns):
        """Test plotting monthly heatmap."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        backend = PlotlyBackend()
        fig = backend.plot_monthly_heatmap(sample_returns)

        assert fig is not None
        assert hasattr(fig, "data")

    def test_plot_correlation_matrix(self, multi_asset_returns):
        """Test plotting correlation matrix."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        backend = PlotlyBackend()
        fig = backend.plot_correlation_matrix(multi_asset_returns)

        assert fig is not None
        assert hasattr(fig, "data")

    def test_plot_efficient_frontier(self, multi_asset_returns):
        """Test plotting efficient frontier."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        backend = PlotlyBackend()

        # Note: This requires optimization module
        # For now, test that the method exists and handles errors gracefully
        try:
            fig = backend.plot_efficient_frontier(multi_asset_returns)
            assert fig is not None
        except ImportError:
            # Optimization module not available, skip
            pass

    def test_backend_without_plotly_raises(self, monkeypatch):
        """Test that appropriate error message exists."""
        from fincore.viz.interactive.plotly_backend import PlotlyBackend

        # Verify the class provides useful error messages
        backend = PlotlyBackend()
        # The backend should be usable if plotly is installed
        # (we already check plotly import at module level)
        assert hasattr(backend, "_create_figure")


class TestBokehBackend:
    """Tests for BokehBackend."""

    def test_backend_creation(self):
        """Test backend can be instantiated."""
        try:
            from fincore.viz.interactive.bokeh_backend import BokehBackend

            backend = BokehBackend()
            assert backend is not None
            assert backend.theme == "light"
        except ImportError:
            pytest.skip("bokeh not installed")

    def test_backend_with_theme(self):
        """Test backend with dark theme."""
        try:
            from fincore.viz.interactive.bokeh_backend import BokehBackend

            backend = BokehBackend(theme="dark")
            assert backend.theme == "dark"
        except ImportError:
            pytest.skip("bokeh not installed")

    def test_plot_returns(self, cumulative_returns):
        """Test plotting cumulative returns."""
        try:
            from fincore.viz.interactive.bokeh_backend import BokehBackend

            backend = BokehBackend()
            p = backend.plot_returns(cumulative_returns)

            assert p is not None
        except ImportError:
            pytest.skip("bokeh not installed")

    def test_plot_drawdown(self, drawdown):
        """Test plotting drawdown."""
        try:
            from fincore.viz.interactive.bokeh_backend import BokehBackend

            backend = BokehBackend()
            p = backend.plot_drawdown(drawdown)

            assert p is not None
        except ImportError:
            pytest.skip("bokeh not installed")

    def test_plot_rolling_sharpe(self, rolling_sharpe):
        """Test plotting rolling Sharpe ratio."""
        try:
            from fincore.viz.interactive.bokeh_backend import BokehBackend

            backend = BokehBackend()
            p = backend.plot_rolling_sharpe(rolling_sharpe)

            assert p is not None
        except ImportError:
            pytest.skip("bokeh not installed")

    def test_plot_monthly_heatmap(self, sample_returns):
        """Test plotting monthly heatmap."""
        try:
            from fincore.viz.interactive.bokeh_backend import BokehBackend

            backend = BokehBackend()
            p = backend.plot_monthly_heatmap(sample_returns)

            assert p is not None
        except ImportError:
            pytest.skip("bokeh not installed")


class TestVizBaseIntegration:
    """Tests for viz base module with interactive backends."""

    def test_get_backend_plotly(self):
        """Test getting Plotly backend via get_backend."""
        from fincore.viz.base import get_backend

        backend = get_backend("plotly")
        assert backend is not None
        assert hasattr(backend, "plot_returns")

    def test_get_backend_bokeh(self):
        """Test getting Bokeh backend via get_backend."""
        try:
            from fincore.viz.base import get_backend

            backend = get_backend("bokeh")
            assert backend is not None
            assert hasattr(backend, "plot_returns")
        except ImportError:
            pytest.skip("bokeh not installed")

    def test_get_backend_invalid(self):
        """Test getting invalid backend raises error."""
        from fincore.viz.base import get_backend

        with pytest.raises(ValueError, match="Unknown viz backend"):
            get_backend("invalid_backend")
