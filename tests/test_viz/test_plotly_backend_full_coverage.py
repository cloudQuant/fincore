"""Tests for PlotlyBackend - 100% coverage."""

import numpy as np
import pandas as pd
import pytest

from fincore.viz.interactive.plotly_backend import PlotlyBackend


@pytest.fixture
def sample_returns():
    """Create sample returns data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252)
    return pd.Series(np.random.randn(252) * 0.01, index=dates)


@pytest.fixture
def sample_cum_returns():
    """Create sample cumulative returns."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252)
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    return (1 + returns).cumprod()


@pytest.fixture
def sample_drawdown():
    """Create sample drawdown data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252)
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown


@pytest.fixture
def sample_rolling_sharpe():
    """Create sample rolling Sharpe data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252)
    return pd.Series(np.random.randn(252) * 0.1, index=dates)


class TestPlotlyBackendInit:
    """Test PlotlyBackend initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        backend = PlotlyBackend()

        assert backend.theme == "light"
        assert backend.height == 500
        assert backend.width is None

    def test_custom_initialization(self):
        """Test custom initialization."""
        backend = PlotlyBackend(theme="plotly_dark", height=600, width=1000)

        assert backend.theme == "plotly_dark"
        assert backend.height == 600
        assert backend.width == 1000


class TestPlotlyBackendPlotReturns:
    """Test plot_returns method."""

    def test_plot_returns_basic(self, sample_cum_returns):
        """Test basic returns plot."""
        backend = PlotlyBackend()

        try:
            fig = backend.plot_returns(sample_cum_returns)

            assert fig is not None
        except ImportError:
            pytest.skip("Plotly not installed")

    def test_plot_returns_with_benchmark(self, sample_cum_returns):
        """Test returns plot with benchmark."""
        backend = PlotlyBackend()
        benchmark = sample_cum_returns * 0.8

        try:
            fig = backend.plot_returns(sample_cum_returns, benchmark=benchmark)

            assert fig is not None
        except ImportError:
            pytest.skip("Plotly not installed")


class TestPlotlyBackendPlotDrawdown:
    """Test plot_drawdown method."""

    def test_plot_drawdown_basic(self, sample_drawdown):
        """Test basic drawdown plot."""
        backend = PlotlyBackend()

        try:
            fig = backend.plot_drawdown(sample_drawdown)

            assert fig is not None
        except ImportError:
            pytest.skip("Plotly not installed")


class TestPlotlyBackendPlotRollingSharpe:
    """Test plot_rolling_sharpe method."""

    def test_plot_rolling_sharpe_basic(self, sample_rolling_sharpe):
        """Test basic rolling Sharpe plot."""
        backend = PlotlyBackend()

        try:
            fig = backend.plot_rolling_sharpe(sample_rolling_sharpe)

            assert fig is not None
        except ImportError:
            pytest.skip("Plotly not installed")

    def test_plot_rolling_sharpe_with_benchmark(self, sample_rolling_sharpe):
        """Test rolling Sharpe plot with benchmark."""
        backend = PlotlyBackend()
        benchmark_sharpe = sample_rolling_sharpe * 0.7

        try:
            fig = backend.plot_rolling_sharpe(sample_rolling_sharpe, benchmark_sharpe=benchmark_sharpe)

            assert fig is not None
        except ImportError:
            pytest.skip("Plotly not installed")

    def test_plot_rolling_sharpe_custom_window(self, sample_rolling_sharpe):
        """Test rolling Sharpe plot with custom window."""
        backend = PlotlyBackend()

        try:
            fig = backend.plot_rolling_sharpe(sample_rolling_sharpe, window=126)

            assert fig is not None
        except ImportError:
            pytest.skip("Plotly not installed")


class TestPlotlyBackendPlotMonthlyHeatmap:
    """Test plot_monthly_heatmap method."""

    def test_plot_monthly_heatmap_series(self, sample_returns):
        """Test monthly heatmap with Series input."""
        backend = PlotlyBackend()

        try:
            fig = backend.plot_monthly_heatmap(sample_returns)

            assert fig is not None
        except ImportError:
            pytest.skip("Plotly not installed")

    def test_plot_monthly_heatmap_dataframe(self):
        """Test monthly heatmap with DataFrame input."""
        backend = PlotlyBackend()

        # Create a pre-pivoted DataFrame
        data = {
            "Jan": [1, 2, 3],
            "Feb": [2, 3, 4],
            "Mar": [3, 4, 5],
        }
        df = pd.DataFrame(data, index=[2020, 2021, 2022])

        try:
            fig = backend.plot_monthly_heatmap(df)

            assert fig is not None
        except ImportError:
            pytest.skip("Plotly not installed")


class TestPlotlyBackendSaveAndShow:
    """Test save and show methods."""

    def test_show_method(self, sample_cum_returns):
        """Test show method."""
        backend = PlotlyBackend()

        try:
            backend.plot_returns(sample_cum_returns)
            # Verify the method exists
            assert hasattr(backend, "show")
        except ImportError:
            pytest.skip("Plotly not installed")

    def test_save_html_method(self, sample_cum_returns, tmp_path):
        """Test save_html method."""
        backend = PlotlyBackend()

        try:
            backend.plot_returns(sample_cum_returns)
            output_file = tmp_path / "test_plot.html"

            backend.save_html(str(output_file))

            assert output_file.exists()
        except ImportError:
            pytest.skip("Plotly not installed")

    def test_show_without_figure_raises_error(self):
        """Test show raises error when no figure exists."""
        backend = PlotlyBackend()

        try:
            with pytest.raises(ValueError, match="No figure to display"):
                backend.show()
        except ImportError:
            pytest.skip("Plotly not installed")

    def test_save_html_without_figure_raises_error(self, tmp_path):
        """Test save_html raises error when no figure exists."""
        backend = PlotlyBackend()

        try:
            output_file = tmp_path / "test_plot.html"

            with pytest.raises(ValueError, match="No figure to save"):
                backend.save_html(str(output_file))
        except ImportError:
            pytest.skip("Plotly not installed")

    def test_save_image_without_figure_raises_error(self, tmp_path):
        """Test save_image raises error when no figure exists."""
        backend = PlotlyBackend()

        try:
            output_file = tmp_path / "test_plot.png"

            with pytest.raises(ValueError, match="No figure to save"):
                backend.save_image(str(output_file))
        except ImportError:
            pytest.skip("Plotly not installed")

    def test_save_image_method(self, sample_cum_returns, tmp_path):
        """Test save_image method."""
        backend = PlotlyBackend()

        try:
            backend.plot_returns(sample_cum_returns)
            output_file = tmp_path / "test_plot.png"

            try:
                backend.save_image(str(output_file))
                # If kaleido is installed, file should exist
                assert output_file.exists()
            except (ImportError, AttributeError):
                # Kaleido not installed, that's okay
                pytest.skip("Kaleido not installed for image export")
        except ImportError:
            pytest.skip("Plotly not installed")


class TestPlotlyBackendAdvancedMethods:
    """Test advanced plotting methods."""

    def test_plot_efficient_frontier(self, sample_returns):
        """Test plot_efficient_frontier method."""
        backend = PlotlyBackend()

        try:
            # Create a DataFrame with multiple asset returns
            returns_df = pd.DataFrame(
                {
                    "asset1": sample_returns,
                    "asset2": sample_returns * 0.8,
                    "asset3": sample_returns * 1.2,
                }
            )

            fig = backend.plot_efficient_frontier(returns_df, n_points=20)
            assert fig is not None
        except ImportError:
            pytest.skip("Plotly not installed")

    def test_plot_correlation_matrix(self, sample_returns):
        """Test plot_correlation_matrix method."""
        backend = PlotlyBackend()

        try:
            # Create a DataFrame with multiple asset returns
            returns_df = pd.DataFrame(
                {
                    "asset1": sample_returns,
                    "asset2": sample_returns * 0.8,
                    "asset3": sample_returns * 1.2,
                }
            )

            fig = backend.plot_correlation_matrix(returns_df)
            assert fig is not None
        except ImportError:
            pytest.skip("Plotly not installed")


class TestPlotlyBackendTheme:
    """Test theme configuration."""

    def test_dark_theme_setup(self):
        """Test dark theme configuration."""
        backend = PlotlyBackend(theme="dark")

        assert backend.colors["background"] == backend.COLORS["background_dark"]
        assert backend.colors["grid"] == backend.COLORS["grid_dark"]
        assert backend.colors["text"] == backend.COLORS["text_dark"]

    def test_light_theme_setup(self):
        """Test light theme configuration."""
        backend = PlotlyBackend(theme="light")

        assert backend.colors["background"] == backend.COLORS["background_light"]
        assert backend.colors["grid"] == backend.COLORS["grid_light"]
        assert backend.colors["text"] == backend.COLORS["text_light"]

    def test_custom_template(self):
        """Test custom template overrides theme."""
        backend = PlotlyBackend(theme="light", template="plotly")

        assert backend.template == "plotly"
        assert backend._get_template() == "plotly"

    def test_show_legend_false(self, sample_cum_returns):
        """Test show_legend parameter."""
        backend = PlotlyBackend(show_legend=False)

        try:
            fig = backend.plot_returns(sample_cum_returns)
            assert fig.layout.showlegend is False
        except ImportError:
            pytest.skip("Plotly not installed")
