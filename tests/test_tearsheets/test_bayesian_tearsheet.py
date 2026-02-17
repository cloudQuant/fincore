"""Smoke tests for fincore.tearsheets.bayesian plotting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.empyrical import Empyrical
from fincore.tearsheets import bayesian as tb


@pytest.fixture(autouse=True)
def _mpl_cleanup():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def test_plot_bayes_cone_returns_scalar_score_and_draws_plot():
    import matplotlib.pyplot as plt

    idx_train = pd.date_range("2020-01-01", periods=60, freq="D")
    idx_test = pd.date_range("2020-03-01", periods=20, freq="D")

    rng = np.random.RandomState(0)
    returns_train = pd.Series(rng.normal(0, 0.01, len(idx_train)), index=idx_train)
    returns_test = pd.Series(rng.normal(0, 0.01, len(idx_test)), index=idx_test)

    ppc = rng.normal(0, 0.01, size=(200, len(idx_test)))

    emp = Empyrical(returns_train)
    _, ax = plt.subplots()
    score = tb.plot_bayes_cone(emp, returns_train, returns_test, ppc, plot_train_len=10, ax=ax)
    assert isinstance(score, float)
    assert 0.0 <= score <= 0.5


def test_plot_best_accepts_dataframe_trace_and_derives_missing_columns():
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(1)
    n = 300
    trace = pd.DataFrame(
        {
            "group1_mean": rng.normal(0.0, 0.001, n),
            "group2_mean": rng.normal(0.0002, 0.001, n),
            "group1_std": rng.uniform(0.005, 0.02, n),
            "group2_std": rng.uniform(0.005, 0.02, n),
            "difference_of_means": rng.normal(0.0002, 0.001, n),
        }
    )

    emp = Empyrical(pd.Series([0.0, 0.0], index=pd.date_range("2020-01-01", periods=2)))
    _, axs = plt.subplots(ncols=2, nrows=4)
    tb.plot_best(emp, trace=trace, burn=10, axs=axs)


def test_plot_stoch_vol_accepts_stub_trace():
    import matplotlib.pyplot as plt

    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    data = pd.Series(np.linspace(-0.02, 0.02, len(idx)), index=idx)

    # Shape matches what plot_stoch_vol expects: (draws, time).
    s = np.abs(np.random.RandomState(0).normal(0.0, 0.01, size=(120, len(idx))))

    class Trace:
        def __getitem__(self, key):
            name, step = key
            assert name == "s"
            return s[step, :]

    emp = Empyrical(data)
    _, ax = plt.subplots()
    ax = tb.plot_stoch_vol(emp, data, trace=Trace(), ax=ax)
    assert ax is not None


class TestPlotBestErrors:
    """Test error conditions in plot_best."""

    def test_plot_best_raises_error_when_no_trace_or_data(self):
        """Test that plot_best raises error when neither trace nor data are provided."""
        emp = Empyrical(pd.Series([0.0, 0.0], index=pd.date_range("2020-01-01", periods=2)))

        with pytest.raises(ValueError, match="Either pass trace, or pass both data_train and data_test"):
            tb.plot_best(emp)

    def test_plot_best_raises_error_when_only_train_data(self):
        """Test that plot_best raises error when only data_train is provided."""
        emp = Empyrical(pd.Series([0.0, 0.0], index=pd.date_range("2020-01-01", periods=2)))
        data_train = pd.Series([0.01] * 50, index=pd.date_range("2020-01-01", periods=50))

        with pytest.raises(ValueError, match="Either pass trace, or pass both data_train and data_test"):
            tb.plot_best(emp, data_train=data_train)

    def test_plot_best_raises_error_when_only_test_data(self):
        """Test that plot_best raises error when only data_test is provided."""
        emp = Empyrical(pd.Series([0.0, 0.0], index=pd.date_range("2020-01-01", periods=2)))
        data_test = pd.Series([0.01] * 50, index=pd.date_range("2020-02-01", periods=50))

        with pytest.raises(ValueError, match="Either pass trace, or pass both data_train and data_test"):
            tb.plot_best(emp, data_test=data_test)

    def test_plot_best_creates_axes_when_none(self):
        """Test that plot_best creates axes when none are provided."""
        import matplotlib.pyplot as plt

        rng = np.random.RandomState(1)
        n = 300
        trace = pd.DataFrame(
            {
                "group1_mean": rng.normal(0.0, 0.001, n),
                "group2_mean": rng.normal(0.0002, 0.001, n),
                "group1_std": rng.uniform(0.005, 0.02, n),
                "group2_std": rng.uniform(0.005, 0.02, n),
                "difference_of_means": rng.normal(0.0002, 0.001, n),
            }
        )

        emp = Empyrical(pd.Series([0.0, 0.0], index=pd.date_range("2020-01-01", periods=2)))
        tb.plot_best(emp, trace=trace, burn=10, axs=None)
        plt.close("all")

    def test_plot_best_creates_derived_difference_column(self):
        """Test that plot_best creates difference of means column when missing."""
        import matplotlib.pyplot as plt

        rng = np.random.RandomState(2)
        n = 300
        trace = pd.DataFrame(
            {
                "group1_mean": rng.normal(0.0, 0.001, n),
                "group2_mean": rng.normal(0.0002, 0.001, n),
                "group1_std": rng.uniform(0.005, 0.02, n),
                "group2_std": rng.uniform(0.005, 0.02, n),
                # No difference_of_means or difference of means column
            }
        )

        emp = Empyrical(pd.Series([0.0, 0.0], index=pd.date_range("2020-01-01", periods=2)))
        _, axs = plt.subplots(ncols=2, nrows=4)
        tb.plot_best(emp, trace=trace, axs=axs)

    def test_plot_best_raises_error_on_insufficient_axes(self):
        """Test that plot_best raises error when insufficient axes provided."""
        import matplotlib.pyplot as plt

        rng = np.random.RandomState(1)
        n = 300
        trace = pd.DataFrame(
            {
                "group1_mean": rng.normal(0.0, 0.001, n),
                "group2_mean": rng.normal(0.0002, 0.001, n),
                "group1_std": rng.uniform(0.005, 0.02, n),
                "group2_std": rng.uniform(0.005, 0.02, n),
                "difference_of_means": rng.normal(0.0002, 0.001, n),
            }
        )

        emp = Empyrical(pd.Series([0.0, 0.0], index=pd.date_range("2020-01-01", periods=2)))
        _, axs = plt.subplots(ncols=2, nrows=2)  # Only 4 axes, need at least 7

        with pytest.raises(ValueError, match="axs must contain at least 7"):
            tb.plot_best(emp, trace=trace, axs=axs)


class TestPlotStochVolErrors:
    """Test plot_stoch_vol behavior."""

    def test_plot_stoch_vol_creates_ax_when_none(self):
        """Test that plot_stoch_vol creates axis when none is provided."""
        idx = pd.date_range("2020-01-01", periods=30, freq="D")
        data = pd.Series(np.linspace(-0.02, 0.02, len(idx)), index=idx)

        s = np.abs(np.random.RandomState(0).normal(0.0, 0.01, size=(120, len(idx))))

        class Trace:
            def __getitem__(self, key):
                name, step = key
                return s[step, :]

        emp = Empyrical(data)
        ax = tb.plot_stoch_vol(emp, data, trace=Trace(), ax=None)
        assert ax is not None


class TestPlotBayesCone:
    """Test plot_bayes_cone behavior."""

    def test_plot_bayes_cone_creates_ax_when_none(self):
        """Test that plot_bayes_cone creates axis when none is provided."""
        import matplotlib.pyplot as plt

        idx_train = pd.date_range("2020-01-01", periods=60, freq="D")
        idx_test = pd.date_range("2020-03-01", periods=20, freq="D")

        rng = np.random.RandomState(3)
        returns_train = pd.Series(rng.normal(0, 0.01, len(idx_train)), index=idx_train)
        returns_test = pd.Series(rng.normal(0, 0.01, len(idx_test)), index=idx_test)
        ppc = rng.normal(0, 0.01, size=(200, len(idx_test)))

        emp = Empyrical(returns_train)
        score = tb.plot_bayes_cone(emp, returns_train, returns_test, ppc, ax=None)
        assert isinstance(score, float)
        plt.close("all")


def test_plot_best_creates_trace_from_data_line_49(monkeypatch):
    """Test plot_best creates trace via model_best when data provided (line 49)."""
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(1)
    n = 300
    mock_trace = pd.DataFrame(
        {
            "group1_mean": rng.normal(0.0, 0.001, n),
            "group2_mean": rng.normal(0.0002, 0.001, n),
            "group1_std": rng.uniform(0.005, 0.02, n),
            "group2_std": rng.uniform(0.005, 0.02, n),
            "difference_of_means": rng.normal(0.0002, 0.001, n),
        }
    )

    class MockEmp:
        def model_best(self, data_train, data_test, samples):
            return mock_trace

    idx_train = pd.date_range("2020-01-01", periods=60, freq="D")
    idx_test = pd.date_range("2020-03-01", periods=20, freq="D")
    data_train = pd.Series(rng.normal(0, 0.01, len(idx_train)), index=idx_train)
    data_test = pd.Series(rng.normal(0, 0.01, len(idx_test)), index=idx_test)

    _, axs = plt.subplots(ncols=2, nrows=4)
    tb.plot_best(MockEmp(), data_train=data_train, data_test=data_test, axs=axs)
    plt.close("all")


def test_plot_best_with_dict_trace_line_57():
    """Test plot_best with dict trace (line 57-58)."""
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(1)
    n = 300
    # Use a dict-like object instead of DataFrame
    dict_trace = {
        "group1_mean": rng.normal(0.0, 0.001, n),
        "group2_mean": rng.normal(0.0002, 0.001, n),
        "group1_std": rng.uniform(0.005, 0.02, n),
        "group2_std": rng.uniform(0.005, 0.02, n),
        "difference_of_means": rng.normal(0.0002, 0.001, n),
    }

    emp = Empyrical(pd.Series([0.0, 0.0], index=pd.date_range("2020-01-01", periods=2)))
    _, axs = plt.subplots(ncols=2, nrows=4)
    tb.plot_best(emp, trace=dict_trace, burn=10, axs=axs)
    plt.close("all")


def test_plot_stoch_vol_creates_trace_from_data_line_153(monkeypatch):
    """Test plot_stoch_vol creates trace via model_stoch_vol when trace is None (line 153)."""
    import matplotlib.pyplot as plt

    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    data = pd.Series(np.linspace(-0.02, 0.02, len(idx)), index=idx)

    # Mock trace with expected format
    s = np.abs(np.random.RandomState(0).normal(0.0, 0.01, size=(120, len(idx))))

    class MockEmp:
        def model_stoch_vol(self, data):
            class Trace:
                def __getitem__(self, key):
                    name, step = key
                    return s[step, :]

            return Trace()

    ax = tb.plot_stoch_vol(MockEmp(), data, trace=None, ax=None)
    assert ax is not None
    plt.close("all")
