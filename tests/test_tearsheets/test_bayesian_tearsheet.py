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
