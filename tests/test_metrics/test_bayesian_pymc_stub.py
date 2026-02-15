"""Tests for fincore.metrics.bayesian using a stubbed PyMC module.

These tests cover the model-building functions without requiring a real
PyMC dependency (or any sampling).
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd


def _install_fake_pymc(monkeypatch):
    calls: list[tuple[str, tuple, dict]] = []

    class _FakeModel:
        def __enter__(self):
            calls.append(("Model.__enter__", (), {}))
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            calls.append(("Model.__exit__", (exc_type, exc_val, exc_tb), {}))
            return False

    pm = types.ModuleType("pymc")

    def _record(name, ret):
        def f(*args, **kwargs):
            calls.append((name, args, kwargs))
            return ret

        return f

    pm.Model = lambda: _FakeModel()  # type: ignore[assignment]
    pm.HalfCauchy = _record("HalfCauchy", 1.0)
    pm.Exponential = _record("Exponential", 1.0)
    pm.Normal = _record("Normal", 0.0)
    pm.Uniform = _record("Uniform", 1.0)
    pm.StudentT = _record("StudentT", 0.0)
    pm.GaussianRandomWalk = _record("GaussianRandomWalk", 0.0)
    pm.Deterministic = _record("Deterministic", 0.0)
    pm.math = types.SimpleNamespace(exp=np.exp)

    def sample(draws, progressbar=True, return_inferencedata=False):
        calls.append(("sample", (draws,), {"progressbar": progressbar, "return_inferencedata": return_inferencedata}))
        return {"draws": int(draws)}

    pm.sample = sample  # type: ignore[assignment]

    monkeypatch.setitem(sys.modules, "pymc", pm)
    return calls


def test_bayesian_models_run_with_stubbed_pymc(monkeypatch):
    calls = _install_fake_pymc(monkeypatch)

    from fincore.metrics import bayesian as b

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    data = pd.Series([0.01, -0.02, 0.0, 0.01, -0.01], index=idx)
    bmark = pd.Series([0.0, 0.01, -0.01, 0.02], index=idx[1:])  # misaligned to cover .align()

    model, trace = b.model_returns_t_alpha_beta(data, bmark, samples=3, progressbar=False)
    assert trace["draws"] == 3

    model, trace = b.model_returns_normal(data, samples=2, progressbar=False)
    assert trace["draws"] == 2

    model, trace = b.model_returns_t(data, samples=2, progressbar=False)
    assert trace["draws"] == 2

    y1 = np.array([0.01, np.nan, 0.02, -0.01])
    y2 = np.array([0.00, 0.01, np.nan, 0.02])
    model, trace = b.model_best(y1, y2, samples=4, progressbar=False)
    assert trace["draws"] == 4

    model, trace = b.model_stoch_vol(data, samples=5, progressbar=False)
    assert trace["draws"] == 5

    # Sanity check: stub got used.
    assert any(name == "sample" for name, _, _ in calls)


def test_run_model_branches_with_stubbed_pymc(monkeypatch):
    _install_fake_pymc(monkeypatch)

    from fincore.metrics.bayesian import run_model

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    returns = pd.Series([0.01, -0.02, 0.0, 0.01, -0.01], index=idx)
    bmark = pd.Series([0.0, 0.01, -0.01, 0.02, 0.0], index=idx)

    _, trace = run_model("alpha_beta", returns, bmark=bmark, samples=2, progressbar=False)
    assert trace["draws"] == 2

    _, trace = run_model("t", returns, samples=2, progressbar=False)
    assert trace["draws"] == 2

    _, trace = run_model("normal", returns, samples=2, progressbar=False)
    assert trace["draws"] == 2

    _, trace = run_model("best", returns, bmark=bmark, samples=2, progressbar=False)
    assert trace["draws"] == 2
