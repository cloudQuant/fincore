"""Canonical namespace contracts for the breaking 0.5 public surface."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


_EMPTY_CANONICAL_NAMESPACES = (
    "fincore.attribution",
    "fincore.data",
    "fincore.extensions",
    "fincore.factor_analysis",
    "fincore.metrics",
    "fincore.optimization",
    "fincore.performance",
    "fincore.portfolio",
    "fincore.report",
    "fincore.report.factor",
    "fincore.report.portfolio",
    "fincore.report.renderers",
    "fincore.risk",
)


def test_canonical_namespaces_load_without_package_root_compatibility_exports() -> None:
    """Domain roots remain importable but keep leaf APIs at their owner paths."""

    namespaces = {
        name: importlib.reload(importlib.import_module(name))
        for name in _EMPTY_CANONICAL_NAMESPACES
    }
    runtime = importlib.reload(importlib.import_module("fincore.runtime"))
    root = importlib.reload(importlib.import_module("fincore"))

    assert root.__all__ == [
        "__version__",
        "attribution",
        "data",
        "errors",
        "extensions",
        "factor_analysis",
        "metrics",
        "optimization",
        "performance",
        "portfolio",
        "report",
        "risk",
        "runtime",
        "simulation",
        "viz",
    ]
    assert all(module.__all__ == [] for module in namespaces.values())
    assert {"OperationCatalog", "OperationSpec", "run"}.issubset(runtime.__all__)
    assert not hasattr(root, "empyrical")
    assert not hasattr(root, "pyfolio")
    assert not hasattr(root, "alphalens")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_bar_consumption": 0.0}, "max_bar_consumption"),
        ({"capital_base": 0.0}, "capital_base"),
        ({"mean_volume_window": 0}, "mean_volume_window"),
        ({"last_n_days": 0}, "last_n_days"),
    ],
)
def test_capacity_configuration_rejects_invalid_explicit_assumptions(
    kwargs: dict[str, float | int], message: str
) -> None:
    from fincore.portfolio.capacity import CapacityConfig

    with pytest.raises(ValueError, match=message):
        CapacityConfig(**kwargs)


def test_capacity_assessment_delegates_to_each_direct_liquidity_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fincore.portfolio import capacity

    index = pd.date_range("2025-01-02", periods=2, freq="B")
    positions = pd.DataFrame({"asset": [100.0, 110.0]}, index=index)
    transactions = pd.DataFrame({"amount": [1.0, -1.0]}, index=index)
    market_data = {"asset": pd.DataFrame({"volume": [10.0, 20.0]}, index=index)}
    liquidation = pd.DataFrame({"days": [1.0]}, index=["asset"])
    maximums = pd.DataFrame({"max_days": [1.0]}, index=["asset"])
    low_liquidity = pd.DataFrame({"amount": []})
    calls: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {}

    def record(name: str, result: pd.DataFrame):
        def kernel(*args: object, **kwargs: object) -> pd.DataFrame:
            calls[name] = (args, kwargs)
            return result

        return kernel

    monkeypatch.setattr(capacity, "days_to_liquidate_positions", record("liquidation", liquidation))
    monkeypatch.setattr(capacity, "get_max_days_to_liquidate_by_ticker", record("maximums", maximums))
    monkeypatch.setattr(capacity, "get_low_liquidity_transactions", record("low_liquidity", low_liquidity))

    config = capacity.CapacityConfig(
        max_bar_consumption=0.4,
        capital_base=2_000_000.0,
        mean_volume_window=7,
        last_n_days=3,
    )
    assessment = capacity.assess_liquidity(positions, transactions, market_data, config)

    assert assessment.liquidation_days is liquidation
    assert assessment.ticker_maximums is maximums
    assert assessment.low_liquidity_transactions is low_liquidity
    assert calls["liquidation"] == (
        (positions, market_data),
        {"max_bar_consumption": 0.4, "capital_base": 2_000_000.0, "mean_volume_window": 7},
    )
    assert calls["maximums"] == (
        (positions, market_data),
        {
            "max_bar_consumption": 0.4,
            "capital_base": 2_000_000.0,
            "mean_volume_window": 7,
            "last_n_days": 3,
        },
    )
    assert calls["low_liquidity"] == ((transactions, market_data), {"last_n_days": 3})


def test_shared_kernel_and_runtime_boundaries_reject_invalid_direct_inputs() -> None:
    from fincore.metrics.basic import ensure_datetime_index_series
    from fincore.metrics.frequencies import annualization_factor, pandas_frequency
    from fincore.metrics._rolling import rolling_window
    from fincore.runtime.backends import NumPyBackend
    from fincore.runtime.engine import OperationRequest

    assert rolling_window(np.asarray([1.0, 2.0, 3.0]), 2).tolist() == [[1.0, 2.0], [2.0, 3.0]]
    with pytest.raises(ValueError, match="one-dimensional"):
        rolling_window(np.asarray([[1.0]]), 1)
    with pytest.raises(ValueError, match="positive integer"):
        rolling_window(np.asarray([1.0]), 0)
    with pytest.raises(ValueError, match="cannot exceed"):
        rolling_window(np.asarray([1.0]), 2)
    with pytest.raises(ValueError, match="unknown frequency"):
        annualization_factor("unknown")
    with pytest.raises(ValueError, match="unknown frequency"):
        pandas_frequency("unknown")
    fallback = ensure_datetime_index_series([1.0, 2.0], period="unknown")
    assert isinstance(fallback.index, pd.DatetimeIndex)
    with pytest.raises(ValueError, match="at least 2 values"):
        NumPyBackend().sample_standard_deviation([1.0])
    with pytest.raises(TypeError, match="inputs must be a mapping"):
        OperationRequest("metrics.demo", [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="inputs keys must be strings"):
        OperationRequest("metrics.demo", {1: 1})  # type: ignore[dict-item]


def test_portfolio_models_keep_direct_inputs_and_exposures_shape_safe() -> None:
    from fincore.exceptions import ValidationError
    from fincore.portfolio.models import ExposureBundle, PortfolioInputs, VolumeExposureBundle

    index = pd.date_range("2025-01-02", periods=2, freq="B")
    frame = pd.DataFrame({"asset": [1.0, 2.0]}, index=index)
    with pytest.raises(ValidationError, match="pandas DataFrame"):
        ExposureBundle([], frame, frame, frame)  # type: ignore[arg-type]
    duplicated = pd.DataFrame([[1.0, 2.0]], columns=["asset", "asset"], index=index[:1])
    with pytest.raises(ValidationError, match="columns must be unique"):
        ExposureBundle(duplicated, duplicated, duplicated, duplicated)
    with pytest.raises(ValidationError, match="pandas Series"):
        VolumeExposureBundle([], pd.Series([1.0]), pd.Series([1.0]))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="positions must be a pandas DataFrame"):
        PortfolioInputs(positions=pd.Series([1.0]))  # type: ignore[arg-type]
    mismatched = pd.DataFrame({"other": [1.0, 2.0]}, index=index)
    with pytest.raises(ValidationError, match="same category columns"):
        ExposureBundle(frame, frame, frame, mismatched)
