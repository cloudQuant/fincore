"""Direct-domain contracts for canonical risk operations."""

from __future__ import annotations

import importlib

from fincore.runtime import OperationCatalog


def test_risk_root_is_a_namespace_without_leaf_reexports() -> None:
    module = importlib.import_module("fincore.risk")

    assert module.__all__ == []
    assert "forecast_var" not in module.__dict__
    assert "RiskModelSpec" not in module.__dict__


def test_risk_operations_resolve_to_direct_model_and_backtest_kernels() -> None:
    from fincore.risk.backtesting import backtest_var
    from fincore.risk.models import forecast_var
    from fincore.risk.operations import operations

    catalog = OperationCatalog(operations())

    assert catalog.resolve("risk.models.forecast_var").callable is forecast_var
    assert catalog.resolve("risk.backtesting.backtest_var").callable is backtest_var
    assert catalog.resolve("risk.models.forecast_var").implementation_fingerprint == "fincore.risk.models:forecast_var"
