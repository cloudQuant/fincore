"""C0 public-surface checks against the pinned Alphalens manifest."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import FrozenInstanceError
from typing import Any

import pandas as pd
import pytest

from .conftest import manifest_entries


def _entry_id(entry: dict[str, Any]) -> str:
    return f"{entry['module']}:{entry['symbol']}"


@pytest.mark.parametrize("entry", manifest_entries(), ids=_entry_id)
def test_frozen_definition_resolves(entry: dict[str, Any]) -> None:
    """Every C0 manifest definition resolves from its declared module."""

    module = importlib.import_module(f"fincore.alphalens.{entry['module']}")
    value = getattr(module, str(entry["symbol"]))
    assert callable(value)


def test_manifest_definition_count_resolves_without_flattening_root_api() -> None:
    """The 64 frozen definitions remain module-scoped rather than root aliases."""

    import fincore
    from fincore import alphalens

    manifest = manifest_entries()
    assert len(manifest) == 64
    assert sum(entry["kind"] == "function" for entry in manifest) == 61
    assert sum(entry["kind"] == "class" for entry in manifest) == 3
    assert fincore.alphalens is alphalens
    assert "alphalens" in fincore.__all__
    for module_name in ("performance", "plotting", "tears", "utils"):
        assert getattr(alphalens, module_name) is importlib.import_module(f"fincore.alphalens.{module_name}")
    assert not hasattr(fincore, "quantize_factor")
    assert not hasattr(fincore, "plot_ic_ts")


def test_facade_modules_export_only_their_pinned_public_definitions() -> None:
    """Implementation helpers do not become accidental compatibility names."""

    for module_name in ("performance", "plotting", "tears", "utils"):
        module = importlib.import_module(f"fincore.alphalens.{module_name}")
        expected_names = tuple(entry["symbol"] for entry in manifest_entries() if entry["module"] == module_name)
        assert tuple(module.__all__) == expected_names
        assert not hasattr(module, "export_deferred_functions")


def test_static_contract_registry_keeps_source_and_introspection_facts_separate() -> None:
    """The facade registry is checked-in code, not a runtime fixture reader."""

    from fincore.contracts.factor_analysis import ALPHALENS_FUNCTION_SPECS, FactorFunctionSpec
    from fincore.contracts.factor_workflows import ALPHALENS_WORKFLOW_SPECS, FactorWorkflowSpec

    assert len(ALPHALENS_FUNCTION_SPECS) == 61
    assert len(ALPHALENS_WORKFLOW_SPECS) == 7
    quantize = ALPHALENS_FUNCTION_SPECS[("utils", "quantize_factor")]
    assert isinstance(quantize, FactorFunctionSpec)
    assert str(quantize.introspection_signature) == "(*args, **kwargs)"
    assert str(quantize.source_signature) == (
        "(factor_data, quantiles=5, bins=None, by_group=False, no_raise=False, zero_aware=False)"
    )
    assert all(isinstance(spec, FactorWorkflowSpec) for spec in ALPHALENS_WORKFLOW_SPECS.values())
    with pytest.raises(FrozenInstanceError):
        quantize.module = "changed"  # type: ignore[misc]


def test_grid_figure_and_legacy_exceptions_resolve() -> None:
    """The non-function C0 names have stable, import-safe definitions."""

    from fincore.alphalens.tears import GridFigure
    from fincore.alphalens.utils import MaxLossExceededError, NonMatchingTimezoneError

    assert str(inspect.signature(GridFigure)) == "(rows, cols)"
    assert isinstance(MaxLossExceededError(), MaxLossExceededError)
    assert isinstance(NonMatchingTimezoneError(), NonMatchingTimezoneError)


def test_shared_synthetic_fixture_contract(
    raw_factor: pd.Series,
    prices: pd.DataFrame,
    tz_aware_prices: pd.DataFrame,
    groups: pd.Series,
) -> None:
    """The future Tasks 3–8 fixture inputs stay deterministic and non-overlapping."""

    assert raw_factor.name == "factor"
    assert raw_factor.index.names == ["date", "asset"]
    assert len(raw_factor) == 120 * 10
    assert raw_factor.index.get_level_values("date").unique().equals(pd.bdate_range("2024-01-02", periods=120))
    assert prices.shape == (120, 12)
    assert prices.index.tz is None
    assert tz_aware_prices.shape == prices.shape
    assert str(tz_aware_prices.index.tz) == "UTC"
    assert set(raw_factor.index.get_level_values("asset")) < set(prices.columns)
    assert groups.index.name == "asset"
    assert groups.to_dict() == {
        f"asset_{ordinal:02d}": "sector_a" if ordinal % 2 == 0 else "sector_b" for ordinal in range(10)
    }


def test_clean_factor_data_fixture_is_an_explicit_task_3_boundary(request: pytest.FixtureRequest) -> None:
    """Task 2 does not pretend to have a cleaned factor-data implementation."""

    with pytest.raises(RuntimeError, match="deferred until Task 3"):
        request.getfixturevalue("clean_factor_data")
