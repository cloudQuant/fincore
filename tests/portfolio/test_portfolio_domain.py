"""Direct-domain contracts for the canonical portfolio package."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from fincore.runtime import OperationCatalog, run


def _positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cash": [20.0, 10.0],
            "long": [80.0, 120.0],
            "short": [-20.0, -30.0],
        },
        index=pd.date_range("2024-01-02", periods=2, tz="UTC"),
    )


def test_portfolio_inputs_copy_on_ingest_and_materialize_independently() -> None:
    from fincore.portfolio.models import PortfolioInputs

    positions = _positions()
    inputs = PortfolioInputs(positions=positions)
    positions.loc[positions.index[0], "long"] = 999.0

    first = inputs.materialize()
    first["positions"].loc[first["positions"].index[0], "long"] = -1.0
    second = inputs.materialize()

    assert second["positions"].loc[second["positions"].index[0], "long"] == 80.0


def test_portfolio_operations_are_direct_domain_callables_and_runtime_reuses_them() -> None:
    from fincore.portfolio.operations import operations
    from fincore.portfolio.positions import get_long_short_pos

    catalog = OperationCatalog(operations())
    operation_id = "portfolio.positions.get_long_short_pos"

    assert operation_id in catalog.operation_ids
    assert catalog.resolve(operation_id).callable is get_long_short_pos
    assert catalog.resolve(operation_id).callable.__module__ == "fincore.portfolio.positions"

    positions = _positions()
    expected = get_long_short_pos(positions)
    result = run(operation_id, {"positions": positions}, catalog=catalog)

    pd.testing.assert_frame_equal(result.value, expected)
    assert result.metadata["implementation_fingerprint"] == "fincore.portfolio.positions:get_long_short_pos"


def test_portfolio_domain_does_not_depend_on_facades_or_legacy_dispatch() -> None:
    package_root = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).parents[2])).resolve() / "fincore" / "portfolio"
    forbidden = ("fincore._registry", "fincore._dispatch", "fincore.empyrical", "fincore.pyfolio", "fincore.alphalens")

    violations = {
        path.name: token
        for path in package_root.glob("*.py")
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    }

    assert violations == {}
