from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import fincore
from fincore.exceptions import FincoreError, NumericalError


def _constant_adapter(_kernel, arguments):
    return ("adapted", tuple(arguments))


def test_importing_dispatch_does_not_import_metric_implementations() -> None:
    code = """
import json, sys
import fincore._dispatch
print(json.dumps(sorted(name for name in sys.modules if name.startswith('fincore.metrics.'))))
"""
    completed = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == []


def test_importing_and_constructing_context_does_not_import_metric_implementations() -> None:
    code = """
import json, sys
import pandas as pd
from fincore.core.context import AnalysisContext
returns = pd.Series([0.01, -0.01], index=pd.date_range('2024-01-01', periods=2))
AnalysisContext(returns)
print(json.dumps(sorted(name for name in sys.modules if name.startswith('fincore.metrics.'))))
"""
    completed = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == []


@pytest.mark.parametrize(
    "code",
    [
        "import fincore.metrics.ratios as metrics; metrics.sharpe_ratio([0.01, -0.01])",
        "import fincore; fincore.sharpe_ratio([0.01, -0.01])",
    ],
)
def test_first_enhanced_metric_call_does_not_load_empyrical_facade(code: str) -> None:
    probe = f"""
import json, sys
{code}
print(json.dumps('fincore.empyrical' in sys.modules))
"""
    completed = subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) is False


@pytest.mark.parametrize(
    "imports",
    [
        "import fincore.metrics.returns as returns_module\nimport fincore.metrics.drawdown as drawdown",
        "import fincore.metrics.drawdown as drawdown\nimport fincore.metrics.returns as returns_module",
    ],
)
def test_drawdown_composition_uses_raw_cumulative_returns_in_any_import_order(imports: str) -> None:
    probe = f"""
import json
import pandas as pd
from fincore.exceptions import FincoreError
{imports}
values = pd.Series([0.10, -0.10, 0.20], index=pd.Index(['z', 'a', 'm'], dtype=object))
try:
    returns_module.cum_returns(values)
except FincoreError:
    public_rejected = True
else:
    public_rejected = False
print(json.dumps({{
    'public_rejected': public_rejected,
    'drawdowns': drawdown.get_top_drawdowns(values, top=10),
}}))
"""
    completed = subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True, text=True)

    assert json.loads(completed.stdout) == {
        "public_rejected": True,
        "drawdowns": [["z", "a", "m"]],
    }


def test_flat_dispatch_binds_keyword_arguments_before_validation() -> None:
    returns = pd.Series([0.01, np.inf, -0.01], index=pd.date_range("2024-01-01", periods=3))

    with pytest.raises(NumericalError, match="finite"):
        fincore.sharpe_ratio(returns=returns)


def test_exact_flat_registry_key_is_selected(monkeypatch) -> None:
    import fincore._dispatch as dispatch

    seen = []
    original = dispatch.get_metric_spec

    def recording(surface, public_name, variant):
        seen.append((surface, public_name, variant))
        return original(surface, public_name, variant)

    monkeypatch.setattr(dispatch, "get_metric_spec", recording)
    returns = pd.Series([0.01, -0.005, 0.02], index=pd.date_range("2024-01-01", periods=3))

    dispatch.invoke_metric("fincore_flat", "sharpe_ratio", "enhanced-0.3.x", returns)

    assert seen == [("fincore_flat", "sharpe_ratio", "enhanced-0.3.x")]


def test_strict_dispatch_profile_is_not_routable_through_enhanced_validator(monkeypatch) -> None:
    import fincore._dispatch as dispatch
    import fincore.contracts.validation as validation
    import fincore.empyrical as legacy

    def forbidden(*_args, **_kwargs):
        pytest.fail("strict facade reached enhanced validation")

    monkeypatch.setattr(validation, "validate_metric_arguments", forbidden)
    returns = pd.Series([0.01, np.nan, -0.01], index=pd.date_range("2024-01-01", periods=3))

    assert np.isfinite(legacy.sharpe_ratio(returns))


def test_strict_cross_module_kernel_bypasses_a_cached_enhanced_wrapper(monkeypatch) -> None:
    import fincore.contracts.validation as validation
    import fincore.empyrical as legacy
    import fincore.metrics.drawdown as drawdown

    # ``drawdown.max_drawdown`` captured raw ``returns.cum_returns`` at module
    # import time.  The raw-depth guard must therefore live in the cached
    # wrapper itself, not only in module attribute lookup.
    assert drawdown.__dict__["_cum_returns"] is vars(drawdown)["_cum_returns"]
    assert not hasattr(drawdown.__dict__["_cum_returns"], "__fincore_dispatch_spec__")

    def forbidden(*_args, **_kwargs):
        pytest.fail("strict cross-module composition reached enhanced validation")

    monkeypatch.setattr(validation, "validate_metric_arguments", forbidden)
    returns = pd.Series([0.01, np.nan, -0.01], index=pd.date_range("2024-01-01", periods=3))

    assert np.isfinite(legacy.max_drawdown(returns))


def test_raw_kernel_guard_bypasses_only_inside_its_context() -> None:
    import fincore.metrics.ratios as ratios
    from fincore._dispatch import _raw_kernel_execution

    returns = pd.Series([0.01, np.nan, -0.01], index=pd.date_range("2024-01-01", periods=3))

    with pytest.raises(NumericalError, match="finite"):
        ratios.sharpe_ratio(returns)
    with _raw_kernel_execution():
        assert np.isfinite(ratios.sharpe_ratio(returns))
    with pytest.raises(NumericalError, match="finite"):
        ratios.sharpe_ratio(returns)


def test_enhanced_dispatch_executes_the_registered_adapter(monkeypatch) -> None:
    import fincore._dispatch as dispatch

    original = dispatch.get_metric_spec("fincore_flat", "sharpe_ratio", "enhanced-0.3.x")
    adapted = replace(original, adapter_ref=f"{__name__}:_constant_adapter")
    monkeypatch.setattr(dispatch, "get_metric_spec", lambda *_args: adapted)
    returns = pd.Series([0.01, -0.01], index=pd.date_range("2024-01-01", periods=2))

    assert dispatch.invoke_metric("fincore_flat", "sharpe_ratio", "enhanced-0.3.x", returns)[0] == "adapted"


def test_dispatch_rejects_registry_out_contract_drift(monkeypatch) -> None:
    import fincore._dispatch as dispatch

    original = dispatch.get_metric_spec("fincore_flat", "sharpe_ratio", "enhanced-0.3.x")
    monkeypatch.setattr(dispatch, "get_metric_spec", lambda *_args: replace(original, out_policy="unsupported"))
    returns = pd.Series([0.01, -0.01], index=pd.date_range("2024-01-01", periods=2))

    with pytest.raises(ValueError, match="out parameter"):
        dispatch.invoke_metric("fincore_flat", "sharpe_ratio", "enhanced-0.3.x", returns)


def test_dispatch_rejects_registry_result_projection_drift(monkeypatch) -> None:
    import fincore._dispatch as dispatch

    original = dispatch.get_metric_spec("fincore_flat", "sharpe_ratio", "enhanced-0.3.x")
    monkeypatch.setattr(dispatch, "get_metric_spec", lambda *_args: replace(original, result_projection="series"))
    returns = pd.Series([0.01, -0.01], index=pd.date_range("2024-01-01", periods=2))

    with pytest.raises(TypeError, match="project a Series"):
        dispatch.invoke_metric("fincore_flat", "sharpe_ratio", "enhanced-0.3.x", returns)
