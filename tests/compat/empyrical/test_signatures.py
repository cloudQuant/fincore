from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

import fincore.empyrical as ep
from fincore import beta as flat_beta
from fincore import calmar_ratio as flat_calmar_ratio
from fincore.metrics.alpha_beta import beta as metrics_beta
from fincore.metrics.ratios import calmar_ratio as metrics_calmar_ratio

MANIFEST = Path(__file__).parents[1] / "fixtures" / "empyrical-0.6.0-api.json"


def _callables() -> list[dict[str, Any]]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["callables"]


@pytest.mark.parametrize("entry", _callables(), ids=lambda entry: entry["symbol"])
def test_every_legacy_callable_has_frozen_signature(entry: dict[str, Any]) -> None:
    public = getattr(ep, entry["symbol"])
    assert str(inspect.signature(public)) == entry["signature"]


@pytest.mark.parametrize("entry", _callables(), ids=lambda entry: entry["symbol"])
def test_every_legacy_callable_rejects_missing_required_arguments(entry: dict[str, Any]) -> None:
    public = getattr(ep, entry["symbol"])
    with pytest.raises(TypeError):
        public()


@pytest.mark.parametrize("entry", _callables(), ids=lambda entry: entry["symbol"])
def test_every_legacy_callable_enforces_positional_arity_at_call_time(entry: dict[str, Any]) -> None:
    public = getattr(ep, entry["symbol"])
    signature = inspect.signature(public)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
    ]
    with pytest.raises(TypeError):
        public(*([None] * (len(positional) + 1)))


@pytest.mark.parametrize(
    "entry",
    [entry for entry in _callables() if "**kwargs" not in entry["signature"]],
    ids=lambda entry: entry["symbol"],
)
def test_non_variadic_legacy_callables_reject_unexpected_keyword(entry: dict[str, Any]) -> None:
    public = getattr(ep, entry["symbol"])
    with pytest.raises(TypeError):
        public(__unexpected_legacy_keyword=True)


def test_key_legacy_positional_contracts_are_frozen() -> None:
    assert str(inspect.signature(ep.calmar_ratio)) == "(returns, period='daily', annualization=None)"
    assert str(inspect.signature(ep.beta)) == "(returns, factor_returns, risk_free=0.0, out=None)"


def test_enhanced_flat_and_metrics_signatures_do_not_drift() -> None:
    expected_calmar = "(returns: 'pd.Series | pd.DataFrame | np.ndarray', risk_free: 'float' = 0, period: 'str' = 'daily', annualization: 'float | None' = None) -> 'float | pd.Series'"
    expected_beta = "(returns: 'ReturnOrDataFrame', factor_returns: 'ReturnOrDataFrame', risk_free: 'float' = 0.0, _period: 'str' = 'daily', _annualization: 'float | None' = None, out: 'np.ndarray | None' = None, *, alignment: 'AlignmentPolicy' = 'inner', normalize_tz: 'str | None' = None) -> 'float | np.ndarray | pd.Series'"
    assert str(inspect.signature(metrics_calmar_ratio)) == expected_calmar
    assert str(inspect.signature(flat_calmar_ratio)) == expected_calmar
    assert str(inspect.signature(metrics_beta)) == expected_beta
    assert str(inspect.signature(flat_beta)) == expected_beta
