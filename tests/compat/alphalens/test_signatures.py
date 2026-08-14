"""C1 signature and accepted-call grammar checks for the strict facade."""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest

from .conftest import accepted_call_cases, callable_entries_with_signature


def _entry_id(entry: dict[str, Any]) -> str:
    return f"{entry['module']}:{entry['symbol']}"


def _case_id(item: tuple[dict[str, Any], dict[str, Any]]) -> str:
    entry, case = item
    return f"{entry['module']}:{entry['symbol']}:{case['case_id']}"


def _required_arguments(entry: dict[str, Any]) -> tuple[object, ...]:
    """Supply harmless opaque values for source-visible required parameters."""

    required = [parameter for parameter in entry["parameters"] if parameter["required"] and parameter["name"] != "self"]
    return tuple(object() for _ in required)


@pytest.mark.parametrize("entry", callable_entries_with_signature(), ids=_entry_id)
def test_signature_matches_manifest(entry: dict[str, Any]) -> None:
    """Public inspection follows the frozen legacy signature projection."""

    module = importlib.import_module(f"fincore.alphalens.{entry['module']}")
    value = getattr(module, str(entry["symbol"]))
    expected = entry["introspection_signature"] or entry["source_signature"]
    assert str(inspect.signature(value)) == expected


@pytest.mark.parametrize("item", accepted_call_cases(), ids=_case_id)
def test_decorator_call_grammar_matches_manifest(item: tuple[dict[str, Any], dict[str, Any]]) -> None:
    """Accepted legacy binding reaches the explicit deferred implementation boundary."""

    entry, case = item
    module = importlib.import_module(f"fincore.alphalens.{entry['module']}")
    value = getattr(module, str(entry["symbol"]))
    args = _required_arguments(entry)
    hidden_kwargs = dict(case["hidden_kwargs"])

    if entry["kind"] == "class" and entry["symbol"] in {"MaxLossExceededError", "NonMatchingTimezoneError"}:
        assert isinstance(value(*args, **hidden_kwargs), value)
        return

    if entry["module"] == "plotting" and entry["symbol"] in {"axes_style", "plotting_context"}:
        context = value(*args, **hidden_kwargs)
        assert hasattr(context, "__enter__")
        assert hasattr(context, "__exit__")
        return

    with pytest.raises(NotImplementedError, match=str(entry["symbol"])):
        value(*args, **hidden_kwargs)


@pytest.mark.parametrize("entry", callable_entries_with_signature(), ids=_entry_id)
def test_source_signature_rejects_unexpected_keyword(entry: dict[str, Any]) -> None:
    """Deferred kernels still reject calls before reporting deferred execution."""

    if entry["kind"] == "class":
        return
    module = importlib.import_module(f"fincore.alphalens.{entry['module']}")
    value = getattr(module, str(entry["symbol"]))
    with pytest.raises(TypeError):
        value(*_required_arguments(entry), unexpected_legacy_keyword=object())


def test_customize_hidden_context_precedes_source_binding() -> None:
    """Tear-sheet wrappers accept hidden context without exposing it to inspection."""

    from fincore.alphalens.tears import create_summary_tear_sheet

    assert "set_context" not in str(inspect.signature(create_summary_tear_sheet))
    for set_context in (True, False):
        with pytest.raises(NotImplementedError, match="create_summary_tear_sheet"):
            create_summary_tear_sheet(object(), set_context=set_context)


def test_quantize_factor_reports_generic_legacy_signature_but_binds_source_grammar() -> None:
    """The decorator projection does not turn quantize_factor into a permissive stub."""

    from fincore.alphalens.utils import quantize_factor

    assert str(inspect.signature(quantize_factor)) == "(*args, **kwargs)"
    with pytest.raises(TypeError):
        quantize_factor()
    with pytest.raises(NotImplementedError, match="quantize_factor"):
        quantize_factor(object(), quantiles=5)
