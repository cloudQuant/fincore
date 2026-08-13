"""Implementation helpers for the import-safe strict Alphalens facade."""

from __future__ import annotations

from typing import Any

from fincore.contracts.factor_analysis import FactorFunctionSpec, function_specs_for_module


def _deferred_message(spec: FactorFunctionSpec) -> str:
    return (
        f"Legacy Alphalens symbol '{spec.public_name}' is available for C0/C1 compatibility, "
        "but its numerical or rendering kernel is not implemented yet."
    )


def make_deferred_callable(spec: FactorFunctionSpec) -> Any:
    """Create a signature-preserving facade that binds before its deferred boundary."""

    def deferred(*args: Any, **kwargs: Any) -> Any:
        source_kwargs = dict(kwargs)
        if spec.adapter == "plotting.customize":
            source_kwargs.pop("set_context", None)
        spec.source_signature.bind(*args, **source_kwargs)
        raise NotImplementedError(_deferred_message(spec))

    wrapped: Any = deferred
    wrapped.__name__ = spec.public_name
    wrapped.__qualname__ = spec.public_name
    wrapped.__module__ = f"fincore.alphalens.{spec.module}"
    wrapped.__doc__ = _deferred_message(spec)
    wrapped.__signature__ = spec.introspection_signature
    wrapped.__fincore_source_signature__ = spec.source_signature
    wrapped.__fincore_factor_spec__ = spec
    return wrapped


def export_deferred_functions(namespace: dict[str, Any], module: str) -> tuple[str, ...]:
    """Install the pinned function names for one facade module without optional imports."""

    names: list[str] = []
    for spec in function_specs_for_module(module):
        namespace[spec.public_name] = make_deferred_callable(spec)
        names.append(spec.public_name)
    return tuple(names)


__all__ = ["export_deferred_functions", "make_deferred_callable"]
