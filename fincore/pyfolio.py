"""Strict, lazy-compatible façade for the pinned Pyfolio workflows.

Importing this module resolves only the workflow manifest.  Plotting libraries,
the stateful :class:`Pyfolio` implementation, and optional scientific
dependencies are loaded on explicit class access or first workflow call.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from fincore.contracts.workflows import (
    PYFOLIO_SIGNATURE_MANIFEST,
    WORKFLOW_REGISTRY,
    WorkflowSpec,
    invoke_workflow,
)

if TYPE_CHECKING:
    from fincore._pyfolio_impl import Pyfolio


def _signature_from_text(signature_text: str) -> inspect.Signature:
    namespace: dict[str, Any] = {}
    exec(f"def _workflow{signature_text}:\n    pass", {"__builtins__": {}}, namespace)
    return inspect.signature(namespace["_workflow"])


def _invoke_workflow(name: str, arguments: dict[str, Any]) -> Any:
    spec = WORKFLOW_REGISTRY[("pyfolio_module", name, "strict-0.9.6")]
    return invoke_workflow(spec, arguments)


def _make_strict_wrapper(spec: WorkflowSpec):
    manifest_key = spec.signature_manifest_key
    try:
        manifest_name, signature_text = PYFOLIO_SIGNATURE_MANIFEST[manifest_key]
    except KeyError:
        raise KeyError(f"unknown pyfolio signature manifest key: {manifest_key}") from None
    if manifest_name != spec.public_name:
        raise ValueError(f"signature manifest symbol {manifest_name!r} does not match public name {spec.public_name!r}")
    signature = _signature_from_text(signature_text)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        return _invoke_workflow(spec.public_name, dict(bound.arguments))

    wrapper.__name__ = spec.public_name
    wrapper.__qualname__ = spec.public_name
    wrapper.__module__ = __name__
    wrapper.__doc__ = f"Run the pinned Pyfolio 0.9.6 {spec.public_name} workflow."
    wrapper.__signature__ = signature
    return wrapper


_STRICT_PUBLIC: list[str] = []
for (_surface, _public_name, _variant), _spec in WORKFLOW_REGISTRY.items():
    if _surface == "pyfolio_module" and _variant == "strict-0.9.6":
        globals()[_public_name] = _make_strict_wrapper(_spec)
        _STRICT_PUBLIC.append(_public_name)


def __getattr__(name: str) -> Any:
    if name == "Pyfolio":
        from fincore._pyfolio_impl import Pyfolio

        globals()["Pyfolio"] = Pyfolio
        return Pyfolio
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), "Pyfolio"})


__all__ = [*_STRICT_PUBLIC, "Pyfolio"]
