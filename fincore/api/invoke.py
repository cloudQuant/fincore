"""Unified enhanced invocation pipeline.

``bind -> profile contract -> normalize/validate -> raw kernel -> result
contract -> projection -> metadata``.  Strict compatibility profiles do not
route through this pipeline; they call their frozen kernels directly.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fincore.api.catalog import OperationCatalog

from fincore.exceptions import InputContractError, ResultContractError
from fincore.results import STATUS_FAILED, STATUS_SUCCESS, AnalysisResult, ResultMetadata

__all__ = ["InvocationResult", "invoke", "resolve_kernel"]


def resolve_kernel(kernel_ref: str) -> Any:
    """Resolve a ``module:qualname`` kernel reference without importing extras."""
    if not kernel_ref:
        raise InputContractError("operation has no kernel_ref", operation_id="<kernel>")
    module_name, _, qualname = kernel_ref.partition(":")
    module = importlib.import_module(module_name)
    target: Any = module
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


def invoke(
    catalog: OperationCatalog,
    operation_id: str,
    profile: str,
    *args: Any,
    **kwargs: Any,
) -> AnalysisResult[Any]:
    """Execute one enhanced operation through the unified pipeline.

    Raises
    ------
    InputContractError
        When the operation is unknown or its kernel raises a contract error.
    """
    definition = catalog.resolve_definition(operation_id, profile)
    kernel = resolve_kernel(definition.kernel_ref)
    metadata = ResultMetadata(
        operation=operation_id,
        profile=profile,
        schema_version="1.0",
        status=STATUS_SUCCESS,
    )
    try:
        value = kernel(*args, **kwargs)
    except InputContractError:
        raise
    except Exception as exc:
        metadata = ResultMetadata(**{**metadata.__dict__, "status": STATUS_FAILED})
        raise InputContractError(
            f"kernel {definition.kernel_ref} failed: {exc}",
            operation_id=operation_id,
            profile=profile,
        ) from exc
    return AnalysisResult.success(value, metadata)


class InvocationResult:
    """Lightweight wrapper carrying the projected value and metadata."""

    __slots__ = ("metadata", "operation_id", "profile", "value")

    def __init__(self, value: Any, metadata: ResultMetadata, operation_id: str, profile: str) -> None:
        self.value = value
        self.metadata = metadata
        self.operation_id = operation_id
        self.profile = profile
