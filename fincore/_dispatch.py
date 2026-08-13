"""Lazy, registry-selected dispatch for enhanced metric surfaces."""

from __future__ import annotations

import functools
import importlib
import inspect
import sys
import types
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import numpy as np

from fincore._registry import MetricSpec, Surface, get_metric_spec

__all__ = [
    "context_alpha_adapter",
    "context_beta_adapter",
    "enhanced_identity_adapter",
    "install_metric_module_surface",
    "invoke_metric",
    "invoke_prevalidated_projections",
    "metric_callable",
    "project_precomputed",
    "resolve_raw_metric",
]

_RAW_KERNEL_DEPTH: ContextVar[int] = ContextVar("fincore_raw_kernel_depth", default=0)


@contextmanager
def _raw_kernel_execution():
    token = _RAW_KERNEL_DEPTH.set(_RAW_KERNEL_DEPTH.get() + 1)
    try:
        yield
    finally:
        _RAW_KERNEL_DEPTH.reset(token)


def enhanced_identity_adapter(kernel: Any, arguments: dict[str, Any]) -> Any:
    """Lightweight enhanced adapter; expands a bound ``**kwargs`` bucket."""

    call_arguments = dict(arguments)
    call_arguments.update(call_arguments.pop("kwargs", {}))
    return kernel(**call_arguments)


def context_alpha_adapter(kernel: Any, arguments: dict[str, Any]) -> Any:
    """Project alpha from the shared ``alpha_beta`` context kernel."""

    return enhanced_identity_adapter(kernel, arguments)[0]


def context_beta_adapter(kernel: Any, arguments: dict[str, Any]) -> Any:
    """Project beta from the shared ``alpha_beta`` context kernel."""

    return enhanced_identity_adapter(kernel, arguments)[1]


def _resolve(reference: str) -> Any:
    module_name, attribute = reference.split(":", 1)
    with _raw_kernel_execution():
        module = importlib.import_module(module_name)
    return vars(module)[attribute]


def resolve_raw_metric(reference: str) -> Any:
    """Resolve an undecorated metric for composition inside another kernel."""

    return _resolve(reference)


def _check_contract(spec: MetricSpec, signature: inspect.Signature) -> None:
    accepts_out = "out" in signature.parameters
    if spec.out_policy == "write_and_return" and not accepts_out:
        raise ValueError(
            f"metric {spec.surface}:{spec.public_name} out policy {spec.out_policy!r} requires a kernel out parameter"
        )
    if spec.out_policy != "write_and_return" and accepts_out and spec.surface != "context":
        raise ValueError(
            f"metric {spec.surface}:{spec.public_name} hides the kernel out parameter "
            f"without a context lifecycle boundary"
        )


def _apply_projection(spec: MetricSpec, result: Any) -> Any:
    projection = spec.result_projection
    if projection in {"identity", "out_buffer"}:
        return result
    if projection == "scalar":
        if not np.isscalar(result):
            raise TypeError(f"metric {spec.surface}:{spec.public_name} must project a scalar result")
        return result
    if projection == "series":
        import pandas as pd

        if not isinstance(result, pd.Series):
            raise TypeError(f"metric {spec.surface}:{spec.public_name} must project a Series result")
        return result
    if projection == "frame":
        import pandas as pd

        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"metric {spec.surface}:{spec.public_name} must project a DataFrame result")
        return result
    if projection == "legacy_tuple":
        if not isinstance(result, tuple):
            raise TypeError(f"metric {spec.surface}:{spec.public_name} must project a tuple result")
        return result
    raise ValueError(f"unknown result projection: {projection!r}")


def project_precomputed(spec: MetricSpec, result: Any) -> Any:
    """Apply a spec's adapter/projection to one already-computed kernel result."""

    adapter = _resolve(spec.adapter_ref)

    def precomputed_kernel(**_arguments: Any) -> Any:
        return result

    return _apply_projection(spec, adapter(precomputed_kernel, {}))


def _validated_arguments(spec: MetricSpec, signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]):
    bound = signature.bind(*args, **kwargs)
    from fincore.contracts.validation import validate_metric_arguments

    checked = validate_metric_arguments(spec.validation_profile, bound.arguments)
    for name, value in checked.items():
        bound.arguments[name] = value
    return bound


def invoke_metric(
    surface: Surface,
    public_name: str,
    variant: str,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Resolve one exact enhanced surface contract and invoke its kernel."""

    spec = get_metric_spec(surface, public_name, variant)
    if spec.validation_profile == "legacy_empyrical":
        raise ValueError("strict compatibility metrics must use the frozen facade")
    kernel = _resolve(spec.kernel_ref)
    signature = inspect.signature(kernel)
    _check_contract(spec, signature)
    bound = _validated_arguments(spec, signature, args, kwargs)
    adapter = _resolve(spec.adapter_ref)
    with _raw_kernel_execution():
        result = adapter(kernel, bound.arguments)
    return _apply_projection(spec, result)


def invoke_prevalidated_metric(
    surface: Surface,
    public_name: str,
    variant: str,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Invoke an exact entry for data already checked at a lifecycle boundary."""

    spec = get_metric_spec(surface, public_name, variant)
    if spec.validation_profile not in {"enhanced", "context"}:
        raise ValueError("prevalidated dispatch is unavailable for strict compatibility metrics")
    kernel = _resolve(spec.kernel_ref)
    signature = inspect.signature(kernel)
    _check_contract(spec, signature)
    bound = signature.bind(*args, **kwargs)
    adapter = _resolve(spec.adapter_ref)
    with _raw_kernel_execution():
        result = adapter(kernel, bound.arguments)
    return _apply_projection(spec, result)


def invoke_prevalidated_projections(
    surface: Surface,
    public_names: tuple[str, ...],
    variant: str,
    /,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute one shared kernel once and apply multiple registry projections."""

    if not public_names:
        return {}
    specs = [get_metric_spec(surface, public_name, variant) for public_name in public_names]
    first = specs[0]
    if any(spec.kernel_ref != first.kernel_ref for spec in specs[1:]):
        raise ValueError("prevalidated projections must share one kernel")
    if any(spec.validation_profile not in {"enhanced", "context"} for spec in specs):
        raise ValueError("prevalidated projections are unavailable for strict compatibility metrics")
    kernel = _resolve(first.kernel_ref)
    signature = inspect.signature(kernel)
    for spec in specs:
        _check_contract(spec, signature)
    bound = signature.bind(*args, **kwargs)
    with _raw_kernel_execution():
        result = kernel(*bound.args, **bound.kwargs)
    return {spec.public_name: project_precomputed(spec, result) for spec in specs}


def metric_callable(surface: Surface, public_name: str, variant: str):
    """Return a signature-preserving callable for an exact registry key."""

    spec = get_metric_spec(surface, public_name, variant)
    kernel = _resolve(spec.kernel_ref)
    adapter = _resolve(spec.adapter_ref)
    return _metric_callable_for_resolved_contract(spec, kernel, adapter)


class _LazyMetricCallable:
    """Preserve the public signature while resolving a fresh kernel per call.

    ``importlib.reload`` replaces kernel function objects inside the metric
    modules.  Wrappers that closed over the first resolved function would pin
    cached facades — flat API entries and instance-bound methods included —
    onto stale kernels.  Resolving through the registry reference on each use
    keeps every cached wrapper consistent with the code the reload left in
    place, and ``__wrapped__`` exposes the kernel that is current right now.
    """

    __name__: str
    __qualname__: str
    __module__: str
    __doc__: str | None
    __annotations__: dict[str, Any]
    __signature__: inspect.Signature
    __fincore_dispatch_spec__: tuple[Surface, str, str]

    def __init__(self, spec: MetricSpec, kernel: Any, adapter: Any, signature: inspect.Signature) -> None:
        self._spec = spec
        self._adapter = adapter
        self._kernel = kernel
        self._signature = signature
        self.__name__ = spec.public_name
        self.__qualname__ = spec.public_name
        self.__module__ = kernel.__module__
        self.__doc__ = getattr(kernel, "__doc__", None)
        self.__annotations__ = dict(getattr(kernel, "__annotations__", {}))
        self.__signature__ = signature
        self.__fincore_dispatch_spec__ = (spec.surface, spec.public_name, spec.variant)

    @property
    def __wrapped__(self) -> Any:
        """The kernel object currently reachable through the registry."""

        return _resolve(self._spec.kernel_ref)

    def _current_contract(self) -> tuple[Any, inspect.Signature]:
        kernel = _resolve(self._spec.kernel_ref)
        if kernel is self._kernel:
            return kernel, self._signature
        self._kernel = kernel
        self._signature = inspect.signature(kernel)
        return kernel, self._signature

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if _RAW_KERNEL_DEPTH.get() > 0:
            return _resolve(self._spec.kernel_ref)(*args, **kwargs)
        kernel, signature = self._current_contract()
        bound = _validated_arguments(self._spec, signature, args, kwargs)
        with _raw_kernel_execution():
            result = self._adapter(kernel, bound.arguments)
        return _apply_projection(self._spec, result)


@functools.cache
def _metric_callable_for_resolved_contract(spec: MetricSpec, kernel: Any, adapter: Any):
    """Cache a wrapper by the resolved callables, including reload identity."""

    signature = inspect.signature(kernel)
    _check_contract(spec, signature)
    return _LazyMetricCallable(spec, kernel, adapter, signature)


class _ValidatedMetricModule(types.ModuleType):
    """Expose enhanced wrappers externally while kernels call raw globals."""

    def __getattribute__(self, name: str) -> Any:
        namespace = types.ModuleType.__getattribute__(self, "__dict__")
        originals = namespace.get("_fincore_metric_originals", {})
        original = originals.get(name)
        enhanced = namespace.get("_fincore_metric_enhanced", {})
        if (
            original is not None
            and namespace.get(name) is original
            and name in enhanced
            and _RAW_KERNEL_DEPTH.get() == 0
        ):
            return enhanced[name]
        return types.ModuleType.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        namespace = types.ModuleType.__getattribute__(self, "__dict__")
        enhanced = namespace.get("_fincore_metric_enhanced", {})
        originals = namespace.get("_fincore_metric_originals", {})
        # pytest monkeypatch (and similar tools) records the externally visible
        # wrapper, then restores it with setattr.  Store the raw function again
        # so a temporary patch cannot permanently turn an internal kernel into
        # an enhanced wrapper.
        if name in enhanced and value is enhanced[name]:
            value = originals[name]
        types.ModuleType.__setattr__(self, name, value)


def install_metric_module_surface(module_name: str) -> None:
    """Install external enhanced dispatch without changing kernel globals."""

    module = sys.modules[module_name]
    originals = {
        name: vars(module)[spec.kernel_ref.split(":", 1)[1]]
        for (surface, name, variant), spec in __import__(
            "fincore._registry", fromlist=["METRIC_REGISTRY"]
        ).METRIC_REGISTRY.items()
        if surface == "metrics"
        and variant == "enhanced"
        and spec.kernel_ref.split(":", 1)[0] == module_name
        and spec.kernel_ref.split(":", 1)[1] in vars(module)
    }
    if not originals:
        return
    for name, original in originals.items():
        # ``importlib.reload`` executes into the existing module dictionary and
        # does not remove aliases materialized by a previous installation.
        # Refresh every raw slot so an alias such as ``cagr`` cannot retain an
        # older unwrapped function and escape the enhanced module boundary.
        vars(module)[name] = original
    vars(module)["_fincore_metric_originals"] = originals
    vars(module)["_fincore_metric_enhanced"] = {
        name: metric_callable("metrics", name, "enhanced") for name in originals
    }
    exported = vars(module).get("__all__")
    if isinstance(exported, list):
        exported.extend(name for name in originals if name not in exported)
    module.__class__ = _ValidatedMetricModule
