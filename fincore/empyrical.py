#
# Copyright 2016 Quantopian, Inc.
# Copyright 2025 CloudQuant Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Empyrical: a financial performance analytics library.

This module includes the original empyrical-style functions plus an object-oriented ``Empyrical`` class.

The codebase has been refactored:

* Implementations live in ``fincore.metrics`` submodules.
* ``@classmethod`` facade methods are auto-generated from ``fincore._registry`` via the ``_LazyMethod`` descriptor
  plus the ``@_populate_from_registry`` class decorator, avoiding 100+ lines of manual delegation.
* ``@_dual_method`` facade methods (which auto-fill ``returns`` / ``factor_returns`` from an instance) remain
  explicitly defined to preserve precise call signatures.
"""

from __future__ import annotations

import functools
import importlib
import inspect
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from fincore._registry import (
    CLASSMETHOD_REGISTRY,
    EMPYRICAL_SIGNATURE_MANIFEST,
    METRIC_REGISTRY,
    STATIC_METHODS,
    MetricSpec,
    _resolve_module,
)

if TYPE_CHECKING:
    from fincore.contracts.time_series import AlignmentPolicy

DAILY = "daily"
WEEKLY = "weekly"
MONTHLY = "monthly"
QUARTERLY = "quarterly"
YEARLY = "yearly"


class _dual_method:
    """Descriptor that allows a method to work both as a class-level call and instance call.

    When accessed on the class (Empyrical.method), behaves like a classmethod -
    passes the class as the first argument.
    When accessed on an instance (emp.method), passes the instance as the first argument,
    allowing access to instance attributes like self.returns.
    """

    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            attr_name = "_cls_bound_" + self.__name__
            try:
                return objtype.__dict__[attr_name]
            except KeyError:

                @functools.wraps(self.func)
                def wrapper(*args, **kwargs):
                    return self.func(objtype, *args, **kwargs)

                setattr(objtype, attr_name, wrapper)
                return wrapper
        else:
            attr_name = "_bound_" + self.__name__
            try:
                return obj.__dict__[attr_name]
            except KeyError:

                @functools.wraps(self.func)
                def wrapper(*args, **kwargs):
                    return self.func(obj, *args, **kwargs)

                obj.__dict__[attr_name] = wrapper
                return wrapper


def _resolve_ref(reference: str):
    """Resolve a lazy ``module:attribute`` registry reference."""

    module_name, attribute = reference.split(":", 1)
    return vars(importlib.import_module(module_name))[attribute]


# ---------------------------------------------------------------------------
# Lazy descriptor + class decorator — replaces metaclass for registry methods
# ---------------------------------------------------------------------------
class _LazyMethod:
    """Non-data descriptor that lazy-resolves a metric function on first access.

    On first attribute access (class-level or instance-level), the underlying
    function is resolved from the metric module, cached as a ``staticmethod``
    on the owner class, and returned.  Subsequent accesses hit the cached
    ``staticmethod`` directly — zero overhead.
    """

    __slots__ = ("_owner", "attr_name", "func_name", "mod_alias")

    def __init__(self, mod_alias, func_name, attr_name, owner):
        self.mod_alias = mod_alias
        self.func_name = func_name
        self.attr_name = attr_name
        self._owner = owner

    def __get__(self, obj, objtype=None):
        func = getattr(_resolve_module(self.mod_alias), self.func_name)
        setattr(self._owner, self.attr_name, staticmethod(func))
        return func


class _MetricMethod:
    """Descriptor for one enhanced class surface with optional state binding."""

    def __init__(self, spec: MetricSpec):
        self.spec = spec
        self.__name__ = spec.public_name

    def __get__(self, obj, objtype=None):
        from fincore._dispatch import metric_callable

        kernel = metric_callable(self.spec.surface, self.spec.public_name, self.spec.variant)
        if obj is None or self.spec.binding == "static":
            return kernel

        cached_name = f"_metric_bound_{self.spec.public_name}"
        cached = obj.__dict__.get(cached_name)
        if cached is not None:
            return cached

        kernel_signature = inspect.signature(kernel)
        state_names = {
            "returns": "returns",
            "arr": "returns",
            "lhs": "returns",
        }
        if self.spec.binding == "returns_factor":
            state_names.update(
                {
                    "factor_returns": "factor_returns",
                    "rhs": "factor_returns",
                    "positions": "positions",
                    "factor_loadings": "factor_loadings",
                }
            )
        state_parameters = {
            name: attribute for name, attribute in state_names.items() if name in kernel_signature.parameters
        }
        public_signature = kernel_signature.replace(
            parameters=[
                parameter
                for parameter in kernel_signature.parameters.values()
                if parameter.name not in state_parameters
            ]
        )

        @functools.wraps(kernel)
        def bound(*args, **kwargs):
            state_available = all(
                getattr(obj, attribute_name, None) is not None for attribute_name in state_parameters.values()
            )
            if not state_available:
                kernel_bound = kernel_signature.bind(*args, **kwargs)
                return kernel(*kernel_bound.args, **kernel_bound.kwargs)
            public_bound = public_signature.bind(*args, **kwargs)
            call_arguments = {}
            for name, value in public_bound.arguments.items():
                parameter = public_signature.parameters[name]
                if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                    call_arguments.update(value)
                else:
                    call_arguments[name] = value
            for parameter_name, attribute_name in state_parameters.items():
                value = getattr(obj, attribute_name, None)
                if value is not None:
                    call_arguments[parameter_name] = value
            kernel_bound = kernel_signature.bind(**call_arguments)
            return kernel(*kernel_bound.args, **kernel_bound.kwargs)

        bound.__signature__ = public_signature
        obj.__dict__[cached_name] = bound
        return bound


def _populate_from_registry(cls):
    """Class decorator: attach ``_LazyMethod`` descriptors for all registry entries."""
    class_names = set()
    for (surface, name, variant), spec in METRIC_REGISTRY.items():
        if surface == "empyrical_class" and variant == "stateful-enhanced":
            setattr(cls, name, _MetricMethod(spec))
            class_names.add(name)
    for name, (mod_alias, func_name) in CLASSMETHOD_REGISTRY.items():
        if name not in class_names:
            setattr(cls, name, _LazyMethod(mod_alias, func_name, name, cls))
    for name, (mod_alias, func_name) in STATIC_METHODS.items():
        setattr(cls, name, _LazyMethod(mod_alias, func_name, name, cls))
    return cls


# ---------------------------------------------------------------------------
# Zipline compatibility stubs
# ---------------------------------------------------------------------------
ZIPLINE = False


class _ZiplineAssetStub:
    price_multiplier = 1


class _EquityStub(_ZiplineAssetStub):
    pass


class _FutureStub(_ZiplineAssetStub):
    pass


_ZIPLINE_WARNING = 'Module "zipline.assets" not found; multipliers will not be applied to position notionals.'

Equity: type[Any] = _EquityStub
Future: type[Any] = _FutureStub

try:
    _zip_assets = importlib.import_module("zipline.assets")
except ModuleNotFoundError:
    pass
else:
    ZIPLINE = True
    Equity = getattr(_zip_assets, "Equity", _EquityStub)
    Future = getattr(_zip_assets, "Future", _FutureStub)


# ---------------------------------------------------------------------------
# Empyrical class
# ---------------------------------------------------------------------------
@_populate_from_registry
class Empyrical:
    """Object-oriented performance metric interface.

    ``Empyrical`` is a thin interface: every metric method is auto-generated
    from ``fincore._registry`` by the ``@_populate_from_registry`` decorator,
    so implementations live in the ``fincore.metrics`` submodules and this
    class stays small.

    * **Class-level access** (e.g. ``Empyrical.sharpe_ratio(returns)``) resolves
      the underlying kernel directly.
    * **Instance-level access** auto-fills ``returns`` / ``factor_returns`` from
      instance state via ``_MetricMethod`` descriptors.

    Only a handful of methods that cannot be expressed as a plain registry
    entry are defined explicitly below (e.g. ``annual_active_risk`` hides its
    kernel's ``out`` parameter, ``regression_annual_return`` composes several
    kernels).
    """

    def __init__(self, returns=None, positions=None, factor_returns=None, factor_loadings=None, **kwargs):
        """Initialize an Empyrical instance and store analysis inputs."""
        self.returns = returns
        self.positions = positions
        self.factor_returns = factor_returns
        self.factor_loadings = factor_loadings

    @property
    def _ctx(self):
        """Compatibility-only lazy context; metric dispatch does not retain it."""

        if self.returns is None:
            return None
        try:
            from fincore.core.context import AnalysisContext

            return AnalysisContext(
                self.returns,
                factor_returns=self.factor_returns,
                positions=self.positions,
            )
        except (TypeError, ValueError, KeyError):
            return None

    def __getattr__(self, name):
        """Safety-net for registry-backed attributes on instance access.

        Normally ``_LazyMethod`` descriptors (set by ``@_populate_from_registry``)
        handle both class-level and instance-level lookups.  This method acts as
        a fallback for edge cases (e.g. subclass access before descriptor
        resolution) by delegating to the same registry and caching the result
        on the class so that subsequent accesses are zero-overhead.
        """
        entry = CLASSMETHOD_REGISTRY.get(name)
        if entry is not None:
            mod_alias, func_name = entry
            func = getattr(_resolve_module(mod_alias), func_name)
            setattr(type(self), name, staticmethod(func))
            return func

        entry = STATIC_METHODS.get(name)
        if entry is not None:
            mod_alias, func_name = entry
            func = getattr(_resolve_module(mod_alias), func_name)
            setattr(type(self), name, staticmethod(func))
            return func

        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

    # ------------------------------------------------------------------
    # Instance data helpers
    # ------------------------------------------------------------------

    @_dual_method
    def _get_returns(self, returns):
        """Get returns, falling back to ``self.returns`` when ``returns`` is None."""
        if returns is not None:
            return returns
        if not isinstance(self, type) and hasattr(self, "returns") and self.returns is not None:
            return self.returns
        return None

    @_dual_method
    def _get_factor_returns(self, factor_return):
        """Get factor returns, falling back to ``self.factor_returns`` when ``factor_return`` is None."""
        if factor_return is not None:
            return factor_return
        if not isinstance(self, type) and hasattr(self, "factor_returns") and self.factor_returns is not None:
            return self.factor_returns
        return None

    # ------------------------------------------------------------------
    # Special-case methods (cannot be expressed as a plain registry entry)
    # ------------------------------------------------------------------

    @_dual_method
    def annual_active_risk(
        self,
        returns=None,
        factor_returns=None,
        period=DAILY,
        annualization=None,
        *,
        alignment: AlignmentPolicy = "inner",
        normalize_tz: str | None = None,
    ):
        """Compute annual active risk (tracking error)."""
        return _resolve_module("_risk").tracking_error(
            self._get_returns(returns),
            self._get_factor_returns(factor_returns),
            period=period,
            annualization=annualization,
            alignment=alignment,
            normalize_tz=normalize_tz,
        )

    @_dual_method
    def regression_annual_return(
        self,
        returns=None,
        factor_returns=None,
        risk_free=0.0,
        period=DAILY,
        annualization=None,
        *,
        alignment: AlignmentPolicy = "inner",
        normalize_tz: str | None = None,
    ):
        """Compute regression-based annual return."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        _ab = _resolve_module("_alpha_beta")
        _yr = _resolve_module("_yearly")
        alpha_val = _ab.alpha(
            returns,
            factor_returns,
            risk_free=risk_free,
            period=period,
            annualization=annualization,
            alignment=alignment,
            normalize_tz=normalize_tz,
        )
        beta_val = _ab.beta(
            returns,
            factor_returns,
            risk_free=risk_free,
            _period=period,
            _annualization=annualization,
            alignment=alignment,
            normalize_tz=normalize_tz,
        )
        if np.isnan(alpha_val) or np.isnan(beta_val):
            return np.nan
        benchmark_annual = _yr.annual_return(factor_returns, period, annualization)
        if np.isnan(benchmark_annual):
            return np.nan
        return alpha_val + beta_val * benchmark_annual

    @classmethod
    def _groupby_consecutive(cls, txn, max_delta=None):
        """Group transactions by consecutive timestamps."""
        if max_delta is None:
            max_delta = pd.Timedelta("8h")
        return _resolve_module("_round_trips").groupby_consecutive(txn, max_delta)


def _signature_from_text(signature_text: str) -> inspect.Signature:
    namespace: dict[str, Any] = {}
    exec(  # nosec B102 # frozen repo-owned manifest text (never user input); empty-builtins sandbox, stub def only
        f"def _f{signature_text}:\n    pass",
        {"__builtins__": {}},
        namespace,
    )
    return inspect.signature(namespace["_f"])


def _make_strict_wrapper(spec: MetricSpec):
    manifest_key = spec.signature_manifest_key
    if manifest_key is None:
        raise KeyError("strict wrapper requires a signature manifest key")
    try:
        manifest_name, signature_text = EMPYRICAL_SIGNATURE_MANIFEST[manifest_key]
    except KeyError:
        raise KeyError(f"unknown signature manifest key: {manifest_key}") from None
    if manifest_name != spec.public_name:
        raise ValueError(f"signature manifest symbol {manifest_name!r} does not match public name {spec.public_name!r}")
    signature = _signature_from_text(signature_text)
    expected_out_policy = "write_and_return" if "out" in signature.parameters else "unsupported"
    if spec.out_policy != expected_out_policy:
        raise ValueError(f"out policy {spec.out_policy!r} does not match signature policy {expected_out_policy!r}")

    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        kernel = _resolve_ref(spec.kernel_ref)
        adapter = _resolve_ref(spec.adapter_ref)
        from fincore._dispatch import _raw_kernel_execution

        with _raw_kernel_execution():
            return adapter(kernel, bound.arguments)

    wrapped: Any = wrapper
    wrapped.__name__ = spec.public_name
    wrapped.__qualname__ = spec.public_name
    wrapped.__module__ = __name__
    wrapped.__signature__ = signature
    return wrapped


_LEGACY_PUBLIC = []
for (_surface, _public_name, _variant), _spec in METRIC_REGISTRY.items():
    if _surface == "empyrical_module" and _variant == "strict-0.6.0":
        globals()[_public_name] = _make_strict_wrapper(_spec)
        _LEGACY_PUBLIC.append(_public_name)


__all__ = [
    *_LEGACY_PUBLIC,
    "DAILY",
    "WEEKLY",
    "MONTHLY",
    "QUARTERLY",
    "YEARLY",
    "ZIPLINE",
    "Empyrical",
]
