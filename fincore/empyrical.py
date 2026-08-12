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
from typing import Any

import numpy as np
import pandas as pd

from fincore._registry import (
    CLASSMETHOD_REGISTRY,
    EMPYRICAL_SIGNATURE_MANIFEST,
    METRIC_REGISTRY,
    MODULE_PATHS,
    STATIC_METHODS,
    MetricSpec,
)

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


# ---------------------------------------------------------------------------
# Lazy module resolver — replaces 17 top-level ``from fincore.metrics import …``
# ---------------------------------------------------------------------------
_MODULE_CACHE: dict = {}


def _resolve_module(alias: str):
    """Return the actual module for a registry alias like ``'_ratios'``."""
    try:
        return _MODULE_CACHE[alias]
    except KeyError:
        mod = importlib.import_module(MODULE_PATHS[alias])
        _MODULE_CACHE[alias] = mod
        return mod


def _resolve_ref(reference: str):
    """Resolve a lazy ``module:attribute`` registry reference."""

    module_name, attribute = reference.split(":", 1)
    return getattr(importlib.import_module(module_name), attribute)


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
        kernel = _resolve_ref(self.spec.kernel_ref)
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

    **Class-level access** (e.g. ``Empyrical.sharpe_ratio(returns)``)
    is auto-generated from the function registry in ``fincore._registry``
    via ``_LazyMethod`` descriptors attached by ``@_populate_from_registry``.

    **Instance-level access** with auto-fill of ``returns`` / ``factor_returns``
    is provided by explicitly defined ``@_dual_method`` methods below.
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
    # CAGR alias (not in registry because it maps to a *different* name)
    # ------------------------------------------------------------------

    @classmethod
    def cagr(cls, returns, period=DAILY, annualization=None):
        """Compute CAGR (alias of ``annual_return``)."""
        return _resolve_module("_yearly").annual_return(returns, period, annualization)

    # ==================================================================
    # @_dual_method wrappers — auto-fill returns from instance state
    # ==================================================================

    # ---- drawdown ----

    @_dual_method
    def get_max_drawdown_period(self, returns=None):
        """Return the start/end timestamps for the maximum drawdown period."""
        return _resolve_module("_drawdown").get_max_drawdown_period(self._get_returns(returns))

    @_dual_method
    def max_drawdown_days(self, returns=None):
        """Return the duration (days) of the maximum drawdown."""
        return _resolve_module("_drawdown").max_drawdown_days(self._get_returns(returns))

    @_dual_method
    def second_max_drawdown(self, returns=None):
        """Return the second-largest drawdown."""
        return _resolve_module("_drawdown").second_max_drawdown(self._get_returns(returns))

    @_dual_method
    def third_max_drawdown(self, returns=None):
        """Return the third-largest drawdown."""
        return _resolve_module("_drawdown").third_max_drawdown(self._get_returns(returns))

    @_dual_method
    def second_max_drawdown_days(self, returns=None):
        """Return the duration (days) of the second-largest drawdown."""
        return _resolve_module("_drawdown").second_max_drawdown_days(self._get_returns(returns))

    @_dual_method
    def second_max_drawdown_recovery_days(self, returns=None):
        """Return the recovery duration (days) of the second-largest drawdown."""
        return _resolve_module("_drawdown").second_max_drawdown_recovery_days(self._get_returns(returns))

    @_dual_method
    def third_max_drawdown_days(self, returns=None):
        """Return the duration (days) of the third-largest drawdown."""
        return _resolve_module("_drawdown").third_max_drawdown_days(self._get_returns(returns))

    @_dual_method
    def third_max_drawdown_recovery_days(self, returns=None):
        """Return the recovery duration (days) of the third-largest drawdown."""
        return _resolve_module("_drawdown").third_max_drawdown_recovery_days(self._get_returns(returns))

    @_dual_method
    def max_drawdown_weeks(self, returns=None):
        """Return the duration (weeks) of the maximum drawdown."""
        return _resolve_module("_drawdown").max_drawdown_weeks(self._get_returns(returns))

    @_dual_method
    def max_drawdown_months(self, returns=None):
        """Return the duration (months) of the maximum drawdown."""
        return _resolve_module("_drawdown").max_drawdown_months(self._get_returns(returns))

    @_dual_method
    def max_drawdown_recovery_days(self, returns=None):
        """Return the recovery duration (days) for the maximum drawdown."""
        return _resolve_module("_drawdown").max_drawdown_recovery_days(self._get_returns(returns))

    @_dual_method
    def max_drawdown_recovery_weeks(self, returns=None):
        """Return the recovery duration (weeks) for the maximum drawdown."""
        return _resolve_module("_drawdown").max_drawdown_recovery_weeks(self._get_returns(returns))

    @_dual_method
    def max_drawdown_recovery_months(self, returns=None):
        """Return the recovery duration (months) for the maximum drawdown."""
        return _resolve_module("_drawdown").max_drawdown_recovery_months(self._get_returns(returns))

    # ---- stats (returns only) ----

    @_dual_method
    def futures_market_correlation(self, returns=None, futures_returns=None):
        """Compute correlation between strategy returns and a futures market series."""
        return _resolve_module("_stats").futures_market_correlation(self._get_returns(returns), futures_returns)

    @_dual_method
    def serial_correlation(self, returns=None, lag=1):
        """Compute serial correlation for the given lag."""
        return _resolve_module("_stats").serial_correlation(self._get_returns(returns), lag)

    @_dual_method
    def win_rate(self, returns=None):
        """Compute win rate (fraction of positive returns)."""
        return _resolve_module("_stats").win_rate(self._get_returns(returns))

    @_dual_method
    def loss_rate(self, returns=None):
        """Compute loss rate (fraction of negative returns)."""
        return _resolve_module("_stats").loss_rate(self._get_returns(returns))

    # ---- consecutive (returns only) ----

    @_dual_method
    def max_consecutive_up_weeks(self, returns=None):
        """Compute the maximum number of consecutive up weeks."""
        return _resolve_module("_consecutive").max_consecutive_up_weeks(self._get_returns(returns))

    @_dual_method
    def max_consecutive_down_weeks(self, returns=None):
        """Compute the maximum number of consecutive down weeks."""
        return _resolve_module("_consecutive").max_consecutive_down_weeks(self._get_returns(returns))

    @_dual_method
    def max_consecutive_up_months(self, returns=None):
        """Compute the maximum number of consecutive up months."""
        return _resolve_module("_consecutive").max_consecutive_up_months(self._get_returns(returns))

    @_dual_method
    def max_consecutive_down_months(self, returns=None):
        """Compute the maximum number of consecutive down months."""
        return _resolve_module("_consecutive").max_consecutive_down_months(self._get_returns(returns))

    @_dual_method
    def max_single_day_gain_date(self, returns=None):
        """Return the date of the maximum single-day gain."""
        return _resolve_module("_consecutive").max_single_day_gain_date(self._get_returns(returns))

    @_dual_method
    def max_single_day_loss_date(self, returns=None):
        """Return the date of the maximum single-day loss."""
        return _resolve_module("_consecutive").max_single_day_loss_date(self._get_returns(returns))

    @_dual_method
    def max_consecutive_up_start_date(self, returns=None):
        """Return the start date of the maximum consecutive-up period."""
        return _resolve_module("_consecutive").max_consecutive_up_start_date(self._get_returns(returns))

    @_dual_method
    def max_consecutive_up_end_date(self, returns=None):
        """Return the end date of the maximum consecutive-up period."""
        return _resolve_module("_consecutive").max_consecutive_up_end_date(self._get_returns(returns))

    @_dual_method
    def max_consecutive_down_start_date(self, returns=None):
        """Return the start date of the maximum consecutive-down period."""
        return _resolve_module("_consecutive").max_consecutive_down_start_date(self._get_returns(returns))

    @_dual_method
    def max_consecutive_down_end_date(self, returns=None):
        """Return the end date of the maximum consecutive-down period."""
        return _resolve_module("_consecutive").max_consecutive_down_end_date(self._get_returns(returns))

    # ---- ratios (returns only) ----

    @_dual_method
    def common_sense_ratio(self, returns=None):
        """Compute the common sense ratio."""
        return _resolve_module("_ratios").common_sense_ratio(self._get_returns(returns))

    @_dual_method
    def sterling_ratio(self, returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Compute the Sterling ratio."""
        return _resolve_module("_ratios").sterling_ratio(self._get_returns(returns), risk_free, period, annualization)

    @_dual_method
    def burke_ratio(self, returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Compute the Burke ratio."""
        return _resolve_module("_ratios").burke_ratio(self._get_returns(returns), risk_free, period, annualization)

    @_dual_method
    def kappa_three_ratio(self, returns=None, risk_free=0.0, period=DAILY, annualization=None, mar=0.0):
        """Compute the Kappa-3 ratio."""
        return _resolve_module("_ratios").kappa_three_ratio(
            self._get_returns(returns), risk_free, period, annualization, mar
        )

    @_dual_method
    def deflated_sharpe_ratio(self, returns=None, risk_free=0, num_trials=1, period=DAILY, annualization=None):
        """Compute the Deflated Sharpe Ratio (DSR)."""
        return _resolve_module("_ratios").deflated_sharpe_ratio(
            self._get_returns(returns), risk_free, num_trials, period, annualization
        )

    # ---- risk (returns only) ----

    @_dual_method
    def gpd_risk_estimates(self, returns=None, var_p=0.01):
        """Estimate VaR and ES using a generalized Pareto distribution (GPD)."""
        return _resolve_module("_risk").gpd_risk_estimates(self._get_returns(returns), var_p)

    @_dual_method
    def gpd_risk_estimates_aligned(self, returns=None, var_p=0.01):
        """Estimate VaR and ES using GPD (aligned output)."""
        return _resolve_module("_risk").gpd_risk_estimates_aligned(self._get_returns(returns), var_p)

    @_dual_method
    def var_excess_return(self, returns=None, cutoff=0.05, risk_free=0.0, period=DAILY, annualization=None):
        """Compute the VaR excess return."""
        return _resolve_module("_risk").var_excess_return(
            self._get_returns(returns), cutoff, risk_free, period, annualization
        )

    @_dual_method
    def mar_ratio(self, returns=None, period=DAILY, annualization=None):
        """Compute the MAR ratio."""
        return _resolve_module("_ratios").mar_ratio(self._get_returns(returns), period, annualization)

    @_dual_method
    def r_cubed_turtle(self, returns=None, period=DAILY, annualization=None):
        """Compute the R^3 Turtle metric."""
        return _resolve_module("_stats").r_cubed_turtle(self._get_returns(returns), period, annualization)

    # ---- yearly (returns only) ----

    @_dual_method
    def annualized_cumulative_return(self, returns=None, period=DAILY, annualization=None):
        """Compute annualized cumulative return."""
        return _resolve_module("_yearly").annual_return(self._get_returns(returns), period, annualization)

    @_dual_method
    def annual_volatility_by_year(self, returns=None, period=DAILY, annualization=None):
        """Compute annual volatility by year."""
        return _resolve_module("_yearly").annual_volatility_by_year(self._get_returns(returns), period, annualization)

    # ---- rolling (returns only) ----

    @_dual_method
    def roll_sharpe_ratio(self, returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """Compute rolling Sharpe ratio."""
        return _resolve_module("_rolling").roll_sharpe_ratio(
            self._get_returns(returns), window, risk_free, period, annualization
        )

    @_dual_method
    def roll_max_drawdown(self, returns=None, window=252):
        """Compute rolling maximum drawdown."""
        return _resolve_module("_rolling").roll_max_drawdown(self._get_returns(returns), window)

    # ---- timing (returns only) ----

    @_dual_method
    def extract_interesting_date_ranges(self, returns=None):
        """Extract interesting date ranges."""
        return _resolve_module("_timing").extract_interesting_date_ranges(self._get_returns(returns))

    # ==================================================================
    # @_dual_method wrappers — auto-fill returns AND factor_returns
    # ==================================================================

    @_dual_method
    def r_cubed(self, returns=None, factor_returns=None):
        """Compute the R^3 metric."""
        return _resolve_module("_stats").r_cubed(self._get_returns(returns), self._get_factor_returns(factor_returns))

    @_dual_method
    def relative_win_rate(self, returns=None, factor_returns=None):
        """Compute relative win rate."""
        return _resolve_module("_stats").relative_win_rate(
            self._get_returns(returns), self._get_factor_returns(factor_returns)
        )

    @_dual_method
    def capm_r_squared(self, returns=None, factor_returns=None):
        """Compute CAPM R^2."""
        return _resolve_module("_stats").capm_r_squared(
            self._get_returns(returns), self._get_factor_returns(factor_returns)
        )

    @_dual_method
    def up_capture_return(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """Compute up-capture return."""
        return _resolve_module("_ratios").up_capture_return(
            self._get_returns(returns), self._get_factor_returns(factor_returns), period, annualization
        )

    @_dual_method
    def down_capture_return(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """Compute down-capture return."""
        return _resolve_module("_ratios").down_capture_return(
            self._get_returns(returns), self._get_factor_returns(factor_returns), period, annualization
        )

    @_dual_method
    def tracking_difference(self, returns=None, factor_returns=None):
        """Compute tracking difference."""
        return _resolve_module("_stats").tracking_difference(
            self._get_returns(returns), self._get_factor_returns(factor_returns)
        )

    @_dual_method
    def beta_fragility_heuristic(self, returns=None, factor_returns=None):
        """Estimate beta fragility (heuristic)."""
        return _resolve_module("_risk").beta_fragility_heuristic(
            self._get_returns(returns), self._get_factor_returns(factor_returns)
        )

    @_dual_method
    def beta_fragility_heuristic_aligned(self, returns=None, factor_returns=None):
        """Estimate beta fragility (heuristic, aligned output)."""
        return _resolve_module("_risk").beta_fragility_heuristic_aligned(
            self._get_returns(returns), self._get_factor_returns(factor_returns)
        )

    @_dual_method
    def treynor_ratio(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Compute the Treynor ratio."""
        return _resolve_module("_ratios").treynor_ratio(
            self._get_returns(returns), self._get_factor_returns(factor_returns), risk_free, period, annualization
        )

    @_dual_method
    def m_squared(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Compute the M^2 measure."""
        return _resolve_module("_ratios").m_squared(
            self._get_returns(returns), self._get_factor_returns(factor_returns), risk_free, period, annualization
        )

    @_dual_method
    def residual_risk(self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None):
        """Compute residual risk."""
        return _resolve_module("_risk").residual_risk(
            self._get_returns(returns), self._get_factor_returns(factor_returns), risk_free, period, annualization
        )

    # ---- rolling (returns + factor_returns) ----

    @_dual_method
    def roll_alpha(
        self, returns=None, factor_returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None
    ):
        """Compute rolling alpha."""
        return _resolve_module("_rolling").roll_alpha(
            self._get_returns(returns),
            self._get_factor_returns(factor_returns),
            window,
            risk_free,
            period,
            annualization,
        )

    @_dual_method
    def roll_beta(self, returns=None, factor_returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None):
        """Compute rolling beta."""
        return _resolve_module("_rolling").roll_beta(
            self._get_returns(returns),
            self._get_factor_returns(factor_returns),
            window,
            risk_free,
            period,
            annualization,
        )

    @_dual_method
    def roll_alpha_beta(
        self, returns=None, factor_returns=None, window=252, risk_free=0.0, period=DAILY, annualization=None
    ):
        """Compute rolling alpha and beta."""
        return _resolve_module("_rolling").roll_alpha_beta(
            self._get_returns(returns),
            self._get_factor_returns(factor_returns),
            window,
            risk_free,
            period,
            annualization,
        )

    @_dual_method
    def roll_up_capture(self, returns=None, factor_returns=None, window=252):
        """Compute rolling up-capture ratio."""
        return _resolve_module("_rolling").roll_up_capture(
            self._get_returns(returns), self._get_factor_returns(factor_returns), window
        )

    @_dual_method
    def roll_down_capture(self, returns=None, factor_returns=None, window=252):
        """Compute rolling down-capture ratio."""
        return _resolve_module("_rolling").roll_down_capture(
            self._get_returns(returns), self._get_factor_returns(factor_returns), window
        )

    @_dual_method
    def roll_up_down_capture(self, returns=None, factor_returns=None, window=252):
        """Compute rolling up/down capture ratio."""
        return _resolve_module("_rolling").roll_up_down_capture(
            self._get_returns(returns), self._get_factor_returns(factor_returns), window
        )

    # ---- yearly (returns + factor_returns) ----

    @_dual_method
    def annual_active_return(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """Compute annual active return."""
        return _resolve_module("_yearly").annual_active_return(
            self._get_returns(returns), self._get_factor_returns(factor_returns), period, annualization
        )

    @_dual_method
    def annual_active_risk(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """Compute annual active risk (tracking error)."""
        return _resolve_module("_risk").tracking_error(
            self._get_returns(returns), self._get_factor_returns(factor_returns), period, annualization
        )

    @_dual_method
    def annual_active_return_by_year(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """Compute annual active return by year."""
        return _resolve_module("_yearly").annual_active_return_by_year(
            self._get_returns(returns), self._get_factor_returns(factor_returns), period, annualization
        )

    @_dual_method
    def information_ratio_by_year(self, returns=None, factor_returns=None, period=DAILY, annualization=None):
        """Compute information ratio by year."""
        return _resolve_module("_yearly").information_ratio_by_year(
            self._get_returns(returns), self._get_factor_returns(factor_returns), period, annualization
        )

    # ==================================================================
    # Special-case methods (custom logic, not simple forwarding)
    # ==================================================================

    @_dual_method
    def perf_attrib(
        self,
        returns=None,
        positions=None,
        factor_returns=None,
        factor_loadings=None,
        transactions=None,
        pos_in_dollars=True,
        regression_style="OLS",
    ):
        """Compute performance attribution."""
        returns = self._get_returns(returns)
        if not isinstance(self, type):
            if positions is None and hasattr(self, "positions") and self.positions is not None:
                positions = self.positions
            if factor_returns is None and hasattr(self, "factor_returns") and self.factor_returns is not None:
                factor_returns = self.factor_returns
            if factor_loadings is None and hasattr(self, "factor_loadings") and self.factor_loadings is not None:
                factor_loadings = self.factor_loadings
        return _resolve_module("_perf_attrib").perf_attrib(
            returns, positions, factor_returns, factor_loadings, transactions, pos_in_dollars, regression_style
        )

    @_dual_method
    def regression_annual_return(
        self, returns=None, factor_returns=None, risk_free=0.0, period=DAILY, annualization=None
    ):
        """Compute regression-based annual return."""
        returns = self._get_returns(returns)
        factor_returns = self._get_factor_returns(factor_returns)
        _ab = _resolve_module("_alpha_beta")
        _yr = _resolve_module("_yearly")
        alpha_val = _ab.alpha(returns, factor_returns, risk_free, period, annualization)
        beta_val = _ab.beta(returns, factor_returns, risk_free, period, annualization)
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


def _enhanced_identity_adapter(kernel, arguments):
    """Forward an enhanced-surface call without changing its contract."""

    return kernel(**arguments)


def _legacy_identity_adapter(kernel, arguments):
    """Project legacy-named arguments onto a compatible metric kernel."""

    call_arguments = dict(arguments)
    call_arguments.update(call_arguments.pop("kwargs", {}))
    return kernel(**call_arguments)


def _legacy_beta_adapter(kernel, arguments):
    """Keep legacy beta's fourth positional parameter bound to ``out``."""

    return kernel(
        arguments["returns"],
        arguments["factor_returns"],
        risk_free=arguments.get("risk_free", 0.0),
        out=arguments.get("out"),
    )


def _legacy_annual_volatility_adapter(kernel, arguments):
    """Map empyrical's ``alpha_`` name to fincore's enhanced spelling."""

    return kernel(
        arguments["returns"],
        period=arguments.get("period", DAILY),
        volatility_power=arguments.get("alpha_", 2.0),
        annualization=arguments.get("annualization"),
        out=arguments.get("out"),
    )


def _legacy_calmar_adapter(kernel, arguments):
    """Hide fincore's enhanced risk-free parameter from the legacy surface."""

    return kernel(
        arguments["returns"],
        period=arguments.get("period", DAILY),
        annualization=arguments.get("annualization"),
    )


def _legacy_rolling_adapter(kernel, arguments):
    """Translate canonical factory names and enforce legacy out projection."""

    call_arguments = dict(arguments)
    out = call_arguments.pop("out", None)
    kernel_parameters = inspect.signature(kernel).parameters
    if "lhs" in call_arguments and "lhs" not in kernel_parameters:
        call_arguments["returns"] = call_arguments.pop("lhs")
    if "rhs" in call_arguments and "rhs" not in kernel_parameters:
        call_arguments["factor_returns"] = call_arguments.pop("rhs")
    if "arr" in call_arguments and "arr" not in kernel_parameters:
        call_arguments["returns"] = call_arguments.pop("arr")
    kwargs = call_arguments.pop("kwargs", {})
    call_arguments.update(kwargs)
    result = kernel(**call_arguments)
    if out is None:
        return result
    out[...] = np.asarray(result)
    return out


def _signature_from_text(signature_text: str) -> inspect.Signature:
    namespace: dict[str, Any] = {}
    exec(f"def _f{signature_text}:\n    pass", {"__builtins__": {}}, namespace)
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
        return adapter(kernel, bound.arguments)

    wrapper.__name__ = spec.public_name
    wrapper.__qualname__ = spec.public_name
    wrapper.__module__ = __name__
    wrapper.__signature__ = signature
    return wrapper


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
