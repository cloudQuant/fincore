"""Plugin registry for fincore framework.

Allows dynamic extension of fincore's capabilities through:
- Custom metric functions
- Custom visualization backends
- Event hooks (pre/post analysis, pre/post compute)
- Custom risk models
- Custom optimization objectives
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# =============================================================================
# Registry Storage
# =============================================================================

_METRIC_REGISTRY: Dict[str, Callable] = {}
_HOOK_REGISTRY: Dict[str, List[Callable]] = {}
_VIZ_BACKEND_REGISTRY: Dict[str, type] = {}


# =============================================================================
# Decorators
# =============================================================================

def register_metric(
    name: Optional[str] = None,
) -> Callable:
    """Decorator to register a custom metric function.

    The decorated function should:
    - Take returns (pd.Series or np.ndarray) as first argument
    - Return a single scalar value

    Example::
        @register_metric("custom_ratio")
        def custom_ratio(returns, period=252):
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            return mean / std * np.sqrt(period)
    """
    def decorator(func: Callable) -> Callable:
        metric_name = name or func.__name__

        @ functools.wraps(func)
        def wrapper(*args, **kwargs):
            # First argument must be returns
            if len(args) == 0:
                raise TypeError("Custom metric must take returns as first argument")

            returns_arg = args[0]

            # Calculate metric value
            result = func(*args, **kwargs)

            # Register the metric
            _METRIC_REGISTRY[metric_name] = result
            return result

        wrapper.__name__ = metric_name
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        return wrapper

    return decorator


def register_viz_backend(
    name: str,
) -> Callable:
    """Decorator to register a custom visualization backend.

    The decorated class should provide the following methods:
    - plot_returns(cum_returns, **kwargs) - Plot cumulative returns
    - plot_drawdown(drawdown, **kwargs) - Plot underwater chart
    - plot_rolling_sharpe(sharpe, **kwargs) - Plot rolling Sharpe ratio
    - plot_monthly_heatmap(returns, **kwargs) - Plot monthly heatmap

    The backend class should be initialized with theme parameter.

    Example::

        @register_viz_backend("my_backend")
        class MyBackend:
            def __init__(self, theme="light"):
                self.theme = theme

            def plot_returns(self, cum_returns, **kwargs):
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(cum_returns.index, cum_returns.values)
                return fig

    """
    def decorator(cls: type) -> type:
        backend_name = name or cls.__name__

        @classmethod
        def create_instance(cls, theme: str = "light") -> "MyBackend":
            return cls(theme=theme)

        def wrapper(cls, *args, **kwargs):
            instance = cls(*args, **kwargs)
            _VIZ_BACKEND_REGISTRY[backend_name] = instance
            return instance

        wrapper.__name__ = backend_name
        wrapper.__doc__ = cls.__doc__
        wrapper.__module__ = cls.__module__
        return wrapper

    return decorator


def register_hook(
    event: str,
    priority: int = 100,
) -> Callable:
    """Decorator to register an event hook.

    Available events:
    - "pre_analysis": Before AnalysisContext computes metrics, validate data
    - "post_analysis": After metrics computed, add custom results
    - "pre_compute": Before solving optimization, validate constraints
    - "post_compute": After optimization, validate results

    Hooks receive (analysis_context, returns, weights, factor_data, etc) depending on event type.

    Example::

        @register_hook("pre_analysis", priority=50)
        def validate_returns(returns, **kwargs):
            # Remove outliers
            returns_clean = returns[returns < 3 * returns.std()]
            return returns_clean
    """
    def decorator(func: Callable) -> Callable:
        event_name = event or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Register the hook
            _HOOK_REGISTRY[event_name] = _HOOK_REGISTRY.get(event_name, [])
            _HOOK_REGISTRY[event_name].append(wrapper)

            return func

        wrapper.__name__ = event_name
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        return wrapper

    return decorator


# =============================================================================
# Public API Functions
# =============================================================================

def list_metrics() -> Dict[str, Callable]:
    """List all registered custom metrics."""
    return _METRIC_REGISTRY.copy()


def list_viz_backends() -> Dict[str, type]:
    """List all registered visualization backends."""
    return _VIZ_BACKEND_REGISTRY.copy()


def list_hooks(event: Optional[str] = None) -> Dict[str, List[Callable]]:
    """List all registered hooks for an event."""
    if event is None:
        return {k: v for k, v in _HOOK_REGISTRY.items() for k in v}
    return _HOOK_REGISTRY.copy()


def get_metric(name: str) -> Optional[Callable]:
    """Get a registered metric by name."""
    return _METRIC_REGISTRY.get(name)


def get_viz_backend(name: str) -> Optional[type]:
    """Get a registered visualization backend by name."""
    return _VIZ_BACKEND_REGISTRY.get(name)


def execute_hooks(
    event: str,
    *args,
    **kwargs,
) -> None:
    """Execute all hooks registered for an event.

    Hooks are executed in registration order (priority value ascending).
    """
    for hook_fn in _HOOK_REGISTRY.get(event, []):
        hook_fn(*args, **kwargs)


# =============================================================================
# Example Custom Metrics
# =============================================================================

def _custom_sharpe(returns, period: int = 252) -> float:
    """Example custom Sharpe ratio calculation.

    Demonstrates the plugin system with a commonly-requested metric.
    """
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    excess_return = mean - 0.02  # Assume 2% risk-free rate

    return excess_return / std * np.sqrt(period)


# Register the custom metric
register_metric = register_metric("custom_sharpe")(_custom_sharpe)
