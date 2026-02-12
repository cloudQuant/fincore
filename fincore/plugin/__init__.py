"""Plugin system for extensible fincore framework.

Allows users to register:
- Custom metrics
- Custom visualization backends
- Event hooks
- Custom risk models
- Custom optimization objectives
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# =============================================================================
# Plugin Registry
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

    The metric should take returns (pd.Series or np.ndarray) and return
    a scalar value.

    Example::

        @register_metric("custom_ratio")
        def custom_ratio(returns, period=252):
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            return mean / std * np.sqrt(period)
    """
    def decorator(func: Callable) -> Callable:
        metric_name = name or func.__name__

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
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

    The backend should be a class with methods:
    - plot_returns(cum_returns, **kwargs)
    - plot_drawdown(drawdown, **kwargs)
    - plot_rolling_sharpe(sharpe, **kwargs)
    - plot_monthly_heatmap(returns, **kwargs)

    Example::

        @register_viz_backend("plotly")
        class PlotlyBackend:
            def __init__(self, theme="light"):
                self.theme = theme

            def plot_returns(self, cum_returns, **kwargs):
                import plotly.express as px
                fig = px.line(x=cum_returns.index, y=cum_returns.values)
                return fig

            def plot_drawdown(self, drawdown, **kwargs):
                import plotly.express as px
                fig = px.area(y=drawdown)
                return fig
    """
    def decorator(cls: type) -> type:
        metric_name = name or cls.__name__

        def wrapper(*args, **kwargs):
            instance = cls(*args, **kwargs)
            _VIZ_BACKEND_REGISTRY[metric_name] = instance
            return instance

        wrapper.__name__ = metric_name
        wrapper.__doc__ = cls.__doc__
        wrapper.__module__ = cls.__module__
        return wrapper

    return decorator


def register_hook(
    event: str,
    priority: int = 100,
) -> Callable:
    """Decorator to register an event hook.

    Event types: "pre_analysis", "post_analysis", "pre_compute", "post_compute"

    Example::

        @register_hook("pre_analysis", priority=50)
        def validate_data(returns):
            # Remove outliers
            returns_clean = returns[returns < 3 * returns.std()]
            return returns_clean

    @register_hook("post_compute", priority=100)
        def log_metrics(ctx):
            # Log computed metrics
            print(f"Sharpe: {ctx.sharpe_ratio:.4f}")
    """
    def decorator(func: Callable) -> Callable:
        event_name = event or func.__name__

        def wrapper(*args, **kwargs):
            def wrapped(*inner_args, **inner_kwargs):
                # Execute all hooks in order
                for hook_fn in _HOOK_REGISTRY.get(event_name, []):
                    hook_fn(*inner_args, **inner_kwargs)

            # Execute the actual function
            result = func(*args, **kwargs)
            return result

        wrapper.__name__ = event_name
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        return wrapper


# =============================================================================
# Core Plugin Functions
# =============================================================================

def list_metrics() -> Dict[str, Callable]:
    """List all registered custom metrics."""
    return _METRIC_REGISTRY.copy()


def list_viz_backends() -> Dict[str, type]:
    """List all registered visualization backends."""
    return _VIZ_BACKEND_REGISTRY.copy()


def list_hooks(event: Optional[str] = None) -> Dict[str, List[Callable]]:
    """List all registered hooks for an event.

    Parameters
    ----------
    event : str, optional
        Filter by event name. If None, returns all hooks.
    """
    if event is None:
        return {k: v for k, v in _HOOK_REGISTRY.items() for v}
    return {event: _HOOK_REGISTRY.get(event, [])}


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

    Parameters
    ----------
    event : str
        Event name (pre_analysis, post_analysis, etc.)
    *args, **kwargs
        Arguments to pass to hook functions.
    """
    for hook_fn in _HOOK_REGISTRY.get(event, []):
        hook_fn(*args, **kwargs)


# =============================================================================
# Examples
# =============================================================================

if __name__ == "__main__":
    # Example: Custom metric
    @register_metric("custom_sharpe")
    def custom_sharpe(returns, period=252):
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        return mean / std * np.sqrt(period)

    # Example: Viz backend
    # @register_viz_backend("my_backend")
    # class MyBackend:
    #     def plot_returns(self, cum_returns, **kwargs):
    #         import matplotlib.pyplot as plt
    #         plt.figure(figsize=(12, 6))
    #         plt.plot(cum_returns)
    #         return plt.gcf()

    # Example: Hook
    # @register_hook("pre_analysis")
    # def my_validation(returns):
    #     # Remove outliers
    #     return returns[returns < 3 * returns.std()]
    pass
