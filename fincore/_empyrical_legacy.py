"""Strict empyrical 0.6.0 compatibility adapters.

These adapters project empyrical 0.6.0's frozen signatures and semantics onto
fincore's enhanced metric kernels.  They are referenced lazily by
``fincore._registry._legacy_adapter_for`` and serve the ``empyrical_module``
(strict-0.6.0) surface only — the ``Empyrical`` class and ``fincore.metrics``
use the enhanced surface instead.

Keeping them in a separate module keeps ``empyrical.py`` a thin enhanced
interface and groups the strict-compatibility projection logic in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fincore._registry import _resolve_module
from fincore.constants import DAILY


def _legacy_identity_adapter(kernel, arguments):
    """Project legacy-named arguments onto a compatible metric kernel."""

    call_arguments = dict(arguments)
    call_arguments.update(call_arguments.pop("kwargs", {}))
    return kernel(**call_arguments)


def _legacy_beta_adapter(kernel, arguments):
    """Keep legacy beta's fourth positional parameter bound to ``out``."""

    del kernel
    returns, factor_returns = _resolve_module("_basic").aligned_series(
        arguments["returns"], arguments["factor_returns"]
    )
    return _resolve_module("_alpha_beta").beta_aligned(
        returns,
        factor_returns,
        risk_free=arguments.get("risk_free", 0.0),
        out=arguments.get("out"),
    )


def _legacy_aligned_binary_adapter(kernel, arguments):
    """Apply the pinned outer-label projection before an aligned kernel."""

    returns, factor_returns = _resolve_module("_basic").aligned_series(
        arguments["returns"], arguments["factor_returns"]
    )
    name = kernel.__name__
    if name == "alpha":
        return _resolve_module("_alpha_beta").alpha_aligned(
            returns,
            factor_returns,
            risk_free=arguments.get("risk_free", 0.0),
            period=arguments.get("period", DAILY),
            annualization=arguments.get("annualization"),
            out=arguments.get("out"),
            _beta=arguments.get("_beta"),
        )
    if name == "alpha_beta":
        return _resolve_module("_alpha_beta").alpha_beta_aligned(
            returns,
            factor_returns,
            risk_free=arguments.get("risk_free", 0.0),
            period=arguments.get("period", DAILY),
            annualization=arguments.get("annualization"),
            out=arguments.get("out"),
        )
    if name == "beta_fragility_heuristic":
        return _resolve_module("_risk").beta_fragility_heuristic_aligned(returns, factor_returns)
    raise KeyError(f"no legacy aligned projection for {name!r}")


def _legacy_capture_adapter(kernel, arguments):
    """Preserve pinned capture filtering without pre-aligning Series labels."""

    name = kernel.__name__
    returns = arguments["returns"]
    factor_returns = arguments["factor_returns"]
    kwargs = dict(arguments.get("kwargs", {}))
    if "period" in arguments:
        kwargs["period"] = arguments["period"]
    period = kwargs.pop("period", DAILY)
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"capture() got an unexpected keyword argument {unexpected!r}")

    annual_return = _resolve_module("_yearly").annual_return

    def capture(left, right):
        return annual_return(left, period=period) / annual_return(right, period=period)

    def filtered(sign):
        mask = factor_returns > 0 if sign == "up" else factor_returns < 0
        return capture(returns[mask], factor_returns[mask])

    if name == "capture":
        return capture(returns, factor_returns)
    if name == "up_capture":
        return filtered("up")
    if name == "down_capture":
        return filtered("down")
    return filtered("up") / filtered("down")


def _legacy_conditional_alpha_beta_adapter(kernel, arguments):
    """Preserve pinned up/down filtering before the aligned alpha-beta kernel."""

    returns = arguments["returns"]
    factor_returns = arguments["factor_returns"]
    kwargs = dict(arguments.get("kwargs", {}))
    mask = factor_returns > 0 if kernel.__name__ == "up_alpha_beta" else factor_returns < 0
    return _resolve_module("_alpha_beta").alpha_beta_aligned(
        returns[mask],
        factor_returns[mask],
        **kwargs,
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


def _legacy_value_at_risk_adapter(kernel, arguments):
    """Preserve empyrical's unvalidated percentile semantics."""

    del kernel
    return np.percentile(arguments["returns"], 100 * arguments.get("cutoff", 0.05))


def _legacy_conditional_value_at_risk_adapter(kernel, arguments):
    """Preserve empyrical's fixed-count order-statistics tail."""

    del kernel
    returns = arguments["returns"]
    cutoff_index = int((len(returns) - 1) * arguments.get("cutoff", 0.05))
    return np.mean(np.partition(returns, cutoff_index)[: cutoff_index + 1])


def _legacy_rolling_window(array, length):
    """Reproduce the pinned factory's first-axis restriding contract."""

    if not length:
        raise ValueError("Can't have 0-length window")
    original_shape = array.shape
    if not original_shape:
        raise IndexError("Can't restride a scalar.")
    if original_shape[0] < length:
        raise IndexError(f"Can't restride array of shape {original_shape} with a window length of {length}")
    new_shape = (original_shape[0] - length + 1, length, *original_shape[1:])
    new_strides = (array.strides[0], *array.strides)
    result = np.lib.stride_tricks.as_strided(array, new_shape, new_strides)
    result.setflags(write=False)
    return result


def _legacy_max_drawdown(returns, out=None):
    """Pinned scalar max-drawdown implementation for strict rolling."""

    if out is None:
        out = np.empty(returns.shape[1:])
    if len(returns) < 1:
        out[()] = np.nan
        return out.item() if returns.ndim == 1 else out

    values = np.asanyarray(returns).copy()
    values[np.isnan(values)] = 0
    cumulative = np.empty((values.shape[0] + 1, *values.shape[1:]), dtype="float64")
    cumulative[0] = 100
    np.add(values, 1, out=cumulative[1:])
    cumulative[1:].cumprod(axis=0, out=cumulative[1:])
    np.multiply(cumulative[1:], 100, out=cumulative[1:])
    maximum = np.fmax.accumulate(cumulative, axis=0)
    np.nanmin((cumulative - maximum) / maximum, axis=0, out=out)
    return out.item() if returns.ndim == 1 else out


def _legacy_sharpe_ratio(returns, risk_free=0, period=DAILY, annualization=None, out=None):
    """Pinned scalar Sharpe implementation for strict rolling."""

    if out is None:
        out = np.empty(returns.shape[1:])
    if len(returns) < 2:
        out[()] = np.nan
        return out.item() if returns.ndim == 1 else out

    adjusted = np.asanyarray(returns if risk_free == 0 else returns - risk_free)
    annual_factor = _resolve_module("_basic").annualization_factor(period, annualization)
    with np.errstate(divide="ignore", invalid="ignore"):
        standard_deviation = np.nanstd(adjusted, ddof=1, axis=0)
        average = np.nanmean(adjusted, axis=0)
        np.multiply(np.divide(average, standard_deviation, out=out), np.sqrt(annual_factor), out=out)
    return out.item() if returns.ndim == 1 else out


def _legacy_factory_scalar(name):
    if name == "roll_max_drawdown":
        return _legacy_max_drawdown
    if name == "roll_sharpe_ratio":
        return _legacy_sharpe_ratio
    references = {
        "roll_alpha": ("_alpha_beta", "alpha"),
        "roll_alpha_aligned": ("_alpha_beta", "alpha_aligned"),
        "roll_alpha_beta": ("_alpha_beta", "alpha_beta_aligned"),
        "roll_alpha_beta_aligned": ("_alpha_beta", "alpha_beta_aligned"),
        "roll_annual_volatility": ("_risk", "annual_volatility"),
        "roll_beta": ("_alpha_beta", "beta"),
        "roll_beta_aligned": ("_alpha_beta", "beta_aligned"),
        "roll_sortino_ratio": ("_ratios", "sortino_ratio"),
    }
    module_name, function_name = references[name]
    return getattr(_resolve_module(module_name), function_name)


def _legacy_rolling_adapter(kernel, arguments):
    """Reproduce pinned unary/binary vectorized rolling factories."""

    name = kernel.__name__
    call_arguments = dict(arguments)
    kwargs = dict(call_arguments.pop("kwargs", {}))
    excluded = {"arr", "lhs", "rhs", "returns", "factor_returns", "window", "out"}
    kwargs.update({key: value for key, value in call_arguments.items() if key not in excluded})
    window = call_arguments["window"]
    out = kwargs.pop("out", call_arguments.get("out"))
    allocated_output = out is None
    scalar = _legacy_factory_scalar(name)

    if "arr" in call_arguments:
        value = call_arguments["arr"]
        if len(value):
            flattened = value.values if isinstance(value, pd.Series) else value
            windows = _legacy_rolling_window(flattened, min(len(value), window)).T
            result = scalar(windows, out=out, **kwargs)
        else:
            result = np.empty(0, dtype="float64")
        if allocated_output and isinstance(value, pd.Series):
            result = pd.Series(result, index=value.index[-len(result) :])
        return result

    left = call_arguments.get("lhs", call_arguments.get("returns"))
    right = call_arguments.get("rhs", call_arguments.get("factor_returns"))
    if name == "roll_alpha_beta":
        left, right = _resolve_module("_basic").aligned_series(left, right)
    if window >= 1 and len(left) and len(right):
        left_values = left.values if isinstance(left, pd.Series) else left
        right_values = right.values if isinstance(right, pd.Series) else right
        left_windows = _legacy_rolling_window(left_values, min(len(left), window)).T
        right_windows = _legacy_rolling_window(right_values, min(len(right), window)).T
        result = scalar(left_windows, right_windows, out=out, **kwargs)
    elif allocated_output:
        result = np.empty(0, dtype="float64")
    else:
        out[()] = np.nan
        result = out
    if allocated_output and isinstance(left, pd.Series):
        if result.ndim == 1:
            result = pd.Series(result, index=left.index[-len(result) :])
        elif result.ndim == 2:
            result = pd.DataFrame(result, index=left.index[-len(result) :])
    return result


def _legacy_capture_scalar(name, returns, factor_returns, **kwargs):
    period = kwargs.pop("period", DAILY)
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"capture() got an unexpected keyword argument {unexpected!r}")
    annual_return = _resolve_module("_yearly").annual_return

    def capture(left, right):
        return annual_return(left, period=period) / annual_return(right, period=period)

    def filtered(sign):
        mask = factor_returns > 0 if sign == "up" else factor_returns < 0
        return capture(returns[mask], factor_returns[mask])

    if name == "roll_up_capture":
        return filtered("up")
    if name == "roll_down_capture":
        return filtered("down")
    return filtered("up") / filtered("down")


def _legacy_capture_rolling_adapter(kernel, arguments):
    """Reproduce the pinned ``utils.roll`` capture-family path."""

    name = kernel.__name__
    returns = arguments["returns"]
    factor_returns = arguments["factor_returns"]
    window = arguments.get("window", 10)
    kwargs = dict(arguments.get("kwargs", {}))
    if not isinstance(returns, type(factor_returns)):
        raise ValueError("The two returns arguments are not the same.")

    if isinstance(returns, np.ndarray):
        return np.array(
            [
                _legacy_capture_scalar(
                    name,
                    returns[index - window : index],
                    factor_returns[index - window : index],
                    **kwargs,
                )
                for index in range(window, len(returns) + 1)
            ]
        )

    data = {}
    index_values = []
    for index in range(window, len(returns) + 1):
        index_value = returns.index[index - 1]
        index_values.append(index_value)
        data[index_value] = _legacy_capture_scalar(
            name,
            returns.iloc[index - window : index],
            factor_returns.iloc[index - window : index],
            **kwargs,
        )
    return pd.Series(data, index=type(returns.index)(index_values), dtype=np.float64)


def _legacy_perf_attrib_adapter(kernel, arguments):
    """Reproduce pinned perf-attribution label and reduction semantics."""

    del kernel
    returns = arguments["returns"]
    positions = arguments["positions"].copy()
    factor_returns = arguments["factor_returns"].loc[returns.index[0] : returns.index[-1]].copy()
    factor_loadings = arguments["factor_loadings"].loc[returns.index[0] : returns.index[-1]].copy()
    factor_loadings.index = factor_loadings.index.set_names(["dt", "ticker"])
    positions.index = positions.index.set_names(["dt", "ticker"])

    risk_exposures = factor_loadings.multiply(positions, axis="rows").groupby(level="dt").sum()
    attribution_by_factor = risk_exposures.multiply(factor_returns)
    common_returns = attribution_by_factor.sum(axis="columns")
    tilt_exposure = risk_exposures.mean()
    tilt_returns = factor_returns.multiply(tilt_exposure).sum(axis="columns")
    timing_returns = common_returns - tilt_returns
    specific_returns = returns - common_returns
    returns_frame = pd.DataFrame(
        {
            "total_returns": returns,
            "common_returns": common_returns,
            "specific_returns": specific_returns,
            "tilt_returns": tilt_returns,
            "timing_returns": timing_returns,
        }
    )
    return risk_exposures, pd.concat([attribution_by_factor, returns_frame], axis="columns")
