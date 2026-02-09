__version__ = "0.1"

# NumPy 2.0 dropped the ``np.unicode_`` alias which some optional
# dependencies (for example, older versions of ``numexpr`` or PyMC) still
# import. Provide a minimal forward-compat shim so importing fincore does
# not fail when those packages are present in the environment.
import numpy as _np

if not hasattr(_np, "unicode_"):
    _np.unicode_ = _np.str_

__all__ = [
    # Core classes
    "Empyrical",
    "Pyfolio",
    "analyze",
    "create_strategy_report",
    # Commonly-used metric functions (flat API)
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "annual_return",
    "annual_volatility",
    "cum_returns",
    "cum_returns_final",
    "alpha",
    "beta",
    "alpha_beta",
    "calmar_ratio",
    "omega_ratio",
    "information_ratio",
    "stability_of_timeseries",
    "tail_ratio",
    "value_at_risk",
    "capture",
    "downside_risk",
    "simple_returns",
    "aggregate_returns",
]

# ---------------------------------------------------------------------------
# Lazy imports — defer heavy submodules until first attribute access.
# ``from fincore import empyrical`` still works because Python resolves
# sub-module names before calling ``__getattr__``.
# ---------------------------------------------------------------------------

# Maps commonly-used function names to (module_path, attr_name)
_FLAT_API = {
    "sharpe_ratio":           ("fincore.metrics.ratios", "sharpe_ratio"),
    "sortino_ratio":          ("fincore.metrics.ratios", "sortino_ratio"),
    "calmar_ratio":           ("fincore.metrics.ratios", "calmar_ratio"),
    "omega_ratio":            ("fincore.metrics.ratios", "omega_ratio"),
    "information_ratio":      ("fincore.metrics.ratios", "information_ratio"),
    "stability_of_timeseries":("fincore.metrics.ratios", "stability_of_timeseries"),
    "capture":                ("fincore.metrics.ratios", "capture"),
    "max_drawdown":           ("fincore.metrics.drawdown", "max_drawdown"),
    "annual_return":          ("fincore.metrics.yearly", "annual_return"),
    "annual_volatility":      ("fincore.metrics.risk", "annual_volatility"),
    "downside_risk":          ("fincore.metrics.risk", "downside_risk"),
    "value_at_risk":          ("fincore.metrics.risk", "value_at_risk"),
    "tail_ratio":             ("fincore.metrics.risk", "tail_ratio"),
    "cum_returns":            ("fincore.metrics.returns", "cum_returns"),
    "cum_returns_final":      ("fincore.metrics.returns", "cum_returns_final"),
    "simple_returns":         ("fincore.metrics.returns", "simple_returns"),
    "aggregate_returns":      ("fincore.metrics.returns", "aggregate_returns"),
    "alpha":                  ("fincore.metrics.alpha_beta", "alpha"),
    "beta":                   ("fincore.metrics.alpha_beta", "beta"),
    "alpha_beta":             ("fincore.metrics.alpha_beta", "alpha_beta"),
}


def __getattr__(name: str):
    if name == "Empyrical":
        from .empyrical import Empyrical
        globals()["Empyrical"] = Empyrical
        return Empyrical
    if name == "Pyfolio":
        from .pyfolio import Pyfolio
        globals()["Pyfolio"] = Pyfolio
        return Pyfolio
    if name == "analyze":
        from .core.context import analyze
        globals()["analyze"] = analyze
        return analyze
    if name == "create_strategy_report":
        from .report import create_strategy_report
        globals()["create_strategy_report"] = create_strategy_report
        return create_strategy_report

    # Flat metric function API
    entry = _FLAT_API.get(name)
    if entry is not None:
        import importlib
        mod_path, attr_name = entry
        mod = importlib.import_module(mod_path)
        func = getattr(mod, attr_name)
        globals()[name] = func
        return func

    raise AttributeError(f"module 'fincore' has no attribute {name!r}")

