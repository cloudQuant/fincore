"""Quantitative finance risk and performance analytics library.

Lazy-loading facade: Empyrical, Pyfolio, analyze(), create_strategy_report(),
and flat API functions (sharpe_ratio, max_drawdown, etc.) load on first access.

``Pyfolio`` is intentionally excluded from ``__all__``: it requires the
optional ``pyfolio`` extra, so star imports must stay core-only.  The
explicit access ``from fincore import Pyfolio`` remains supported and raises
:class:`~fincore.exceptions.DependencyError` naming
``pip install fincore[pyfolio]`` when the extra is absent.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, NoReturn

from fincore._registry import METRIC_REGISTRY

# ---------------------------------------------------------------------------
# Single version source: pyproject.toml is authoritative.  Runtime resolution
# prefers the installed distribution metadata (wheel/editable); bare source
# checkouts fall back to reading pyproject.toml (tested).
# ---------------------------------------------------------------------------


def _version_from_pyproject() -> str:
    """Return the source-tree version from pyproject.toml (single source)."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.is_file():
        raise RuntimeError(
            "fincore is neither installed (no distribution metadata) nor a "
            "checkout (no pyproject.toml next to the package); cannot resolve version"
        )
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    return str(data["project"]["version"])


def _resolve_version() -> str:
    """Resolve the runtime version from installed metadata with a source-tree fallback."""
    import importlib.metadata as _md

    try:
        return _md.version("fincore")
    except _md.PackageNotFoundError:
        return _version_from_pyproject()


__version__ = _resolve_version()

#: Import roots whose absence means the ``pyfolio`` extra is missing.
_PYFOLIO_EXTRA_ROOTS = frozenset({"matplotlib", "seaborn", "IPython"})


def _raise_pyfolio_dependency_error(exc: ModuleNotFoundError) -> NoReturn:
    """Convert a missing optional dependency into an actionable DependencyError."""
    missing_root = (exc.name or "").split(".", 1)[0]
    if missing_root not in _PYFOLIO_EXTRA_ROOTS:
        raise exc
    from fincore.exceptions import DependencyError

    raise DependencyError(
        "Pyfolio requires the optional 'pyfolio' extra. Install it with:\n    pip install fincore[pyfolio]",
        dependency=missing_root or "pyfolio-extra",
    ) from exc


__all__ = [
    # Core classes (Pyfolio excluded: it requires the optional pyfolio extra)
    "Empyrical",
    "aggregate_returns",
    "alpha",
    "alpha_beta",
    "analyze",
    "annual_return",
    "annual_volatility",
    "beta",
    "calmar_ratio",
    "capture",
    "create_strategy_report",
    "cum_returns",
    "cum_returns_final",
    "downside_risk",
    "information_ratio",
    "max_drawdown",
    "omega_ratio",
    # Commonly-used metric functions (flat API)
    "sharpe_ratio",
    "simple_returns",
    "sortino_ratio",
    "stability_of_timeseries",
    "tail_ratio",
    "value_at_risk",
]

# ---------------------------------------------------------------------------
# Lazy imports — defer heavy submodules until first attribute access.
# ``from fincore import empyrical`` still works because Python resolves
# sub-module names before calling ``__getattr__``.
# ---------------------------------------------------------------------------

_FLAT_API = {
    "sharpe_ratio": ("fincore.metrics.ratios", "sharpe_ratio"),
    "sortino_ratio": ("fincore.metrics.ratios", "sortino_ratio"),
    "calmar_ratio": ("fincore.metrics.ratios", "calmar_ratio"),
    "omega_ratio": ("fincore.metrics.ratios", "omega_ratio"),
    "information_ratio": ("fincore.metrics.ratios", "information_ratio"),
    "stability_of_timeseries": ("fincore.metrics.ratios", "stability_of_timeseries"),
    "capture": ("fincore.metrics.ratios", "capture"),
    "max_drawdown": ("fincore.metrics.drawdown", "max_drawdown"),
    "annual_return": ("fincore.metrics.yearly", "annual_return"),
    "annual_volatility": ("fincore.metrics.risk", "annual_volatility"),
    "downside_risk": ("fincore.metrics.risk", "downside_risk"),
    "value_at_risk": ("fincore.metrics.risk", "value_at_risk"),
    "tail_ratio": ("fincore.metrics.risk", "tail_ratio"),
    "cum_returns": ("fincore.metrics.returns", "cum_returns"),
    "cum_returns_final": ("fincore.metrics.returns", "cum_returns_final"),
    "simple_returns": ("fincore.metrics.returns", "simple_returns"),
    "aggregate_returns": ("fincore.metrics.returns", "aggregate_returns"),
    "alpha": ("fincore.metrics.alpha_beta", "alpha"),
    "beta": ("fincore.metrics.alpha_beta", "beta"),
    "alpha_beta": ("fincore.metrics.alpha_beta", "alpha_beta"),
}
_FLAT_REGISTRY = {
    name: (surface, name, variant)
    for (surface, name, variant), spec in METRIC_REGISTRY.items()
    if surface == "fincore_flat" and variant == "enhanced-0.3.x"
}
assert set(_FLAT_REGISTRY) == set(_FLAT_API)


def __getattr__(name: str) -> Any:
    if name == "empyrical":
        import importlib

        module = importlib.import_module("fincore.empyrical")
        globals()["empyrical"] = module
        return module
    if name == "Empyrical":
        from .empyrical import Empyrical

        globals()["Empyrical"] = Empyrical
        return Empyrical
    if name == "Pyfolio":
        try:
            from .pyfolio import Pyfolio
        except ModuleNotFoundError as exc:
            _raise_pyfolio_dependency_error(exc)
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
    entry = _FLAT_REGISTRY.get(name)
    if entry is not None:
        from fincore._dispatch import metric_callable

        func = metric_callable(*entry)
        globals()[name] = func
        return func

    raise AttributeError(f"module 'fincore' has no attribute {name!r}")
