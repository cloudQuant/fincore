"""Fincore 0.5: domain-oriented quantitative analytics.

The root intentionally exposes only versioning, structured errors, and stable
domain namespaces. Metrics, reports, risk, factor analysis, data and runtime
operations live in their owning modules; no legacy compatibility API is
installed here.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from . import (
    attribution,
    data,
    extensions,
    factor_analysis,
    metrics,
    optimization,
    performance,
    portfolio,
    report,
    risk,
    runtime,
    simulation,
    viz,
)
from . import (
    exceptions as errors,
)


def _version_from_pyproject() -> str:
    """Read the version when importing an unpackaged source checkout."""

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
    """Read checkout metadata locally and wheel metadata only after installation.

    A source checkout may be imported while an unrelated editable ``fincore``
    distribution remains installed in the interpreter. Its distribution
    metadata must not override the candidate checkout's declared version.
    """

    if (Path(__file__).resolve().parent.parent / "pyproject.toml").is_file():
        return _version_from_pyproject()

    import importlib.metadata as _md

    try:
        return _md.version("fincore")
    except _md.PackageNotFoundError:
        return _version_from_pyproject()


__version__ = _resolve_version()

__all__ = [
    "__version__",
    "attribution",
    "data",
    "errors",
    "extensions",
    "factor_analysis",
    "metrics",
    "optimization",
    "performance",
    "portfolio",
    "report",
    "risk",
    "runtime",
    "simulation",
    "viz",
]
