"""Optional-extras behavior contracts for the lazy root facade.

Two independent strategies:

1. A fresh subprocess imports the *checkout* with a meta-path blocker that
   makes every optional-extra root (matplotlib, pymc, playwright, ...) raise
   ``ModuleNotFoundError``.  This simulates a core-only install without
   touching the environment:

   - ``from fincore import *`` must succeed and must not even attempt an
     optional module (the blocker makes any attempt fail loudly);
   - ``Pyfolio`` must NOT be star-exported;
   - explicit ``from fincore import Pyfolio`` (and the ``fincore.pyfolio``
     module path) must raise ``DependencyError`` naming
     ``pip install fincore[pyfolio]``.

2. In-process tests pin the single version source: ``fincore.__version__``
   comes from ``importlib.metadata`` when installed, with a pyproject.toml
   fallback for bare source checkouts.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _pyproject_version() -> str:
    with PYPROJECT.open("rb") as fh:
        return str(tomllib.load(fh)["project"]["version"])


#: Every optional root that any extra installs; blocking any of them must not
#: break ``import fincore`` or ``from fincore import *``.
OPTIONAL_ROOTS = [
    "matplotlib",
    "seaborn",
    "IPython",
    "plotly",
    "bokeh",
    "playwright",
    "PyPDF2",
    "openpyxl",
    "pymc",
    "yfinance",
    "requests",
    "tushare",
    "akshare",
    "pandas_datareader",
]

_CONSUMER = textwrap.dedent(
    f"""
    import sys

    BLOCKED = {OPTIONAL_ROOTS!r}

    class _BlockOptionalExtras:
        # Strict core-only simulation: any import attempt (or unguarded
        # find_spec probe) of an optional root raises ModuleNotFoundError,
        # exactly like an absent module under a raising import hook.
        def find_spec(self, fullname, path=None, target=None):
            root = fullname.split(".", 1)[0]
            if root in BLOCKED:
                raise ModuleNotFoundError(f"No module named {{fullname!r}}", name=fullname)
            return None

    sys.meta_path.insert(0, _BlockOptionalExtras())

    import fincore
    from fincore.exceptions import DependencyError

    assert "Pyfolio" not in fincore.__all__, "Pyfolio must not be star-exported"

    # 1. Star import must succeed and must never import an optional module.
    namespace = {{}}
    exec("from fincore import *", namespace)
    for name in ("sharpe_ratio", "max_drawdown", "analyze", "Empyrical", "create_strategy_report"):
        assert name in namespace, f"star import missed {{name}}"
    assert "Pyfolio" not in namespace, "Pyfolio leaked through star import"
    for root in BLOCKED:
        assert root not in sys.modules, f"star import pulled optional module {{root}}"

    # 2. Explicit Pyfolio access raises the actionable DependencyError.
    for statement in ("from fincore import Pyfolio", "from fincore.pyfolio import Pyfolio"):
        try:
            exec(statement)
        except DependencyError as exc:
            assert "pip install fincore[pyfolio]" in str(exc), str(exc)
        else:
            raise AssertionError(f"expected DependencyError for {{statement!r}}")

    # 3. Core paths stay functional under the blocker.
    assert hasattr(fincore, "sharpe_ratio")
    assert fincore.__version__
    print("OPTIONAL_EXTRAS_OK")
    """
)


def test_core_only_facade_behavior_in_blocked_subprocess() -> None:
    """Star-import safety and Pyfolio DependencyError under blocked optional roots."""
    proc = subprocess.run(
        [sys.executable, "-c", _CONSUMER],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"consumer failed:\n{proc.stdout}\n{proc.stderr}"
    assert "OPTIONAL_EXTRAS_OK" in proc.stdout


def test_runtime_version_matches_pyproject() -> None:
    """The checkout's runtime ``__version__`` agrees with pyproject.toml."""
    import fincore

    assert Path(fincore.__file__).resolve().is_relative_to(REPO_ROOT), "imported fincore is not the checkout"
    assert fincore.__version__ == _pyproject_version()


def test_version_fallback_reads_pyproject_when_metadata_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source-tree fallback: without distribution metadata, read pyproject.toml."""
    import importlib.metadata

    import fincore

    def _missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(_name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    assert fincore._resolve_version() == _pyproject_version()


def test_version_fallback_function_reads_pyproject_directly() -> None:
    """The pyproject-reading helper returns the single-source version."""
    import fincore

    assert fincore._version_from_pyproject() == _pyproject_version()
