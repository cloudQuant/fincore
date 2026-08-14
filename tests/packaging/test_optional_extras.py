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

import ast
import os
import runpy
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
    "statsmodels",
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


def test_installed_wheel_cli_accepts_required_alphalens_profiles(tmp_path: Path) -> None:
    """The wheel-consumer CLI exposes each required Alphalens install profile."""
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "test_installed_wheel.py"),
            "--dist",
            str(tmp_path),
            "--profiles",
            "core",
            "factor-analysis",
            "alphalens",
            "alphalens-pyfolio",
            "all",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # A missing test wheel is expected here; argparse must accept every
    # documented profile before the script reaches that artifact check.
    assert proc.returncode == 1, proc.stderr
    assert "no wheel matching fincore-" in proc.stderr


def test_isolated_wheel_consumer_bootstrap_imports_json() -> None:
    """The emitted ``-S -E`` consumer owns every standard-library import it uses."""
    namespace = runpy.run_path(str(REPO_ROOT / "scripts" / "test_installed_wheel.py"), run_name="wheel_consumer_test")
    consumer_tree = ast.parse(namespace["_CONSUMER"])
    imported_modules = {
        alias.name for node in ast.walk(consumer_tree) if isinstance(node, ast.Import) for alias in node.names
    }
    assert "json" in imported_modules


def test_required_wheel_profiles_have_explicit_bounded_timeouts() -> None:
    """A dependency resolver or consumer hang cannot block release CI forever."""
    namespace = runpy.run_path(str(REPO_ROOT / "scripts" / "test_installed_wheel.py"), run_name="wheel_timeout_test")
    profiles = namespace["PROFILES"]
    for name in ("core", "factor-analysis", "alphalens", "alphalens-pyfolio", "all"):
        spec = profiles[name]
        assert 0 < spec["install_timeout"] <= 900
        assert 0 < spec["consumer_timeout"] <= 300


def test_all_profile_uses_a_real_venv_for_pip_check() -> None:
    """``pip check`` must inspect its own environment, not a ``--target`` path."""
    namespace = runpy.run_path(str(REPO_ROOT / "scripts" / "test_installed_wheel.py"), run_name="wheel_pip_check_test")
    assert namespace["PROFILES"]["all"]["install_mode"] == "venv"


def test_pip_check_in_a_venv_detects_an_unsatisfied_installed_requirement(tmp_path: Path) -> None:
    """Regression: venv ``pip check`` sees its own broken installed metadata."""
    venv_dir = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True, timeout=120)
    python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    purelib = Path(
        subprocess.run(
            [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    )
    metadata = purelib / "fincore_pip_check_regression-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: fincore-pip-check-regression\n"
        "Version: 1.0\n"
        "Requires-Dist: fincore-definitely-missing-test-dependency (==1.0)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [python, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "fincore-definitely-missing-test-dependency" in result.stdout + result.stderr
