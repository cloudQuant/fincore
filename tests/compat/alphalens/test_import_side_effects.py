"""Import-side-effect regressions for the core-only Alphalens facade."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_core_alphalens_imports_do_not_eagerly_import_optional_visual_stack() -> None:
    """Core imports stay usable without matplotlib, seaborn, IPython, or statsmodels."""

    code = "\n".join(
        (
            "import sys",
            "import fincore",
            "import fincore.alphalens",
            "import fincore.alphalens.performance",
            "import fincore.alphalens.utils",
            "assert hasattr(fincore, 'alphalens')",
            "for root in ('matplotlib', 'seaborn', 'IPython', 'statsmodels'):",
            "    assert root not in sys.modules, root",
        )
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_plotting_facade_import_does_not_change_an_existing_matplotlib_backend() -> None:
    """Loading the strict plotting module must not call ``matplotlib.use`` or pyplot."""

    code = "\n".join(
        (
            "import matplotlib",
            "before = matplotlib.get_backend()",
            "import fincore.alphalens.plotting",
            "assert matplotlib.get_backend() == before",
            "assert 'matplotlib.pyplot' not in __import__('sys').modules",
        )
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    # Pin a headless backend: with MPLBACKEND unset, ``import matplotlib``
    # itself may select an interactive backend and load pyplot, which would
    # make the pyplot assertion below unverifiable.
    environment["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
