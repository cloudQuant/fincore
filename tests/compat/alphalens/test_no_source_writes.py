"""Regression gate: visual factor workflows never write their source package."""

from __future__ import annotations

import hashlib
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path


def _non_cache_manifest(root: Path) -> dict[str, str]:
    """Hash only durable source files, excluding interpreter cache artifacts."""

    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
    }


_WORKFLOW_SCRIPT = r"""
import matplotlib
matplotlib.use("Agg")

import os
from pathlib import Path
import site
import sys


def _inside(path, root):
    try:
        Path(path).resolve().relative_to(root)
    except (OSError, TypeError, ValueError):
        return False
    return True


_protected_roots = tuple(
    Path(root).resolve()
    for root in os.environ["FINCORE_PROTECTED_ROOTS"].split(os.pathsep)
    if root
)


def _write_open(args):
    if len(args) < 3:
        return False
    mode, flags = args[1], args[2]
    if isinstance(mode, str) and any(token in mode for token in ("w", "a", "x", "+")):
        return True
    if isinstance(flags, int):
        return bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    return False


def _audit(event, args):
    if event == "open" and _write_open(args):
        targets = args[:1]
    elif event in {"os.remove", "os.unlink", "os.rename", "os.replace", "os.mkdir", "os.rmdir"}:
        targets = args[:2]
    else:
        return
    for target in targets:
        if any(_inside(target, root) for root in _protected_roots):
            raise RuntimeError(f"workflow attempted to write protected path: {target}")


sys.addaudithook(_audit)

import pandas as pd

from fincore.factor_analysis import analyze_factor
from fincore.factor_analysis.tears import (
    close_owned_figures,
    create_event_returns_tear_sheet,
    create_event_study_tear_sheet,
    create_full_tear_sheet,
    create_information_tear_sheet,
    create_returns_tear_sheet,
    create_summary_tear_sheet,
    create_turnover_tear_sheet,
)
from fincore.alphalens.tears import (
    create_event_returns_tear_sheet as strict_event_returns,
    create_event_study_tear_sheet as strict_event_study,
    create_full_tear_sheet as strict_full,
    create_information_tear_sheet as strict_information,
    create_returns_tear_sheet as strict_returns,
    create_summary_tear_sheet as strict_summary,
    create_turnover_tear_sheet as strict_turnover,
)

dates = pd.bdate_range("2024-01-02", periods=12, name="date")
assets = pd.Index(["A", "B", "C", "D"], name="asset")
index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
factor_data = pd.DataFrame(
    {
        "factor": [float(asset + 1) for _date in dates for asset in range(len(assets))],
        "factor_quantile": [asset + 1 for _date in dates for asset in range(len(assets))],
        "1D": [0.001 * (asset - 1.5) + 0.0001 * day for day, _date in enumerate(dates) for asset in range(len(assets))],
        "group": ["left" if asset < 2 else "right" for _date in dates for asset in range(len(assets))],
    },
    index=index,
)
event_returns = pd.DataFrame(
    {
        asset: [0.001 * (ordinal + 1) * ((day % 3) - 1) for day, _date in enumerate(dates)]
        for ordinal, asset in enumerate(assets)
    },
    index=dates,
)
model = analyze_factor(
    factor_data,
    periods=("1D",),
    turnover_periods=(1,),
    by_group=True,
    include_pyfolio=False,
    event_returns=event_returns,
    event_before=1,
    event_after=2,
)
for workflow, kwargs in (
    (create_summary_tear_sheet, {}),
    (create_returns_tear_sheet, {"by_group": True}),
    (create_information_tear_sheet, {"by_group": True}),
    (create_turnover_tear_sheet, {}),
    (create_full_tear_sheet, {"by_group": True}),
    (create_event_returns_tear_sheet, {"by_group": True}),
    (create_event_study_tear_sheet, {"avgretplot": (1, 2)}),
):
    artifacts = workflow(model, **kwargs)
    close_owned_figures(artifacts)

for workflow, kwargs in (
    (strict_summary, {}),
    (strict_returns, {"by_group": True}),
    (strict_information, {"by_group": True}),
    (strict_turnover, {}),
    (strict_full, {"by_group": True}),
    (strict_event_returns, {"by_group": True}),
    (strict_event_study, {"avgretplot": (1, 2)}),
):
    workflow(factor_data, event_returns, set_context=False, **kwargs) if workflow in {
        strict_event_returns,
        strict_event_study,
    } else workflow(factor_data, set_context=False, **kwargs)
"""


def test_all_factor_workflows_are_source_tree_read_only(tmp_path: Path) -> None:
    """A disposable checkout remains byte-identical after all seven workflows run."""

    repository_root = Path(__file__).resolve().parents[3]
    checkout = tmp_path / "checkout"
    source_package = repository_root / "fincore"
    copied_package = checkout / "fincore"
    shutil.copytree(source_package, copied_package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    before_checkout = _non_cache_manifest(checkout)
    before_source = _non_cache_manifest(source_package)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    protected_roots = [
        checkout,
        source_package,
        *(Path(path) for path in site.getsitepackages()),
        # A base Conda interpreter can keep its user-site enabled even when
        # the current checkout has no user-installed fincore package.
        Path(site.getusersitepackages()),
    ]
    environment["FINCORE_PROTECTED_ROOTS"] = os.pathsep.join(str(path) for path in protected_roots)

    completed = subprocess.run(
        [sys.executable, "-c", _WORKFLOW_SCRIPT],
        cwd=checkout,
        env=environment,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _non_cache_manifest(checkout) == before_checkout
    assert _non_cache_manifest(source_package) == before_source
