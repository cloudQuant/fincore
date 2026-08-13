#!/usr/bin/env python
"""Install the built wheel into isolated targets and smoke-test fresh consumers.

For each profile the script:

1. ``pip --target <tmp>`` installs the freshly built wheel (plus the profile
   extra when requested) into an isolated ``TemporaryDirectory``;
2. launches a **fresh** subprocess (``python -S -E`` — no site/user site, no
   environment injection) from a cwd outside the repository, whose bootstrap
   inserts ONLY the target on ``sys.path`` (stdlib stays available; nothing
   is inherited from ``PYTHONPATH``);
3. asserts, inside that consumer:

   - ``fincore.__file__`` resolves inside the target (importing from the
     checkout or the interpreter's site-packages fails the profile);
   - the core third-party stack (numpy/pandas/scipy/pytz/packaging) and every
     profile-allowed extra module (matplotlib/pymc/playwright/...) also load
     from the target;
   - profile-forbidden extras are absent via ``importlib.util.find_spec``;
   - explicit capability access raises ``DependencyError`` naming
     ``pip install fincore[pyfolio]`` when the extra is missing;
   - ``from fincore import *`` succeeds without triggering any optional
     module, and ``Pyfolio`` is not star-exported;
   - a real metric computation (``sharpe_ratio`` + ``analyze``) runs on the
     isolated stack.

Playwright smoke here is the Python-package import matrix entry; browser
binary installation and real PDF rendering belong to a separate two-stage
job (see .github/workflows notes).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import textwrap
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Every optional root any extra installs (star-import must never trigger one).
OPTIONAL_ROOTS = (
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
)

CORE_THIRD_PARTY = ("numpy", "pandas", "scipy", "pytz", "packaging")

PROFILES: dict[str, dict] = {
    "core": {
        "extra": None,
        "allowed_extra_imports": (),
        "forbidden_imports": OPTIONAL_ROOTS,
        "pyfolio_expected": "missing",
    },
    "pyfolio": {
        "extra": "pyfolio",
        "allowed_extra_imports": ("matplotlib", "seaborn", "IPython"),
        "forbidden_imports": (
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
        ),
        "pyfolio_expected": "available",
    },
    "interactive": {
        "extra": "interactive",
        "allowed_extra_imports": ("plotly", "bokeh"),
        "forbidden_imports": (
            "matplotlib",
            "seaborn",
            "IPython",
            "playwright",
            "PyPDF2",
            "openpyxl",
            "pymc",
            "yfinance",
            "requests",
            "tushare",
            "akshare",
            "pandas_datareader",
        ),
        "pyfolio_expected": "missing",
    },
    "bayesian": {
        "extra": "bayesian",
        # matplotlib may arrive transitively via arviz; pymc itself is required.
        "allowed_extra_imports": ("pymc",),
        "forbidden_imports": (
            "seaborn",
            "IPython",
            "plotly",
            "bokeh",
            "playwright",
            "PyPDF2",
            "openpyxl",
            "yfinance",
            "requests",
            "tushare",
            "akshare",
            "pandas_datareader",
        ),
        "pyfolio_expected": "missing",
    },
    "report-pdf": {
        "extra": "report-pdf",
        "allowed_extra_imports": ("playwright", "PyPDF2"),
        "forbidden_imports": (
            "matplotlib",
            "seaborn",
            "IPython",
            "plotly",
            "bokeh",
            "openpyxl",
            "pymc",
            "yfinance",
            "requests",
            "tushare",
            "akshare",
            "pandas_datareader",
        ),
        "pyfolio_expected": "missing",
    },
    "all": {
        "extra": "all",
        "allowed_extra_imports": OPTIONAL_ROOTS,
        "forbidden_imports": (),
        "pyfolio_expected": "available",
    },
}

_CONSUMER = textwrap.dedent(
    """
    import importlib
    import importlib.util
    import json
    import sys
    from pathlib import Path

    target, expected_version, profile = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    payload = json.loads(sys.argv[4])

    # Isolated bootstrap: ONLY the target joins the path (-S already dropped
    # site-packages; stdlib remains).
    sys.path.insert(0, str(target))

    import fincore
    from fincore.exceptions import DependencyError

    assert Path(fincore.__file__).resolve().is_relative_to(target.resolve()), \\
        f"fincore imported from outside the target: {fincore.__file__}"
    assert fincore.__version__ == expected_version, (fincore.__version__, expected_version)

    # Star import must never trigger an optional module.  Checked FIRST, before
    # the profile-allowed extras are deliberately imported below, so the
    # sys.modules assertion covers only what the star import itself pulled.
    namespace = {}
    exec("from fincore import *", namespace)
    assert "Pyfolio" not in namespace, "Pyfolio must not be star-exported"
    for root in payload["optional_roots"]:
        assert root not in sys.modules, f"star import pulled optional module {root} in {profile}"

    # The vendored report asset ships in the wheel regardless of extras.
    asset = Path(fincore.__file__).resolve().parent / "report" / "assets" / "echarts.min.js"
    assert asset.is_file(), "vendored ECharts asset missing from wheel"

    # Core third-party stack provenance.
    for mod_name in payload["core"]:
        module = importlib.import_module(mod_name)
        file = Path(module.__file__).resolve()
        assert file.is_relative_to(target.resolve()), f"{mod_name} loaded from {file}"

    # Profile-allowed extras load from the target too.
    for mod_name in payload["allowed"]:
        module = importlib.import_module(mod_name)
        file = Path(module.__file__).resolve()
        assert file.is_relative_to(target.resolve()), f"{mod_name} loaded from {file}"

    # Profile-forbidden extras are genuinely absent.
    for mod_name in payload["forbidden"]:
        assert importlib.util.find_spec(mod_name) is None, f"{mod_name} present in {profile} target"

    # Capability gate: Pyfolio access depends on the pyfolio extra.
    if payload["pyfolio_expected"] == "missing":
        try:
            from fincore import Pyfolio  # noqa: F401
        except DependencyError as exc:
            assert "pip install fincore[pyfolio]" in str(exc), str(exc)
        else:
            raise AssertionError("expected DependencyError for Pyfolio without the pyfolio extra")
        try:
            from fincore.pyfolio import Pyfolio  # noqa: F401
        except DependencyError as exc:
            assert "pip install fincore[pyfolio]" in str(exc), str(exc)
        else:
            raise AssertionError("expected DependencyError from fincore.pyfolio without the pyfolio extra")
    else:
        from fincore import Pyfolio  # noqa: F401
        assert Pyfolio.__name__ == "Pyfolio"
        instance = Pyfolio()
        assert instance.__class__.__name__ == "Pyfolio"

    # Real computation on the isolated stack.
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2024-01-02", periods=60, freq="B")
    returns = pd.Series(np.random.default_rng(7).normal(0.0005, 0.01, 60), index=idx)
    sharpe = fincore.sharpe_ratio(returns)
    assert np.isfinite(sharpe), sharpe
    context = fincore.analyze(returns)
    assert np.isfinite(context.sharpe_ratio)
    assert context.max_drawdown <= 0

    print(f"PROFILE {profile} OK")
    """
)


def _pyproject_version() -> str:
    with PYPROJECT.open("rb") as fh:
        return str(tomllib.load(fh)["project"]["version"])


def _scrubbed_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}


def _payload(profile: str) -> str:
    import json

    spec = PROFILES[profile]
    return json.dumps(
        {
            "core": list(CORE_THIRD_PARTY),
            "allowed": list(spec["allowed_extra_imports"]),
            "forbidden": list(spec["forbidden_imports"]),
            "optional_roots": list(OPTIONAL_ROOTS),
            "pyfolio_expected": spec["pyfolio_expected"],
        }
    )


def _run_profile(wheel: Path, profile: str, version: str) -> None:
    spec = PROFILES[profile]
    print(f"[{profile}] installing wheel{'[%s]' % spec['extra'] if spec['extra'] else ''} into isolated target ...")
    with tempfile.TemporaryDirectory(prefix=f"fincore-wheel-{profile}-") as work:
        target = Path(work) / "target"
        target.mkdir()
        consumer = Path(work) / "consumer.py"
        consumer.write_text(_CONSUMER, encoding="utf-8")
        cwd = Path(work) / "cwd"
        cwd.mkdir()

        install_spec = str(wheel) if spec["extra"] is None else f"fincore[{spec['extra']}] @ {wheel.resolve().as_uri()}"
        install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--target", str(target), install_spec],
            cwd=cwd,
            env=_scrubbed_env(),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if install.returncode != 0:
            raise RuntimeError(f"[{profile}] pip install failed:\n{install.stdout}\n{install.stderr}")

        run = subprocess.run(
            [sys.executable, "-S", "-E", str(consumer), str(target), version, profile, _payload(profile)],
            cwd=cwd,
            env=_scrubbed_env(),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if run.returncode != 0:
            raise RuntimeError(f"[{profile}] fresh consumer failed:\n{run.stdout}\n{run.stderr}")
        print(f"[{profile}] {run.stdout.strip()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", required=True, help="directory containing the built wheel")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(PROFILES),
        choices=list(PROFILES),
        help="profiles to smoke-test (default: all)",
    )
    args = parser.parse_args()

    version = _pyproject_version()
    # Resolve once: the per-profile pip installs run from temporary cwd's,
    # where a relative --dist path would no longer point at the wheel.
    dist = Path(args.dist).resolve()
    wheels = sorted(dist.glob(f"fincore-{version}-*.whl"))
    if not wheels:
        print(f"no wheel matching fincore-{version}-*.whl in {args.dist}", file=sys.stderr)
        return 1
    wheel = wheels[0]
    print(f"testing wheel: {wheel} (version {version})")

    failures = 0
    for profile in args.profiles:
        try:
            _run_profile(wheel, profile, version)
        except RuntimeError as exc:
            failures += 1
            print(f"[{profile}] FAILED: {exc}", file=sys.stderr)
    print(f"{len(args.profiles) - failures}/{len(args.profiles)} profiles passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
