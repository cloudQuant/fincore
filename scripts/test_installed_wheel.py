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
import json
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
    "statsmodels",
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

_PYFOLIO_VISUAL_ROOTS = ("matplotlib", "seaborn", "IPython")
_ALPHALENS_ROOTS = ("statsmodels", *_PYFOLIO_VISUAL_ROOTS)


def _forbidden_except(*allowed: str) -> tuple[str, ...]:
    allowed_roots = set(allowed)
    return tuple(root for root in OPTIONAL_ROOTS if root not in allowed_roots)


PROFILES: dict[str, dict] = {
    "core": {
        "extra": None,
        "allowed_extra_imports": (),
        "forbidden_imports": OPTIONAL_ROOTS,
        "pyfolio_expected": "missing",
        "smokes": ("core",),
        "install_timeout": 300,
        "consumer_timeout": 90,
    },
    "factor-analysis": {
        "extra": "factor-analysis",
        "allowed_extra_imports": ("statsmodels",),
        "forbidden_imports": _forbidden_except("statsmodels"),
        "pyfolio_expected": "missing",
        "smokes": ("factor-analysis",),
        "install_timeout": 300,
        "consumer_timeout": 120,
    },
    "alphalens": {
        "extra": "alphalens",
        "allowed_extra_imports": _ALPHALENS_ROOTS,
        "forbidden_imports": _forbidden_except(*_ALPHALENS_ROOTS),
        # Alphalens has the same visualization roots as the pyfolio facade,
        # but this profile exercises only the factor/plotting boundary.
        "pyfolio_expected": "skip",
        "smokes": ("factor-analysis", "alphalens"),
        "install_timeout": 600,
        "consumer_timeout": 180,
    },
    "alphalens-pyfolio": {
        "extra": "alphalens,pyfolio",
        "allowed_extra_imports": _ALPHALENS_ROOTS,
        "forbidden_imports": _forbidden_except(*_ALPHALENS_ROOTS),
        "pyfolio_expected": "available",
        "smokes": ("factor-analysis", "alphalens", "alphalens-pyfolio"),
        "install_timeout": 600,
        "consumer_timeout": 240,
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
        "smokes": ("core",),
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
        "smokes": ("core",),
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
        "smokes": ("core",),
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
        "smokes": ("core",),
    },
    "all": {
        "extra": "all",
        "allowed_extra_imports": OPTIONAL_ROOTS,
        "forbidden_imports": (),
        "pyfolio_expected": "available",
        "smokes": ("core", "factor-analysis", "alphalens", "alphalens-pyfolio"),
        "pip_check": True,
        "install_mode": "venv",
        "install_timeout": 900,
        "consumer_timeout": 300,
        "pip_check_timeout": 120,
    },
}

_CONSUMER = textwrap.dedent(
    """
    import importlib
    import importlib.util
    import json
    import os
    import sys
    from pathlib import Path

    target, expected_version, profile = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    payload = json.loads(sys.argv[4])
    os.environ.setdefault("MPLBACKEND", "Agg")

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
    elif payload["pyfolio_expected"] == "available":
        from fincore import Pyfolio  # noqa: F401
        assert Pyfolio.__name__ == "Pyfolio"
        instance = Pyfolio()
        assert instance.__class__.__name__ == "Pyfolio"

    # Real core computation on the isolated stack.
    print(f"PROFILE {profile}: core smoke", flush=True)
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2024-01-02", periods=60, freq="B")
    returns = pd.Series(np.random.default_rng(7).normal(0.0005, 0.01, 60), index=idx)
    sharpe = fincore.sharpe_ratio(returns)
    assert np.isfinite(sharpe), sharpe
    context = fincore.analyze(returns)
    assert np.isfinite(context.sharpe_ratio)
    assert context.max_drawdown <= 0

    def factor_data():
        # Create a small in-consumer factor table without checkout fixtures.
        from fincore.factor_analysis.data import prepare_factor_data

        dates = pd.date_range("2024-01-02", periods=24, freq="B")
        assets = ("A", "B", "C")
        index = pd.MultiIndex.from_product((dates[:-3], assets), names=("date", "asset"))
        values = [
            float((date_index + asset_index) % len(assets))
            for date_index in range(len(dates) - 3)
            for asset_index in range(len(assets))
        ]
        factor = pd.Series(values, index=index, dtype=float)
        steps = np.arange(len(dates), dtype=float)
        prices = pd.DataFrame(
            {
                asset: 100.0 * (1.0 + 0.0005 * (asset_index + 1)) ** steps
                for asset_index, asset in enumerate(assets)
            },
            index=dates,
        )
        return prepare_factor_data(factor, prices, periods=(1,), quantiles=3, max_loss=1.0).data

    factor_table = None
    if "factor-analysis" in payload["smokes"]:
        print(f"PROFILE {profile}: factor-analysis smoke", flush=True)
        from fincore.factor_analysis.performance import factor_alpha_beta, factor_information_coefficient

        factor_table = factor_data()
        information = factor_information_coefficient(factor_table)
        alpha_beta = factor_alpha_beta(factor_table)
        assert not information.empty and not alpha_beta.empty

    if "alphalens" in payload["smokes"]:
        print(f"PROFILE {profile}: alphalens plot and summary smoke", flush=True)
        import matplotlib.pyplot as plt
        from fincore.alphalens import performance as alphalens_performance
        from fincore.alphalens import plotting as alphalens_plotting
        from fincore.alphalens import tears as alphalens_tears

        assert factor_table is not None
        mean_returns, _ = alphalens_performance.mean_return_by_quantile(factor_table)
        figure, axis = plt.subplots()
        alphalens_plotting.plot_quantile_returns_bar(mean_returns, ax=axis)
        assert figure.axes
        plt.close(figure)
        assert alphalens_tears.create_summary_tear_sheet(factor_table) is None
        plt.close("all")

    if "alphalens-pyfolio" in payload["smokes"]:
        print(f"PROFILE {profile}: Alphalens-to-Pyfolio smoke", flush=True)
        import matplotlib.pyplot as plt
        from fincore import Pyfolio
        from fincore.alphalens.performance import create_pyfolio_input

        assert factor_table is not None
        factor_returns, positions, benchmark = create_pyfolio_input(factor_table, "1D", capital=1_000.0)
        figure = Pyfolio().create_returns_tear_sheet(
            factor_returns,
            positions=positions,
            benchmark_rets=benchmark,
            run_flask_app=True,
        )
        assert figure is not None and figure.axes
        plt.close(figure)

    print(f"PROFILE {profile} OK")
    """
)


def _pyproject_version() -> str:
    with PYPROJECT.open("rb") as fh:
        return str(tomllib.load(fh)["project"]["version"])


def _scrubbed_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}


def _payload(profile: str) -> str:
    spec = PROFILES[profile]
    return json.dumps(
        {
            "core": list(CORE_THIRD_PARTY),
            "allowed": list(spec["allowed_extra_imports"]),
            "forbidden": list(spec["forbidden_imports"]),
            "optional_roots": list(OPTIONAL_ROOTS),
            "pyfolio_expected": spec["pyfolio_expected"],
            "smokes": list(spec["smokes"]),
        }
    )


def _venv_python(venv_dir: Path) -> Path:
    """Return the interpreter path for a platform-native virtual environment."""
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _create_venv(work: Path, profile: str) -> tuple[Path, Path]:
    """Create a profile-local venv and return its Python and purelib paths."""
    venv_dir = work / "venv"
    create = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        cwd=work,
        env=_scrubbed_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if create.returncode != 0:
        raise RuntimeError(f"[{profile}] could not create isolated venv:\n{create.stdout}\n{create.stderr}")
    python = _venv_python(venv_dir)
    purelib = subprocess.run(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        cwd=work,
        env=_scrubbed_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if purelib.returncode != 0:
        raise RuntimeError(f"[{profile}] could not locate venv site-packages:\n{purelib.stderr}")
    return python, Path(purelib.stdout.strip())


def _run_profile(wheel: Path, profile: str, version: str) -> None:
    spec = PROFILES[profile]
    print(f"[{profile}] installing wheel{'[%s]' % spec['extra'] if spec['extra'] else ''} into isolated target ...")
    with tempfile.TemporaryDirectory(prefix=f"fincore-wheel-{profile}-") as work:
        work_path = Path(work)
        install_mode = spec.get("install_mode", "target")
        if install_mode == "venv":
            install_python, target = _create_venv(work_path, profile)
        else:
            install_python = Path(sys.executable)
            target = work_path / "target"
            target.mkdir()
        consumer = Path(work) / "consumer.py"
        consumer.write_text(_CONSUMER, encoding="utf-8")
        cwd = Path(work) / "cwd"
        cwd.mkdir()

        install_spec = str(wheel) if spec["extra"] is None else f"fincore[{spec['extra']}] @ {wheel.resolve().as_uri()}"
        install_command = [str(install_python), "-m", "pip", "install", "--quiet"]
        if install_mode == "target":
            install_command.extend(("--target", str(target)))
        install_command.append(install_spec)
        try:
            install = subprocess.run(
                install_command,
                cwd=cwd,
                env=_scrubbed_env(),
                capture_output=True,
                text=True,
                timeout=spec.get("install_timeout", 600),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"[{profile}] pip install exceeded its bounded timeout") from exc
        if install.returncode != 0:
            raise RuntimeError(f"[{profile}] pip install failed:\n{install.stdout}\n{install.stderr}")

        if spec.get("pip_check"):
            pip_check = subprocess.run(
                [str(install_python), "-m", "pip", "check"],
                cwd=cwd,
                env=_scrubbed_env(),
                capture_output=True,
                text=True,
                timeout=spec.get("pip_check_timeout", 120),
            )
            if pip_check.returncode != 0:
                raise RuntimeError(f"[{profile}] pip check failed:\n{pip_check.stdout}\n{pip_check.stderr}")

        try:
            run = subprocess.run(
                [str(install_python), "-S", "-E", str(consumer), str(target), version, profile, _payload(profile)],
                cwd=cwd,
                env={**_scrubbed_env(), "MPLBACKEND": "Agg"},
                capture_output=True,
                text=True,
                timeout=spec.get("consumer_timeout", 300),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"[{profile}] isolated consumer exceeded its bounded timeout") from exc
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
