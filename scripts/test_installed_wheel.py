#!/usr/bin/env python
"""Install a built wheel into isolated consumers and smoke direct capabilities.

Every profile names one 0.5 capability extra.  The consumer starts with
``-S -E`` and receives only the profile target on ``sys.path`` so an editable
checkout or ambient site-packages cannot satisfy an import accidentally.
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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

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
CANONICAL_ROOT_EXPORTS = (
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
)


def _profile(
    extra: str | None,
    *,
    imports: tuple[str, ...] = (),
    smoke: str = "core",
    forbid_optional: bool = False,
    install_mode: str = "target",
    install_timeout: int = 300,
    consumer_timeout: int = 120,
    pip_check: bool = False,
) -> dict[str, Any]:
    return {
        "extra": extra,
        "imports": imports,
        "smoke": smoke,
        "forbidden_imports": OPTIONAL_ROOTS if forbid_optional else (),
        "install_mode": install_mode,
        "install_timeout": install_timeout,
        "consumer_timeout": consumer_timeout,
        "pip_check": pip_check,
        "pip_check_timeout": 120,
    }


PROFILES: dict[str, dict[str, Any]] = {
    "core": _profile(None, forbid_optional=True),
    "factor-analysis": _profile("factor-analysis", imports=("statsmodels",), smoke="factor"),
    "visualization": _profile(
        "visualization",
        imports=("matplotlib", "seaborn", "IPython", "plotly", "bokeh"),
        smoke="visualization",
        install_timeout=600,
        consumer_timeout=180,
    ),
    "interactive": _profile("interactive", imports=("plotly", "bokeh"), smoke="interactive", install_timeout=600),
    "report-pdf": _profile("report-pdf", imports=("playwright", "PyPDF2"), install_timeout=600),
    "report-xlsx": _profile("report-xlsx", imports=("openpyxl",), smoke="xlsx"),
    "bayesian": _profile("bayesian", imports=("pymc",), install_timeout=900, consumer_timeout=180),
    "data-yahoo": _profile("data-yahoo", imports=("yfinance",)),
    "data-alphavantage": _profile("data-alphavantage", imports=("requests",)),
    "data-pandas-datareader": _profile("data-pandas-datareader", imports=("pandas_datareader",)),
    "data-cn": _profile("data-cn", imports=("tushare", "akshare"), install_timeout=900, consumer_timeout=180),
    "all": _profile(
        "all",
        imports=OPTIONAL_ROOTS,
        smoke="visualization",
        install_mode="venv",
        install_timeout=900,
        consumer_timeout=300,
        pip_check=True,
    ),
}
DEFAULT_PROFILES = ("core", "factor-analysis", "visualization", "report-pdf", "report-xlsx", "bayesian", "all")
DATA_PROVIDER_PROFILES = ("data-yahoo", "data-alphavantage", "data-pandas-datareader", "data-cn")

_CONSUMER = textwrap.dedent(
    """
    import importlib
    import json
    import os
    import sys
    from pathlib import Path

    target, expected_version, profile = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    payload = json.loads(sys.argv[4])
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(target))

    import fincore

    assert Path(fincore.__file__).resolve().is_relative_to(target.resolve()), fincore.__file__
    assert fincore.__version__ == expected_version, (fincore.__version__, expected_version)
    assert tuple(fincore.__all__) == tuple(payload["canonical_root_exports"])

    namespace = {}
    exec("from fincore import *", namespace)
    assert set(payload["canonical_root_exports"]) <= set(namespace)
    for legacy_name in ("Empyrical", "Pyfolio", "sharpe_ratio", "analyze"):
        assert legacy_name not in namespace
    for root in payload["optional_roots"]:
        assert root not in sys.modules, f"root import pulled optional module {root}"

    for module_name in payload["core"]:
        module = importlib.import_module(module_name)
        assert Path(module.__file__).resolve().is_relative_to(target.resolve()), module.__file__
    for module_name in payload["imports"]:
        module = importlib.import_module(module_name)
        assert Path(module.__file__).resolve().is_relative_to(target.resolve()), module.__file__
    for module_name in payload["forbidden_imports"]:
        assert module_name not in sys.modules, f"{module_name} loaded in core profile"

    import numpy as np
    import pandas as pd
    from fincore.metrics.drawdown import max_drawdown
    from fincore.metrics.ratios import sharpe_ratio

    dates = pd.date_range("2024-01-02", periods=32, freq="B")
    returns = pd.Series(np.resize([0.01, -0.004, 0.002, 0.003], len(dates)), index=dates)
    assert np.isfinite(sharpe_ratio(returns))
    assert max_drawdown(returns) <= 0

    def report_document():
        from fincore.report.portfolio.compute import build_portfolio_report

        return build_portfolio_report(returns, rolling_window=8)

    smoke = payload["smoke"]
    if smoke == "factor":
        from fincore.factor_analysis.data import prepare_factor_data

        assets = ("A", "B", "C")
        index = pd.MultiIndex.from_product((dates[:-2], assets), names=("date", "asset"))
        factor = pd.Series(np.resize([0.0, 1.0, 2.0], len(index)), index=index)
        prices = pd.DataFrame(
            {asset: 100.0 + number + np.arange(len(dates)) for number, asset in enumerate(assets)}, index=dates
        )
        prepared = prepare_factor_data(factor, prices, periods=(1,), quantiles=3, max_loss=1.0)
        assert not prepared.data.empty
    elif smoke == "visualization":
        import matplotlib.pyplot as plt
        from fincore.report.renderers.matplotlib import render_matplotlib

        bundle = render_matplotlib(report_document())
        assert bundle.named_artifacts
        bundle.close()
        plt.close("all")
    elif smoke == "interactive":
        from fincore.report.renderers.interactive import render_bokeh, render_plotly

        assert render_plotly(report_document()).named_artifacts["figure"].data
        assert render_bokeh(report_document()).named_artifacts["figure"].renderers
    elif smoke == "xlsx":
        from fincore.report.renderers.xlsx import write_xlsx

        output = Path("report.xlsx")
        assert write_xlsx(report_document(), output).named_artifacts["file"].is_file()

    print(f"PROFILE {profile} OK")
    """
)


def _pyproject_version() -> str:
    with PYPROJECT.open("rb") as fh:
        return str(tomllib.load(fh)["project"]["version"])


def _scrubbed_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}


def _payload(profile: str) -> str:
    spec = PROFILES[profile]
    return json.dumps(
        {
            "canonical_root_exports": CANONICAL_ROOT_EXPORTS,
            "core": CORE_THIRD_PARTY,
            "forbidden_imports": spec["forbidden_imports"],
            "imports": spec["imports"],
            "optional_roots": OPTIONAL_ROOTS,
            "smoke": spec["smoke"],
        }
    )


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _create_venv(work: Path, profile: str) -> tuple[Path, Path]:
    venv_dir = work / "venv"
    created = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        cwd=work,
        env=_scrubbed_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if created.returncode != 0:
        raise RuntimeError(f"[{profile}] could not create isolated venv:\n{created.stdout}\n{created.stderr}")
    python = _venv_python(venv_dir)
    purelib = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
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
    with tempfile.TemporaryDirectory(prefix=f"fincore-wheel-{profile}-") as work:
        work_path = Path(work)
        install_mode = str(spec["install_mode"])
        if install_mode == "venv":
            install_python, target = _create_venv(work_path, profile)
        else:
            install_python = Path(sys.executable)
            target = work_path / "target"
            target.mkdir()
        consumer = work_path / "consumer.py"
        consumer.write_text(_CONSUMER, encoding="utf-8")
        cwd = work_path / "cwd"
        cwd.mkdir()

        extra = spec["extra"]
        install_spec = str(wheel) if extra is None else f"fincore[{extra}] @ {wheel.resolve().as_uri()}"
        command = [str(install_python), "-m", "pip", "install", "--quiet"]
        if install_mode == "target":
            command.extend(("--target", str(target)))
        command.append(install_spec)
        try:
            installed = subprocess.run(
                command,
                cwd=cwd,
                env=_scrubbed_env(),
                capture_output=True,
                text=True,
                timeout=int(spec["install_timeout"]),
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"[{profile}] pip install exceeded its bounded timeout") from error
        if installed.returncode != 0:
            raise RuntimeError(f"[{profile}] pip install failed:\n{installed.stdout}\n{installed.stderr}")

        if spec["pip_check"]:
            checked = subprocess.run(
                [str(install_python), "-m", "pip", "check"],
                cwd=cwd,
                env=_scrubbed_env(),
                capture_output=True,
                text=True,
                timeout=int(spec["pip_check_timeout"]),
            )
            if checked.returncode != 0:
                raise RuntimeError(f"[{profile}] pip check failed:\n{checked.stdout}\n{checked.stderr}")

        try:
            consumed = subprocess.run(
                [str(install_python), "-S", "-E", str(consumer), str(target), version, profile, _payload(profile)],
                cwd=cwd,
                env={**_scrubbed_env(), "MPLBACKEND": "Agg"},
                capture_output=True,
                text=True,
                timeout=int(spec["consumer_timeout"]),
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"[{profile}] isolated consumer exceeded its bounded timeout") from error
        if consumed.returncode != 0:
            raise RuntimeError(f"[{profile}] fresh consumer failed:\n{consumed.stdout}\n{consumed.stderr}")
        print(f"[{profile}] {consumed.stdout.strip()}")


def _selected_profiles(profile_names: list[str], data_providers: list[str]) -> list[str]:
    selected = list(profile_names)
    providers = DATA_PROVIDER_PROFILES if data_providers == ["all"] else tuple(data_providers)
    for profile in providers:
        if profile not in selected:
            selected.append(profile)
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", required=True, help="directory containing the built wheel")
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=list(DEFAULT_PROFILES))
    parser.add_argument("--data-providers", nargs="+", choices=("all", *DATA_PROVIDER_PROFILES), default=[])
    args = parser.parse_args(argv)

    version = _pyproject_version()
    dist = Path(args.dist).resolve()
    wheels = sorted(dist.glob(f"fincore-{version}-*.whl"))
    if not wheels:
        print(f"no wheel matching fincore-{version}-*.whl in {args.dist}", file=sys.stderr)
        return 1

    selected = _selected_profiles(args.profiles, args.data_providers)
    failures = 0
    for profile in selected:
        try:
            _run_profile(wheels[0], profile, version)
        except RuntimeError as error:
            failures += 1
            print(f"[{profile}] FAILED: {error}", file=sys.stderr)
    print(f"{len(selected) - failures}/{len(selected)} profiles passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
