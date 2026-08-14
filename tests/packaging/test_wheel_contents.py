"""Wheel contents contract tests.

Builds the wheel from the checkout and asserts the artifact layout:

- runtime assets ship (vendored ECharts for offline reports, the report model
  module, ``py.typed``);
- no stray assets: tests/examples/docs never ship, and the only ``.js`` is
  the report asset; no example CSV/XLSX anywhere;
- METADATA carries the single-source version, the full extras set, the Beta
  classifier, and no ``Requires-Dist: fincore[...]`` self-dependency.
"""

from __future__ import annotations

import email
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

FUNCTIONAL_EXTRAS = {
    "pyfolio",
    "factor-analysis",
    "alphalens",
    "interactive",
    "report-pdf",
    "report-xlsx",
    "bayesian",
    "data-yahoo",
    "data-alphavantage",
    "data-pandas-datareader",
    "data-cn",
}
ALIAS_EXTRAS = {"viz", "datareader"}


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _pyproject_version() -> str:
    return str(_pyproject()["project"]["version"])


def _build_wheel(out_dir: Path) -> Path:
    """Build a wheel from the checkout; skip only when build tooling is absent."""
    attempts: list[list[str]] = [
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir), str(REPO_ROOT)],
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(out_dir),
            str(REPO_ROOT),
        ],
    ]
    for cmd in attempts:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=REPO_ROOT)
        except FileNotFoundError as exc:  # interpreter missing (should not happen)
            raise AssertionError(f"cannot run {cmd[0]}: {exc}") from exc
        if proc.returncode == 0:
            break
        if "No module named build" in proc.stderr:
            continue  # `build` package not installed; try the pip fallback
        if "Cannot import 'setuptools.build_meta'" in proc.stderr:
            continue  # build backend absent without isolation; tooling unavailable
        raise AssertionError(f"wheel build failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    else:
        pytest.skip("cannot build a wheel: `build` and `pip wheel` are both unavailable")
    wheels = sorted(out_dir.glob("fincore-*.whl"))
    assert wheels, "wheel build produced no artifact"
    return wheels[0]


@pytest.fixture(scope="session")
def wheel_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_wheel(tmp_path_factory.mktemp("fincore-wheel"))


def _names(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as zf:
        return set(zf.namelist())


def _metadata(wheel: Path) -> email.message.Message:
    with zipfile.ZipFile(wheel) as zf:
        dist_info = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        return email.message_from_bytes(zf.read(dist_info))


# ---------------------------------------------------------------------------
# Artifact layout
# ---------------------------------------------------------------------------


def test_runtime_assets_ship_in_wheel(wheel_path: Path) -> None:
    names = _names(wheel_path)
    assert "fincore/report/assets/echarts.min.js" in names, "vendored ECharts asset missing from wheel"
    assert "fincore/report/model.py" in names, "report model module missing from wheel"
    assert "fincore/py.typed" in names, "py.typed marker missing from wheel"
    required_modules = {
        "fincore/alphalens/__init__.py",
        "fincore/alphalens/performance.py",
        "fincore/alphalens/plotting.py",
        "fincore/alphalens/tears.py",
        "fincore/factor_analysis/__init__.py",
        "fincore/factor_analysis/data.py",
        "fincore/factor_analysis/performance.py",
        "fincore/factor_analysis/portfolio.py",
    }
    assert required_modules <= names, f"Alphalens runtime modules missing: {sorted(required_modules - names)}"


def test_wheel_includes_the_approved_apache_license_only(wheel_path: Path) -> None:
    """The artifact carries the project license; notices need a separate approval."""
    with zipfile.ZipFile(wheel_path) as zf:
        license_files = [name for name in zf.namelist() if name.endswith(".dist-info/licenses/LICENSE")]
        assert len(license_files) == 1, f"expected one bundled LICENSE, found {license_files}"
        license_text = zf.read(license_files[0]).decode("utf-8")
    assert "Apache License" in license_text
    assert not [name for name in _names(wheel_path) if "THIRD_PARTY_NOTICES" in name], (
        "third-party notice files require a separate human license decision"
    )


def test_no_stray_assets_in_wheel(wheel_path: Path) -> None:
    names = _names(wheel_path)
    for banned_prefix in ("tests/", "examples/", "docs/", "benchmarks/", "site/"):
        assert not any(n.startswith(banned_prefix) for n in names), f"stray {banned_prefix!r} content in wheel"
    js_files = [n for n in names if n.endswith(".js")]
    assert js_files == ["fincore/report/assets/echarts.min.js"], f"unexpected .js assets: {js_files}"
    data_files = [n for n in names if n.endswith((".csv", ".xlsx", ".ipynb", ".png"))]
    assert not data_files, f"example/notebook data files must not ship in the wheel: {data_files}"
    forbidden_fragments = ("versioneer", "requirements-alphalens", "requirements-empyrical", "requirements-pyfolio")
    assert not [n for n in names if any(fragment in n.lower() for fragment in forbidden_fragments)], (
        "compatibility oracle or Versioneer files must not ship in the wheel"
    )


def test_wheel_filename_embeds_single_source_version(wheel_path: Path) -> None:
    assert wheel_path.name.startswith(f"fincore-{_pyproject_version()}-"), (
        f"wheel filename {wheel_path.name} disagrees with pyproject version {_pyproject_version()}"
    )


# ---------------------------------------------------------------------------
# METADATA contract
# ---------------------------------------------------------------------------


def test_metadata_version_matches_pyproject(wheel_path: Path) -> None:
    assert _metadata(wheel_path)["Version"] == _pyproject_version()


def test_metadata_declares_full_extra_set(wheel_path: Path) -> None:
    provides = set(_metadata(wheel_path).get_all("Provides-Extra", []))
    expected = FUNCTIONAL_EXTRAS | ALIAS_EXTRAS | {"dev", "all"}
    assert provides == expected, f"Provides-Extra mismatch: {provides ^ expected}"


def test_metadata_has_no_self_dependency(wheel_path: Path) -> None:
    for req in _metadata(wheel_path).get_all("Requires-Dist", []):
        assert not req.strip().lower().startswith("fincore"), f"self-dependency in wheel metadata: {req!r}"


def test_metadata_classifier_is_beta_not_production(wheel_path: Path) -> None:
    classifiers = _metadata(wheel_path).get_all("Classifier", [])
    assert "Development Status :: 4 - Beta" in classifiers
    assert "Development Status :: 5 - Production/Stable" not in classifiers
