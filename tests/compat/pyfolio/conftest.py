from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import pytest

# These are real plotting-chain tests.  Fix the backend before importing
# ``fincore.pyfolio`` so they remain headless and deterministic in CI.
matplotlib.use("Agg", force=True)


PYFOLIO_MANIFEST = Path(__file__).parents[1] / "fixtures" / "pyfolio-0.9.6-api.json"
PACKAGE_ROOT = Path(__file__).parents[3] / "fincore"


@dataclass(frozen=True)
class ImportProbe:
    backend_before: str
    backend_after: str
    eager_optional_modules: tuple[str, ...]
    private_implementation_loaded: bool

    @property
    def backend_unchanged(self) -> bool:
        return self.backend_before == self.backend_after


def load_pyfolio_profile() -> list[dict[str, Any]]:
    data = json.loads(PYFOLIO_MANIFEST.read_text(encoding="utf-8"))
    return [dict(entry, name=name) for name, entry in data["compatibility_profile"].items()]


def run_isolated_import_probe(
    module_name: str = "fincore.pyfolio",
    public_names: tuple[str, ...] = (),
) -> ImportProbe:
    script = r"""
import inspect
import json
import sys
import matplotlib

before_backend = matplotlib.get_backend()
before_modules = set(sys.modules)
module = __import__("MODULE_NAME", fromlist=["*"])
for public_name in PUBLIC_NAMES:
    inspect.signature(getattr(module, public_name))
after_backend = matplotlib.get_backend()
heavy_prefixes = (
    "fincore._pyfolio_impl",
    "fincore.tearsheets",
    "matplotlib.pyplot",
    "IPython",
    "scipy",
    "seaborn",
    "statsmodels",
)
new_modules = set(sys.modules) - before_modules
eager = sorted(
    name for name in new_modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in heavy_prefixes)
)
print(json.dumps({
    "backend_before": before_backend,
    "backend_after": after_backend,
    "eager_optional_modules": eager,
    "private_implementation_loaded": "fincore._pyfolio_impl" in sys.modules,
}))
""".replace("MODULE_NAME", module_name).replace("PUBLIC_NAMES", repr(public_names))
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PACKAGE_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return ImportProbe(
        backend_before=payload["backend_before"],
        backend_after=payload["backend_after"],
        eager_optional_modules=tuple(payload["eager_optional_modules"]),
        private_implementation_loaded=payload["private_implementation_loaded"],
    )


def hash_tracked_package_files() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        hashes[str(path.relative_to(PACKAGE_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


@pytest.fixture
def workflow_returns() -> pd.Series:
    index = pd.date_range("2024-01-02", periods=260, freq="B", tz="UTC")
    values = [0.004 if i % 17 else -0.025 for i in range(len(index))]
    return pd.Series(values, index=index, name="returns")


@pytest.fixture
def short_drawdown_returns() -> pd.Series:
    index = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")
    index = pd.DatetimeIndex(index.tolist())
    return pd.Series([0.02, -0.05, 0.005, 0.005, 0.005], index=index, name="returns")


@dataclass(frozen=True)
class PyfolioRiskInputs:
    positions: pd.DataFrame
    sectors: pd.DataFrame
    caps: pd.DataFrame
    shares_held: pd.DataFrame
    volumes: pd.DataFrame
    returns: pd.Series
    percentile: float


@pytest.fixture
def pyfolio_risk_inputs() -> PyfolioRiskInputs:
    """Three dates and four assets deliberately exercise false unpacking.

    The current broken implementation returns a four-column DataFrame for
    sector/cap computations and a three-row Series for volume computation.
    Python can unpack those objects as if they were the promised 4/4/3
    compatibility tuples, so neither dimension may be changed casually.
    """

    index = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC")
    assets = ["AAA", "BBB", "CCC", "DDD"]

    positions = pd.DataFrame(
        [
            [60.0, -20.0, 10.0, -10.0, 60.0],
            [-30.0, 10.0, 20.0, 0.0, 100.0],
            [0.0, 40.0, -20.0, 10.0, 70.0],
        ],
        index=index,
        columns=[*assets, "cash"],
    )

    # Deliberately use a different column order from positions.  Computation
    # must align by labels, never by physical column position.
    sectors = pd.DataFrame(
        {
            "DDD": [101, 101, 101],
            "BBB": [309, 309, 309],
            "AAA": [311, 311, 311],
            "CCC": [311, 311, 311],
        },
        index=index,
    )
    caps = pd.DataFrame(
        {
            "CCC": [5.0e9, 5.2e9, 5.4e9],
            "AAA": [1.0e11, 1.1e11, 1.2e11],
            "DDD": [1.0e8, 1.1e8, 1.2e8],
            "BBB": [1.0e9, 1.1e9, 1.2e9],
        },
        index=index,
    )
    shares_held = pd.DataFrame(
        [
            [100.0, -50.0, 0.0, 20.0],
            [40.0, -120.0, 60.0, 0.0],
            [10.0, -20.0, 50.0, -100.0],
        ],
        index=index,
        columns=assets,
    )
    volumes = pd.DataFrame(
        {
            "DDD": [100.0, 200.0, 400.0],
            "BBB": [1000.0, 800.0, 500.0],
            "AAA": [1000.0, 400.0, 200.0],
            "CCC": [100.0, 300.0, 250.0],
        },
        index=index,
    )
    returns = pd.Series([0.01, -0.005, 0.002], index=index, name="returns")

    return PyfolioRiskInputs(
        positions=positions,
        sectors=sectors,
        caps=caps,
        shares_held=shares_held,
        volumes=volumes,
        returns=returns,
        percentile=0.5,
    )
