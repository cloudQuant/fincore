from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import pandas as pd
import pytest

# These are real plotting-chain tests.  Fix the backend before importing
# ``fincore.pyfolio`` so they remain headless and deterministic in CI.
matplotlib.use("Agg", force=True)


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
