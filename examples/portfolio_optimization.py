"""Direct allocation-optimisation example."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fincore.optimization.frontier import efficient_frontier
from fincore.optimization.objectives import optimize
from fincore.optimization.risk_parity import risk_parity


def inputs() -> pd.DataFrame:
    """Create deterministic multi-asset daily returns."""

    random = np.random.default_rng(42)
    return pd.DataFrame(
        random.normal(loc=[0.0005, 0.0003, 0.0007], scale=[0.012, 0.008, 0.015], size=(252, 3)),
        index=pd.date_range("2024-01-02", periods=252, freq="B"),
        columns=("equity", "bonds", "alternatives"),
    )


def main() -> None:
    """Run each direct allocation operation and print its canonical result."""

    returns = inputs()
    frontier = efficient_frontier(returns, n_points=8)
    parity = risk_parity(returns)
    maximum_sharpe = optimize(returns, objective="max_sharpe")

    print("Frontier returns:", frontier["frontier_returns"])
    print("Risk-parity weights:", parity["weights"])
    print("Maximum-Sharpe weights:", maximum_sharpe["weights"])


if __name__ == "__main__":
    main()
