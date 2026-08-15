"""Offline, deterministic factor-analysis migration quickstart.

Run this example with the plotting extra installed:

    MPLBACKEND=Agg python examples/factor_analysis_quickstart.py

The example creates only in-memory synthetic data.  It makes no network
requests and does not write files; figures returned by the summary tear sheet
are closed explicitly after inspection.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fincore.alphalens import utils as strict_utils
from fincore.factor_analysis import (
    PyfolioFactorInputs,
    analyze_factor,
    close_owned_figures,
    create_summary_tear_sheet,
    prepare_factor_data,
)

if TYPE_CHECKING:
    from fincore.factor_analysis.models import FactorAnalysisModel
    from fincore.factor_analysis.tears import FactorTearSheetArtifacts

# The enhanced renderer uses this documented, optional distribution extra.
OPTIONAL_EXTRA_INSTALL = "fincore[alphalens]"


def synthetic_factor_inputs(seed: int = 7) -> tuple[pd.Series, pd.DataFrame]:
    """Build a fixed local factor panel and price matrix for this quickstart."""

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=9, freq="B", name="date")
    assets = pd.Index(("A", "B", "C", "D", "E"), name="asset")
    factor_index = pd.MultiIndex.from_product((dates[:-1], assets), names=("date", "asset"))
    factor = pd.Series(rng.normal(size=len(factor_index)), index=factor_index, name="factor")
    daily_returns = pd.DataFrame(rng.normal(0.0005, 0.01, size=(len(dates), len(assets))), index=dates, columns=assets)
    prices = 100.0 * (1.0 + daily_returns).cumprod()
    return factor, prices


def strict_quickstart() -> pd.DataFrame:
    """Use the source-shaped Alphalens facade with an explicit one-day period."""

    factor, prices = synthetic_factor_inputs()
    return strict_utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1,),
        filter_zscore=None,
        max_loss=0.35,
    )


def enhanced_prepare_and_analyze() -> tuple[object, FactorAnalysisModel]:
    """Use the enhanced prepare-once and analyze-once workflow for new code."""

    factor, prices = synthetic_factor_inputs()
    prepared = prepare_factor_data(
        factor,
        prices,
        periods=(1,),
        filter_zscore=None,
        max_loss=0.35,
    )
    model = analyze_factor(
        prepared.data,
        periods=("1D",),
        turnover_periods=(1,),
        include_pyfolio=True,
    )
    return prepared, model


def pyfolio_bridge() -> PyfolioFactorInputs:
    """Return typed bridge inputs; Pyfolio itself is not invoked or required."""

    _, model = enhanced_prepare_and_analyze()
    assert model.pyfolio_inputs is not None
    return model.pyfolio_inputs


def summary_tear_sheet() -> FactorTearSheetArtifacts:
    """Render a summary with Agg, then close the caller-owned figures."""

    # Set before Matplotlib's lazy import without changing a caller-provided backend.
    os.environ.setdefault("MPLBACKEND", "Agg")
    _, model = enhanced_prepare_and_analyze()
    artifacts = create_summary_tear_sheet(model, show=False)
    close_owned_figures(artifacts)
    return artifacts


def main() -> None:
    """Run every in-memory step and print only a compact console summary."""

    strict = strict_quickstart()
    prepared, model = enhanced_prepare_and_analyze()
    inputs = pyfolio_bridge()
    artifacts = summary_tear_sheet()
    print(
        {
            "strict_rows": len(strict),
            "enhanced_rows": len(prepared.data),
            "forward_periods": model.forward_periods,
            "pyfolio_position_columns": tuple(inputs.positions.columns),
            "summary_tables": tuple(artifacts.tables),
        }
    )


if __name__ == "__main__":
    main()
