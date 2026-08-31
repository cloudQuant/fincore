"""Walk-forward risk validation using direct risk-domain APIs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fincore.risk.diagnostics import walk_forward_var
from fincore.risk.report import build_risk_validation_report
from fincore.risk.specs import RiskModelSpec


def main() -> None:
    """Compute a deterministic walk-forward VaR audit and print its status."""

    returns = pd.Series(
        np.linspace(-0.02, 0.02, 60),
        index=pd.date_range("2024-01-02", periods=60, freq="B", tz="UTC"),
        name="strategy",
    )
    spec = RiskModelSpec(confidence_level=0.95, distribution="normal", window=40, refit_cadence=5)
    walk_forward = walk_forward_var(returns, spec)
    report = build_risk_validation_report(walk_forward)
    print(f"Risk-validation status: {report.status}")
    print(f"Input digest: {walk_forward.inputs_digest}")


if __name__ == "__main__":
    main()
