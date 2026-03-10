#!/usr/bin/env python
"""Quick verification of code quality fixes (ratios, risk, edge cases).

Run from project root: python scripts/verify_quality_fixes.py
Or: PYTHONPATH=. python scripts/verify_quality_fixes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when run as script
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd


# Avoid heavy imports at top level
def _verify():
    from fincore.metrics.ratios import (
        conditional_sharpe_ratio,
        excess_sharpe,
        information_ratio,
    )
    from fincore.metrics.risk import annual_volatility, value_at_risk

    errors = []

    # 1. Zero tracking error -> NaN (information_ratio, excess_sharpe)
    r = pd.Series([0.01, 0.02, 0.03])
    f = pd.Series([0.01, 0.02, 0.03])
    if not np.isnan(information_ratio(r, f)):
        errors.append("information_ratio: expected NaN for zero tracking error")
    if not np.isnan(excess_sharpe(r.values, f.values)):
        errors.append("excess_sharpe: expected NaN for zero tracking error")

    # 2. conditional_sharpe invalid cutoff
    if not np.isnan(conditional_sharpe_ratio(r.values, cutoff=0)):
        errors.append("conditional_sharpe: expected NaN for cutoff=0")
    if not np.isnan(conditional_sharpe_ratio(r.values, cutoff=1)):
        errors.append("conditional_sharpe: expected NaN for cutoff=1")

    # 3. VaR normal path
    v = value_at_risk(r.values, 0.05)
    if not np.isfinite(v):
        errors.append("value_at_risk: expected finite for valid cutoff")

    # 4. annual_volatility alpha_ param
    vol = annual_volatility(r.values, alpha_=2.0)
    if not np.isfinite(vol):
        errors.append("annual_volatility: expected finite with alpha_=2.0")

    return errors


def main() -> int:
    errors = _verify()
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("OK: All quality fix verifications passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
