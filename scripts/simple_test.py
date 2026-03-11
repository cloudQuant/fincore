#!/usr/bin/env python3
"""Simplified test runner to verify imports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import fincore

# Import and test fincore imports
print("Importing fincore.metrics.ratios...")
from fincore.metrics.ratios import excess_sharpe, sharpe_ratio, sortino_ratio

print("Sharpe ratio available:", hasattr(fincore, "sharpe_ratio"))

print("\nImporting fincore.metrics.risk...")
from fincore.metrics.risk import annual_volatility, downside_risk, tail_ratio, tracking_error

print("Annual volatility available:", hasattr(fincore, "annual_volatility"))

print("\nTesting function calls...")
result = sharpe_ratio([0.01] * 252)
print(f"Sharpe ratio result: {result}")

result = annual_volatility([0.01] * 252)
print(f"Annual volatility result: {result}")
print("\n\nAll imports successful!")
