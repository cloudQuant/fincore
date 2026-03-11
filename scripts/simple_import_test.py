#!/usr/bin/env python3
"""Simple import test."""

import subprocess
import sys
from pathlib import Path

# Add tests to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

# Test import
print("Testing import from fincore.metrics.ratios...")
from fincore.metrics.ratios import sharpe_ratio

print("Result:", sharpe_ratio([0.01] * 252))

sys.exit(0)
