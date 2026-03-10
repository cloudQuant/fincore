#!/usr/bin/env python3
"""Simple import test."""

import sys
import subprocess

# Add tests to path
sys.path.insert(0, "/Users/yunjinqi/Documents/source_code/fincore/tests")

# Test import
print("Testing import from fincore.metrics.ratios...")
from fincore.metrics.ratios import sharpe_ratio

print("Result:", sharpe_ratio([0.01] * 252))

sys.exit(0)
