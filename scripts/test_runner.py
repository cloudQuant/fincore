#!/usr/bin/env python3
"""Test runner that reloads fincore before running tests."""

import sys
import importlib
import subprocess

# Add tests directory to path
sys.path.insert(0, "/Users/yunjinqi/Documents/source_code/fincore/tests")

# Force reload fincore to pick up __all__ changes
if "fincore" in sys.modules:
    del sys.modules["fincore"]

import fincore

# Reload all submodules
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("fincore."):
        del sys.modules[mod_name]

# Re-import
import fincore.metrics.ratios

# Now run the test
result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_metrics/test_ratios_complete.py::TestSharpeRatioEdgeCases::test_sharpe_normal_case",
        "-v",
        "--tb=short",
    ],
    capture_output=True,
    text=True,
)

print(result.stdout)
print(result.stderr)
