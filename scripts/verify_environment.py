#!/usr/bin/env python3
"""Verify fincore environment setup.

This script checks that all required dependencies are properly installed
and the environment is correctly configured.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version meets minimum requirement."""
    print("✓ Checking Python version...")
    version = sys.version_info
    if version < (3, 11):
        print(f"  ✗ Python {version.major}.{version.minor} found")
        print(f"  ✗ Python 3.11+ required")
        return False
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check all required dependencies are installed."""
    print("\n✓ Checking dependencies...")

    required = {
        "numpy": "1.17.0",
        "pandas": "0.25.0",
        "scipy": "1.3.0",
        "pytz": "2023.3",
        "packaging": "21.0",
    }

    optional = {
        "matplotlib": "3.3",
        "seaborn": "0.11",
        "ipython": "7.0",
    }

    all_ok = True

    # Check required dependencies
    for package, min_version in required.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"  ✓ {package} {version}")
        except ImportError:
            print(f"  ✗ {package} NOT INSTALLED (required >= {min_version})")
            all_ok = False

    # Check optional dependencies
    print("\n✓ Checking optional dependencies...")
    for package, min_version in optional.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"  ✓ {package} {version}")
        except ImportError:
            print(f"  ⚠ {package} not installed (optional, >= {min_version})")

    return all_ok


def check_fincore_imports():
    """Check fincore can be imported correctly."""
    print("\n✓ Checking fincore imports...")

    try:
        # Use subprocess to avoid path issues
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from fincore import sharpe_ratio, max_drawdown; print('OK')",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        if result.returncode == 0:
            print("  ✓ Key functions imported")
            return True
        else:
            print(f"  ✗ Import error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False


def check_test_framework():
    """Check test framework is available."""
    print("\n✓ Checking test framework...")

    try:
        import pytest

        print(f"  ✓ pytest {pytest.__version__}")

        # Check key pytest plugins
        plugins = ["pytest_cov", "pytest_xdist"]
        for plugin in plugins:
            try:
                __import__(plugin)
                print(f"  ✓ {plugin} installed")
            except ImportError:
                print(f"  ⚠ {plugin} not installed")

        return True
    except ImportError:
        print("  ✗ pytest NOT INSTALLED")
        return False


def main():
    """Run all environment checks."""
    print("=" * 70)
    print("Fincore Environment Verification")
    print("=" * 70)

    results = []

    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Fincore Imports", check_fincore_imports()))
    results.append(("Test Framework", check_test_framework()))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    for check_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:.<50} {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n✅ All checks passed! Environment is correctly configured.")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        print("\nTo install dependencies:")
        print("  pip install -e '.[dev,viz]'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
