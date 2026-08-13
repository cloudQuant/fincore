#!/usr/bin/env python
"""Compatibility shim for legacy ``python setup.py`` invocations.

All project metadata is declared in ``pyproject.toml`` (PEP 621), which is the
single source of truth for packaging.  This module duplicates no metadata; it
only forwards to setuptools so legacy tooling that invokes
``python setup.py <command>`` keeps working.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
