"""Independent risk-domain oracles.

Every oracle in this package is a plain NumPy/SciPy reference that never
imports ``fincore``, so it is genuinely independent of the implementation
under test.  See ``docs/quality/numerical-oracle-register.md`` for formulas,
sources, units and tolerances.
"""
