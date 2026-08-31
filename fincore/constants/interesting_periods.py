"""Migration-only location for historical market-period data.

The metric timing domain owns this data at
:mod:`fincore.metrics._historical_periods`.  The module is removed at the
atomic 0.5 cutover together with ``fincore.constants``.
"""

from fincore.metrics._historical_periods import PERIODS

__all__ = ["PERIODS"]
