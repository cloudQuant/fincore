"""Migration-only import location for the pre-0.5 time-series contract.

The canonical implementation lives in :mod:`fincore.runtime.time_series`.
This module is retained only while the pre-cutover oracle sources are present
and is deleted with the rest of ``fincore.contracts`` in the atomic 0.5
cutover.
"""

from fincore.runtime.time_series import (
    AlignmentPolicy,
    align_binary_metric_inputs,
    align_time_series,
    normalize_time_series_timezone,
    validate_time_series_timezones,
)

__all__ = [
    "AlignmentPolicy",
    "align_binary_metric_inputs",
    "align_time_series",
    "normalize_time_series_timezone",
    "validate_time_series_timezones",
]
