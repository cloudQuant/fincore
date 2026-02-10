"""Top-level constants exposed by fincore."""

from .color import COLORMAP, COLORS
from .interesting_periods import PERIODS
from .periods import (
    ANNUALIZATION_FACTORS,
    APPROX_BDAYS_PER_MONTH,
    APPROX_BDAYS_PER_YEAR,
    DAILY,
    MONTHLY,
    MONTHS_PER_YEAR,
    PERIOD_TO_FREQ,
    QTRS_PER_YEAR,
    QUARTERLY,
    WEEKLY,
    WEEKS_PER_YEAR,
    YEARLY,
)
from .style import (
    CAP_BUCKETS,
    DURATION_STATS,
    FACTOR_PARTITIONS,
    FACTOR_STAT_FUNCS,
    MM_DISPLAY_UNIT,
    PERF_ATTRIB_TURNOVER_THRESHOLD,
    PNL_STATS,
    RETURN_STATS,
    SECTORS,
    SIMPLE_STAT_FUNCS,
    STAT_FUNC_NAMES,
    STAT_FUNCS_PCT,
    SUMMARY_STATS,
)

__all__ = [
    # periods
    "APPROX_BDAYS_PER_MONTH",
    "APPROX_BDAYS_PER_YEAR",
    "MONTHS_PER_YEAR",
    "WEEKS_PER_YEAR",
    "QTRS_PER_YEAR",
    "DAILY",
    "WEEKLY",
    "MONTHLY",
    "QUARTERLY",
    "YEARLY",
    "ANNUALIZATION_FACTORS",
    "PERIOD_TO_FREQ",
    # interesting_periods
    "PERIODS",
    # style
    "PERF_ATTRIB_TURNOVER_THRESHOLD",
    "SECTORS",
    "CAP_BUCKETS",
    "SIMPLE_STAT_FUNCS",
    "FACTOR_STAT_FUNCS",
    "STAT_FUNC_NAMES",
    "FACTOR_PARTITIONS",
    "STAT_FUNCS_PCT",
    "PNL_STATS",
    "SUMMARY_STATS",
    "RETURN_STATS",
    "DURATION_STATS",
    "MM_DISPLAY_UNIT",
    # color
    "COLORMAP",
    "COLORS",
]
