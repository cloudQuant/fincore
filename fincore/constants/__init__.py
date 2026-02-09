"""Top-level constants exposed by fincore."""

from .periods import (
    APPROX_BDAYS_PER_MONTH,
    APPROX_BDAYS_PER_YEAR,
    MONTHS_PER_YEAR,
    WEEKS_PER_YEAR,
    QTRS_PER_YEAR,
    DAILY,
    WEEKLY,
    MONTHLY,
    QUARTERLY,
    YEARLY,
    ANNUALIZATION_FACTORS,
    PERIOD_TO_FREQ,
)
from .interesting_periods import PERIODS
from .style import (
    PERF_ATTRIB_TURNOVER_THRESHOLD,
    SECTORS,
    CAP_BUCKETS,
    SIMPLE_STAT_FUNCS,
    FACTOR_STAT_FUNCS,
    STAT_FUNC_NAMES,
    FACTOR_PARTITIONS,
    STAT_FUNCS_PCT,
    PNL_STATS,
    SUMMARY_STATS,
    RETURN_STATS,
    DURATION_STATS,
    MM_DISPLAY_UNIT,
)
from .color import COLORMAP, COLORS

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
