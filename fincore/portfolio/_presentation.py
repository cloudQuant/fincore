"""Portfolio-owned presentation vocabulary for positions and round trips."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

SECTORS = OrderedDict(
    [
        (101, "Basic Materials"),
        (102, "Consumer Cyclical"),
        (103, "Financial Services"),
        (104, "Real Estate"),
        (205, "Consumer Defensive"),
        (206, "Healthcare"),
        (207, "Utilities"),
        (308, "Communication Services"),
        (309, "Energy"),
        (310, "Industrials"),
        (311, "Technology"),
    ]
)

CAP_BUCKETS = OrderedDict(
    [
        ("Micro", (50_000_000, 300_000_000)),
        ("Small", (300_000_000, 2_000_000_000)),
        ("Mid", (2_000_000_000, 10_000_000_000)),
        ("Large", (10_000_000_000, 200_000_000_000)),
        ("Mega", (200_000_000_000, np.inf)),
    ]
)

PNL_STATS = OrderedDict(
    [
        ("Total profit", lambda value: value.sum()),
        ("Gross profit", lambda value: value[value > 0].sum()),
        ("Gross loss", lambda value: value[value < 0].sum()),
        (
            "Profit factor",
            lambda value: (
                value[value > 0].sum() / value[value < 0].abs().sum() if value[value < 0].abs().sum() != 0 else np.nan
            ),
        ),
        ("Avg. trade net profit", "mean"),
        ("Avg. winning trade", lambda value: value[value > 0].mean()),
        ("Avg. losing trade", lambda value: value[value < 0].mean()),
        (
            "Ratio Avg. Win:Avg. Loss",
            lambda value: (
                value[value > 0].mean() / value[value < 0].abs().mean()
                if value[value < 0].abs().mean() != 0
                else np.nan
            ),
        ),
        ("Largest winning trade", "max"),
        ("Largest losing trade", "min"),
    ]
)

SUMMARY_STATS = OrderedDict(
    [
        ("Total number of round_trips", "count"),
        ("Percent profitable", lambda value: len(value[value > 0]) / float(len(value))),
        ("Winning round_trips", lambda value: len(value[value > 0])),
        ("Losing round_trips", lambda value: len(value[value < 0])),
        ("Even round_trips", lambda value: len(value[value == 0])),
    ]
)

RETURN_STATS = OrderedDict(
    [
        ("Avg returns all round_trips", lambda value: value.mean()),
        ("Avg returns winning", lambda value: value[value > 0].mean()),
        ("Avg returns losing", lambda value: value[value < 0].mean()),
        ("Median returns all round_trips", lambda value: value.median()),
        ("Median returns winning", lambda value: value[value > 0].median()),
        ("Median returns losing", lambda value: value[value < 0].median()),
        ("Largest winning trade", "max"),
        ("Largest losing trade", "min"),
    ]
)

DURATION_STATS = OrderedDict(
    [
        ("Avg duration", lambda value: value.mean()),
        ("Median duration", lambda value: value.median()),
        ("Longest duration", lambda value: value.max()),
        ("Shortest duration", lambda value: value.min()),
    ]
)


def _duration_span_days(group: pd.DataFrame) -> float:
    if "open_dt" not in group.columns or "close_dt" not in group.columns:
        return 0.0
    span = group["close_dt"].max() - group["open_dt"].min()
    if pd.isna(span):
        return 0.0
    return max(float(span.total_seconds()) / 86_400.0, 0.0)


DURATION_STATS_GROUP = OrderedDict(
    [
        (
            "Avg # round_trips per day",
            lambda value, group: (
                float(len(value)) / _duration_span_days(group)
                if len(value) > 0 and _duration_span_days(group) > 0
                else np.nan
            ),
        ),
        (
            "Avg # round_trips per month",
            lambda value, group: (
                float(len(value)) / (_duration_span_days(group) / 21.0)
                if len(value) > 0 and _duration_span_days(group) > 0
                else np.nan
            ),
        ),
    ]
)
