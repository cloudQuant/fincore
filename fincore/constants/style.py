import numpy as np
from collections import OrderedDict

PERF_ATTRIB_TURNOVER_THRESHOLD = 0.25

SECTORS = OrderedDict([
    (101, 'Basic Materials'),
    (102, 'Consumer Cyclical'),
    (103, 'Financial Services'),
    (104, 'Real Estate'),
    (205, 'Consumer Defensive'),
    (206, 'Healthcare'),
    (207, 'Utilities'),
    (308, 'Communication Services'),
    (309, 'Energy'),
    (310, 'Industrials'),
    (311, 'Technology')
])

CAP_BUCKETS = OrderedDict([
    ('Micro', (50000000, 300000000)),
    ('Small', (300000000, 2000000000)),
    ('Mid', (2000000000, 10000000000)),
    ('Large', (10000000000, 200000000000)),
    ('Mega', (200000000000, np.inf))
])

SIMPLE_STAT_FUNCS = [
    ep.annual_return,
    ep.cum_returns_final,
    ep.annual_volatility,
    ep.sharpe_ratio,
    ep.calmar_ratio,
    ep.stability_of_timeseries,
    ep.max_drawdown,
    ep.omega_ratio,
    ep.sortino_ratio,
    stats.skew,
    stats.kurtosis,
    ep.tail_ratio,
    value_at_risk
]

FACTOR_STAT_FUNCS = [
    ep.alpha,
    ep.beta,
]

STAT_FUNC_NAMES = {
    'annual_return': 'Annual return',
    'cum_returns_final': 'Cumulative returns',
    'annual_volatility': 'Annual volatility',
    'sharpe_ratio': 'Sharpe ratio',
    'calmar_ratio': 'Calmar ratio',
    'stability_of_timeseries': 'Stability',
    'max_drawdown': 'Max drawdown',
    'omega_ratio': 'Omega ratio',
    'sortino_ratio': 'Sortino ratio',
    'skew': 'Skew',
    'kurtosis': 'Kurtosis',
    'tail_ratio': 'Tail ratio',
    'common_sense_ratio': 'Common sense ratio',
    'value_at_risk': 'Daily value at risk',
    'alpha': 'Alpha',
    'beta': 'Beta',
}

FACTOR_PARTITIONS = {
    'style': ['momentum', 'size', 'value', 'reversal_short_term',
              'volatility'],
    'sector': ['basic_materials', 'consumer_cyclical', 'financial_services',
               'real_estate', 'consumer_defensive', 'health_care',
               'utilities', 'communication_services', 'energy', 'industrials',
               'technology']
}

STAT_FUNCS_PCT = [
    'Annual return',
    'Cumulative returns',
    'Annual volatility',
    'Max drawdown',
    'Daily value at risk',
    'Daily turnover'
]

PNL_STATS = OrderedDict(
    [('Total profit', lambda x: x.sum()),
     ('Gross profit', lambda x: x[x > 0].sum()),
     ('Gross loss', lambda x: x[x < 0].sum()),
     ('Profit factor', lambda x: x[x > 0].sum() / x[x < 0].abs().sum()
     if x[x < 0].abs().sum() != 0 else np.nan),
     ('Avg. trade net profit', 'mean'),
     ('Avg. winning trade', lambda x: x[x > 0].mean()),
     ('Avg. losing trade', lambda x: x[x < 0].mean()),
     ('Ratio Avg. Win:Avg. Loss', lambda x: x[x > 0].mean() /
                                            x[x < 0].abs().mean() if x[x < 0].abs().mean() != 0 else np.nan),
     ('Largest winning trade', 'max'),
     ('Largest losing trade', 'min'),
     ])

SUMMARY_STATS = OrderedDict(
    [('Total number of round_trips', 'count'),
     ('Percent profitable', lambda x: len(x[x > 0]) / float(len(x))),
     ('Winning round_trips', lambda x: len(x[x > 0])),
     ('Losing round_trips', lambda x: len(x[x < 0])),
     ('Even round_trips', lambda x: len(x[x == 0])),
     ])

RETURN_STATS = OrderedDict(
    [('Avg returns all round_trips', lambda x: x.mean()),
     ('Avg returns winning', lambda x: x[x > 0].mean()),
     ('Avg returns losing', lambda x: x[x < 0].mean()),
     ('Median returns all round_trips', lambda x: x.median()),
     ('Median returns winning', lambda x: x[x > 0].median()),
     ('Median returns losing', lambda x: x[x < 0].median()),
     ('Largest winning trade', 'max'),
     ('Largest losing trade', 'min'),
     ])

DURATION_STATS = OrderedDict(
    [('Avg duration', lambda x: x.mean()),
     ('Median duration', lambda x: x.median()),
     ('Longest duration', lambda x: x.max()),
     ('Shortest duration', lambda x: x.min())
     #  FIXME: Instead of x.max() - x.min() this should be
     #  rts.close_dt.max() - rts.open_dt.min() which is not
     #  available here. As it would require a new approach here
     #  that passes in multiple fields we disable these measures
     #  for now.
     #  ('Avg # round_trips per day', lambda x: float(len(x)) /
     #   (x.max() - x.min()).days),
     #  ('Avg # round_trips per month', lambda x: float(len(x)) /
     #   (((x.max() - x.min()).days) / APPROX_BDAYS_PER_MONTH)),
     ])