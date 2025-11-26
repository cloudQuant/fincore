#
# Copyright 2016 Quantopian, Inc.
# Copyright 2025 CloudQuant Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""持仓分析函数模块."""

import numpy as np
import pandas as pd

__all__ = [
    'get_percent_alloc',
    'get_top_long_short_abs',
    'get_max_median_position_concentration',
    'extract_pos',
    'get_sector_exposures',
    'get_long_short_pos',
    'compute_style_factor_exposures',
    'compute_sector_exposures',
    'compute_cap_exposures',
    'compute_volume_exposures',
    'gross_lev',
    'stack_positions',
]


def get_percent_alloc(values):
    """Get percent allocation of values."""
    return values / np.abs(values).sum()


def get_top_long_short_abs(positions, top=10):
    """Get top long and short positions by absolute value."""
    positions = positions.copy()
    df_max = positions.abs().max(axis=0).nlargest(top)
    return df_max


def get_max_median_position_concentration(positions):
    """Get maximum and median position concentration."""
    positions = positions.copy()
    
    if 'cash' in positions.columns:
        positions = positions.drop('cash', axis=1)

    positions_pct = positions.apply(get_percent_alloc, axis=1)

    max_concentration = positions_pct.abs().max(axis=1).max()
    median_concentration = positions_pct.abs().max(axis=1).median()

    return max_concentration, median_concentration


def extract_pos(positions, cash):
    """Extract positions and cash."""
    positions = positions.copy()
    if cash is not None:
        if isinstance(cash, pd.Series):
            positions['cash'] = cash
        else:
            positions['cash'] = cash
    return positions


def get_sector_exposures(positions, symbol_sector_map):
    """Get sector exposures from positions."""
    positions = positions.copy()
    
    sector_exp = positions.T.groupby(symbol_sector_map).sum().T

    return sector_exp


def get_long_short_pos(positions):
    """Get long and short positions."""
    positions = positions.copy()
    
    if 'cash' in positions.columns:
        positions = positions.drop('cash', axis=1)

    longs = positions.where(positions > 0, 0).sum(axis=1)
    shorts = positions.where(positions < 0, 0).abs().sum(axis=1)

    return longs, shorts


def compute_style_factor_exposures(positions, risk_factor):
    """Compute style factor exposures."""
    positions = positions.copy()
    
    aligned = positions.align(risk_factor, axis=0, join='inner')[0]
    risk_factor_aligned = risk_factor.loc[aligned.index]

    exposures = aligned.mul(risk_factor_aligned, axis=0).sum(axis=1)

    return exposures


def compute_sector_exposures(positions, sectors, sector_dict=None):
    """Compute sector exposures."""
    positions = positions.copy()
    
    if sector_dict is None:
        sector_dict = {}

    exposures = {}
    for sector in sectors:
        sector_stocks = [s for s, sec in sector_dict.items() if sec == sector]
        sector_pos = positions[[c for c in positions.columns if c in sector_stocks]]
        exposures[sector] = sector_pos.sum(axis=1)

    return pd.DataFrame(exposures)


def compute_cap_exposures(positions, caps):
    """Compute market cap exposures."""
    positions = positions.copy()
    
    exposures = {}
    for cap_category, cap_stocks in caps.items():
        cap_pos = positions[[c for c in positions.columns if c in cap_stocks]]
        exposures[cap_category] = cap_pos.sum(axis=1)

    return pd.DataFrame(exposures)


def compute_volume_exposures(shares_held, volumes, percentile):
    """Compute volume exposures."""
    days_to_liquidate = shares_held.abs() / volumes

    return (days_to_liquidate > percentile).sum(axis=1)


def gross_lev(positions):
    """Calculate the gross leverage of a strategy.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily net position values.

    Returns
    -------
    pd.Series
        Gross leverage.
    """
    positions = positions.copy()
    
    if 'cash' in positions.columns:
        cash = positions['cash']
        positions = positions.drop('cash', axis=1)
    else:
        cash = pd.Series(0, index=positions.index)

    gross = positions.abs().sum(axis=1)
    net = gross + cash.abs()

    return gross / net


def stack_positions(positions, pos_in_dollars=True):
    """Stack positions into a multi-index DataFrame."""
    positions = positions.copy()
    
    if 'cash' in positions.columns:
        positions = positions.drop('cash', axis=1)

    stacked = positions.stack()
    stacked.index.names = ['dt', 'ticker']

    return stacked
