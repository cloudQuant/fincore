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

"""往返交易函数模块."""

import numpy as np
import pandas as pd
from collections import deque

__all__ = [
    'agg_all_long_short',
    'groupby_consecutive',
    'extract_round_trips',
    'add_closing_transactions',
    'apply_sector_mappings_to_round_trips',
    'gen_round_trip_stats',
]


def agg_all_long_short(round_trips, col, stats_dict):
    """Aggregate statistics for long and short round trips."""
    stats = []
    
    for kind, group in round_trips.groupby('long'):
        if kind:
            label = 'Long'
        else:
            label = 'Short'

        stat = {}
        stat['direction'] = label
        
        for name, func in stats_dict.items():
            stat[name] = func(group[col])

        stats.append(stat)

    return pd.DataFrame(stats)


def groupby_consecutive(txn, max_delta=pd.Timedelta("8h")):
    """Group transactions by consecutive trades."""
    txn = txn.sort_index()
    
    time_diff = txn.index.to_series().diff()
    
    group_ids = (time_diff > max_delta).cumsum()
    
    return group_ids


def extract_round_trips(transactions, portfolio_value=None):
    """Extract round trips from transactions."""
    transactions = transactions.copy()
    
    if 'symbol' not in transactions.columns:
        if 'sid' in transactions.columns:
            transactions['symbol'] = transactions['sid']
        else:
            return pd.DataFrame()

    round_trips = []
    open_trades = {}

    for symbol in transactions['symbol'].unique():
        symbol_txns = transactions[transactions['symbol'] == symbol].sort_index()
        open_amount = 0
        open_value = 0
        open_dt = None

        for dt, txn in symbol_txns.iterrows():
            amount = txn['amount']
            price = txn['price']

            if open_amount == 0:
                # New position
                open_amount = amount
                open_value = amount * price
                open_dt = dt
            elif (open_amount > 0 and amount > 0) or (open_amount < 0 and amount < 0):
                # Adding to position
                open_amount += amount
                open_value += amount * price
            else:
                # Closing position (fully or partially)
                close_amount = min(abs(amount), abs(open_amount))
                if open_amount > 0:
                    close_amount = close_amount
                else:
                    close_amount = -close_amount

                avg_open_price = open_value / open_amount if open_amount != 0 else 0

                round_trip = {
                    'symbol': symbol,
                    'open_dt': open_dt,
                    'close_dt': dt,
                    'long': open_amount > 0,
                    'open_price': avg_open_price,
                    'close_price': price,
                    'pnl': (price - avg_open_price) * close_amount,
                    'returns': (price / avg_open_price - 1) if avg_open_price != 0 else 0,
                    'duration': (dt - open_dt).total_seconds() / 86400 if hasattr(dt - open_dt, 'total_seconds') else 1,
                }
                round_trips.append(round_trip)

                remaining = abs(amount) - abs(close_amount)
                if remaining > 0:
                    open_amount = amount - close_amount if amount > 0 else -(abs(amount) - abs(close_amount))
                    open_value = open_amount * price
                    open_dt = dt
                else:
                    open_amount = 0
                    open_value = 0
                    open_dt = None

    return pd.DataFrame(round_trips)


def add_closing_transactions(positions, transactions):
    """Add closing transactions for open positions."""
    transactions = transactions.copy()
    positions = positions.copy()

    if len(positions) == 0:
        return transactions

    last_positions = positions.iloc[-1]
    last_date = positions.index[-1]

    closing_txns = []
    for symbol, amount in last_positions.items():
        if symbol != 'cash' and amount != 0:
            closing_txns.append({
                'symbol': symbol,
                'amount': -amount,
                'price': np.nan,
                'dt': last_date,
            })

    if closing_txns:
        closing_df = pd.DataFrame(closing_txns)
        if 'dt' in closing_df.columns:
            closing_df = closing_df.set_index('dt')
        transactions = pd.concat([transactions, closing_df])

    return transactions


def apply_sector_mappings_to_round_trips(round_trips, sector_mappings):
    """Apply sector mappings to round trips."""
    round_trips = round_trips.copy()
    
    if 'symbol' in round_trips.columns:
        round_trips['sector'] = round_trips['symbol'].map(sector_mappings)

    return round_trips


def gen_round_trip_stats(round_trips):
    """Generate round trip statistics."""
    if len(round_trips) == 0:
        return pd.DataFrame()

    stats_dict = {
        'count': len,
        'mean': np.mean,
        'median': np.median,
        'std': np.std,
        'sum': np.sum,
    }

    def apply_custom_and_built_in_funcs(grouped, stats_dict):
        results = {}
        for name, func in stats_dict.items():
            try:
                results[name] = func(grouped)
            except Exception:
                results[name] = np.nan
        return pd.Series(results)

    stats = {}

    if 'pnl' in round_trips.columns:
        pnl_stats = apply_custom_and_built_in_funcs(round_trips['pnl'], stats_dict)
        for k, v in pnl_stats.items():
            stats['pnl_' + k] = v

    if 'returns' in round_trips.columns:
        returns_stats = apply_custom_and_built_in_funcs(round_trips['returns'], stats_dict)
        for k, v in returns_stats.items():
            stats['returns_' + k] = v

    if 'duration' in round_trips.columns:
        duration_stats = apply_custom_and_built_in_funcs(round_trips['duration'], stats_dict)
        for k, v in duration_stats.items():
            stats['duration_' + k] = v

    if 'long' in round_trips.columns:
        stats['long_count'] = round_trips['long'].sum()
        stats['short_count'] = (~round_trips['long']).sum()

    if 'pnl' in round_trips.columns:
        stats['win_rate'] = (round_trips['pnl'] > 0).mean()

    return pd.DataFrame([stats])
