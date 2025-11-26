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

"""交易分析函数模块."""

import numpy as np
import pandas as pd

__all__ = [
    'daily_txns_with_bar_data',
    'days_to_liquidate_positions',
    'get_max_days_to_liquidate_by_ticker',
    'get_low_liquidity_transactions',
    'apply_slippage_penalty',
    'map_transaction',
    'make_transaction_frame',
    'get_txn_vol',
    'adjust_returns_for_slippage',
    'get_turnover',
]


def daily_txns_with_bar_data(transactions, market_data):
    """Augment transactions with bar data."""
    txns = transactions.copy()
    
    if market_data is None:
        return txns

    txns = txns.reset_index()
    txns['date'] = pd.to_datetime(txns['date']).dt.date
    txns = txns.set_index('date')

    return txns


def days_to_liquidate_positions(positions, market_data, max_bar_consumption=0.05, capital_base=1e6):
    """Calculate days to liquidate positions."""
    positions = positions.copy()
    
    if 'cash' in positions.columns:
        positions = positions.drop('cash', axis=1)

    days_to_liquidate = pd.DataFrame(index=positions.index, columns=positions.columns)

    for col in positions.columns:
        if col in market_data.columns:
            volume = market_data[col]
            max_tradable = volume * max_bar_consumption
            pos_value = positions[col].abs()
            days_to_liquidate[col] = pos_value / max_tradable

    return days_to_liquidate


def get_max_days_to_liquidate_by_ticker(positions, market_data, max_bar_consumption=0.05, capital_base=1e6):
    """Get maximum days to liquidate by ticker."""
    dtl = days_to_liquidate_positions(positions, market_data, max_bar_consumption, capital_base)
    return dtl.max(axis=0)


def get_low_liquidity_transactions(transactions, market_data, last_n_days=None):
    """Get low liquidity transactions."""
    if last_n_days is not None:
        transactions = transactions.iloc[-last_n_days:]

    return transactions


def apply_slippage_penalty(returns, txn_vol, slippage_bps=10):
    """Apply slippage penalty to returns."""
    returns = returns.copy()
    
    penalty = txn_vol * slippage_bps / 10000
    
    return returns - penalty


def map_transaction(txn):
    """Map a single transaction."""
    return {
        'amount': txn.get('amount', 0),
        'price': txn.get('price', 0),
        'sid': txn.get('sid', None),
        'symbol': txn.get('symbol', ''),
        'dt': txn.get('dt', None),
    }


def make_transaction_frame(transactions):
    """Make transaction DataFrame."""
    if isinstance(transactions, pd.DataFrame):
        return transactions

    txns = [map_transaction(t) for t in transactions]
    df = pd.DataFrame(txns)

    if 'dt' in df.columns:
        df = df.set_index('dt')

    return df


def get_txn_vol(transactions):
    """Get transaction volume."""
    transactions = transactions.copy()
    
    if 'amount' in transactions.columns and 'price' in transactions.columns:
        transactions['value'] = transactions['amount'].abs() * transactions['price']
        return transactions.groupby(transactions.index.date)['value'].sum()

    return pd.Series(dtype=float)


def adjust_returns_for_slippage(returns, positions, transactions, slippage_bps=10):
    """Adjust returns for slippage."""
    from fincore.empyricals.returns import cum_returns
    
    returns = returns.copy()
    
    turnover = get_turnover(positions, transactions)
    penalty = turnover * slippage_bps / 10000

    penalty = penalty.reindex(returns.index, fill_value=0)

    return returns - penalty


def get_turnover(positions, transactions, denominator="AGB"):
    """Get portfolio turnover.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily net position values.
    transactions : pd.DataFrame
        Executed trade prices and amounts.
    denominator : str, optional
        Either 'AGB' (average gross book) or 'portfolio_value'.

    Returns
    -------
    pd.Series
        Time-series of daily turnover.
    """
    positions = positions.copy()
    
    if 'cash' in positions.columns:
        positions = positions.drop('cash', axis=1)

    abs_delta = positions.diff().abs().sum(axis=1)

    if denominator == "AGB":
        total_abs = positions.abs().sum(axis=1)
        turnover = abs_delta / total_abs
    else:
        turnover = abs_delta

    return turnover
