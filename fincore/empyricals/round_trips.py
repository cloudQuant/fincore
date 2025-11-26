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
    stats_all = []
    stats_long_short = []

    for kind, group in round_trips.groupby('long'):
        if kind:
            label = 'long'
        else:
            label = 'short'

        stat = {}
        data = group[col]

        for name, func in stats_dict.items():
            try:
                if callable(func):
                    stat[name] = func(data)
                elif isinstance(func, str):
                    # Handle string method names like 'mean', 'sum', etc.
                    stat[name] = getattr(data, func)()
                else:
                    stat[name] = np.nan
            except Exception:
                stat[name] = np.nan

        stat_series = pd.Series(stat, name=label)
        stats_long_short.append(stat_series)

    # Compute 'All' statistics
    all_data = round_trips[col]
    all_stat = {}
    for name, func in stats_dict.items():
        try:
            if callable(func):
                all_stat[name] = func(all_data)
            elif isinstance(func, str):
                all_stat[name] = getattr(all_data, func)()
            else:
                all_stat[name] = np.nan
        except Exception:
            all_stat[name] = np.nan

    all_series = pd.Series(all_stat, name='All trades')
    stats_all.append(all_series)

    # Combine into DataFrame
    if stats_long_short:
        df_long_short = pd.concat(stats_long_short, axis=1).T
    else:
        df_long_short = pd.DataFrame()

    df_all = pd.concat(stats_all, axis=1).T

    return pd.concat([df_all, df_long_short], axis=0)


def groupby_consecutive(txn, max_delta=pd.Timedelta("8h")):
    """Merge transactions of the same direction separated by less than max_delta time duration.

    Parameters
    ----------
    txn : pd.DataFrame
        Prices and amounts of executed round_trips. One row per trade.
    max_delta : pandas.Timedelta (optional)
        Merge transactions in the same direction separated by less
        than max_delta time duration.

    Returns
    -------
    transactions : pd.DataFrame
    """
    import warnings
    import numpy as np

    def vwap(transaction):
        if transaction.amount.sum() == 0:
            warnings.warn("Zero transacted shares, setting vwap to nan.")
            return np.nan
        return (
            transaction.amount * transaction.price
        ).sum() / transaction.amount.sum()

    out = []
    for sym, t in txn.groupby("symbol"):
        t = t.sort_index()
        t.index.name = "dt"
        t.index = pd.to_datetime(t.index)
        t = t.reset_index()

        t["order_sign"] = t.amount > 0
        t["block_dir"] = (
            (t.order_sign.shift(1) != t.order_sign).astype(int).cumsum()
        )
        t["block_time"] = ((t.dt - t.dt.shift(1)) >
                           max_delta).astype(int).cumsum()
        grouped_price = t.groupby(["block_dir", "block_time"])[
            ["amount", "price"]
        ].apply(vwap)
        grouped_price.name = "price"
        grouped_rest = t.groupby(["block_dir", "block_time"]).agg(
            {"amount": "sum", "symbol": "first", "dt": "first"}
        )

        grouped = grouped_rest.join(grouped_price)

        out.append(grouped)

    out = pd.concat(out)
    out = out.set_index("dt")
    return out


def extract_round_trips(transactions, portfolio_value=None):
    """Group transactions into "round trips".

    First, transactions are grouped by day and directionality. Then, long and short
    transactions are matched to create round-trip round_trips for which
    PnL, duration and returns are computed. Crossings where a position
    changes from long to short and vice versa are handled correctly.

    Under the hood, we reconstruct the individual shares in a
    portfolio over time and match round_trips in a FIFO order.

    Parameters
    ----------
    transactions : pd.DataFrame
        Prices and amounts of executed round_trips. One row per trade.

    portfolio_value : pd.Series (optional)
        Portfolio value (all net assets including cash) over time.

    Returns
    -------
    round_trips : pd.DataFrame:
        DataFrame with one row per round trip.
    """
    import warnings

    transactions = groupby_consecutive(transactions)
    roundtrips = []

    for sym, trans_sym in transactions.groupby("symbol"):
        trans_sym = trans_sym.sort_index()
        price_stack = deque()
        dt_stack = deque()
        trans_sym["signed_price"] = trans_sym.price * np.sign(trans_sym.amount)
        trans_sym["abs_amount"] = trans_sym.amount.abs().astype(int)
        for dt, t in trans_sym.iterrows():
            if t.price < 0:
                warnings.warn("Negative price detected, ignoring for round-trip.")
                continue

            indiv_prices = [t.signed_price] * t.abs_amount
            if (len(price_stack) == 0) or (
                np.copysign(1, price_stack[-1]) == np.copysign(1, t.amount)
            ):
                price_stack.extend(indiv_prices)
                dt_stack.extend([dt] * len(indiv_prices))
            else:
                # Close round-trip
                pnl = 0
                invested = 0
                cur_open_dts = []

                for price in indiv_prices:
                    if len(price_stack) != 0 and (
                        np.copysign(1, price_stack[-1]) != np.copysign(1, price)
                    ):
                        # Retrieve the first dt, stock-price pair from stack
                        prev_price = price_stack.popleft()
                        prev_dt = dt_stack.popleft()

                        pnl += -(price + prev_price)
                        cur_open_dts.append(prev_dt)
                        invested += abs(prev_price)
                    else:
                        # Push additional stock prices onto the stack
                        price_stack.append(price)
                        dt_stack.append(dt)

                if cur_open_dts:  # Only add if we have valid data
                    roundtrips.append(
                        {
                            "pnl": pnl,
                            "open_dt": cur_open_dts[0],
                            "close_dt": dt,
                            "long": price < 0,
                            "rt_returns": pnl / invested if invested != 0 else 0,
                            "symbol": sym,
                        }
                    )

    roundtrips = pd.DataFrame(roundtrips)

    if len(roundtrips) == 0:
        return roundtrips

    roundtrips["duration"] = roundtrips["close_dt"].sub(roundtrips["open_dt"])

    if portfolio_value is not None:
        # Need to normalize so that we can join
        pv = pd.DataFrame(
            portfolio_value,
            columns=["portfolio_value"],
        ).assign(date=portfolio_value.index)

        roundtrips["date"] = roundtrips.close_dt.apply(
            lambda close_dt: close_dt.replace(hour=0, minute=0, second=0)
        )
        # Convert 'roundtrips.date' to UTC to match 'portfolio_value.index'
        if pv.index.tz is not None:
            # Only localize if not already tz-aware
            if roundtrips["date"].dt.tz is None:
                roundtrips["date"] = roundtrips["date"].dt.tz_localize("UTC")
            else:
                roundtrips["date"] = roundtrips["date"].dt.tz_convert("UTC")

        tmp = roundtrips.join(pv, on="date", lsuffix="_")

        roundtrips["returns"] = tmp.pnl / tmp.portfolio_value
        roundtrips = roundtrips.drop("date", axis="columns")

    return roundtrips


def add_closing_transactions(positions, transactions):
    """Append transactions that close out all positions at the end of the timespan.

    Utilizes pricing information in the positions DataFrame to determine closing price.

    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.
    transactions : pd.DataFrame
        Prices and amounts of executed round_trips. One row per trade.

    Returns
    -------
    pd.DataFrame
        Transactions with closing transactions appended.
    """
    closed_txns = transactions[["symbol", "amount", "price"]]

    pos_at_end = positions.drop("cash", axis=1).iloc[-1]
    open_pos = pos_at_end.replace(0, np.nan).dropna()
    # Add closing transactions one second after the close to be sure
    # they don't conflict with other transactions executed at that time.
    end_dt = open_pos.name + pd.Timedelta(seconds=1)

    for sym, ending_val in open_pos.items():
        txn_sym = transactions[transactions.symbol == sym]

        ending_amount = txn_sym.amount.sum()

        ending_price = ending_val / ending_amount
        closing_txn = {
            "symbol": sym,
            "amount": -ending_amount,
            "price": ending_price,
        }

        closing_txn = pd.DataFrame(closing_txn, index=[end_dt])
        closed_txns = pd.concat([closed_txns, closing_txn], ignore_index=False)

    closed_txns = closed_txns[closed_txns.amount != 0]

    return closed_txns


def apply_sector_mappings_to_round_trips(round_trips, sector_mappings):
    """Apply sector mappings to round trips."""
    round_trips = round_trips.copy()
    
    if 'symbol' in round_trips.columns:
        round_trips['sector'] = round_trips['symbol'].map(sector_mappings)

    return round_trips


def gen_round_trip_stats(round_trips):
    """Generate various round-trip statistics.

    Parameters
    ----------
    round_trips : pd.DataFrame
        DataFrame with one row per-round-trip trade.

    Returns
    -------
    dict
        A dictionary where each value is a pandas DataFrame containing
        various round-trip statistics.
    """
    from fincore.constants.style import (
        PNL_STATS, SUMMARY_STATS, DURATION_STATS, RETURN_STATS
    )

    if len(round_trips) == 0:
        return {
            "pnl": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "duration": pd.DataFrame(),
            "returns": pd.DataFrame(),
            "symbols": pd.DataFrame(),
        }

    # Helper function to apply custom and built-in functions
    def apply_custom_and_built_in_funcs(grouped, stats_dict):
        # Separate custom functions from built-in functions
        custom_funcs = {k: v for k, v in stats_dict.items() if callable(v)}
        built_in_funcs = [v for k, v in stats_dict.items() if not callable(v)]

        # Apply custom functions manually
        custom_results = {}
        for func_name, func in custom_funcs.items():
            custom_results[func_name] = grouped.apply(func)
        custom_results = pd.DataFrame(custom_results)

        # Apply built-in functions
        if built_in_funcs:
            built_in_results = grouped.agg(built_in_funcs)
            # Combine results
            return pd.concat([custom_results, built_in_results], axis=1)
        return custom_results

    # Check if 'returns' column exists, if not use 'rt_returns'
    returns_col = 'returns' if 'returns' in round_trips.columns else 'rt_returns'

    # Generate statistics for pnl, summary, duration, and returns
    round_trip_stats = {
        "pnl": agg_all_long_short(round_trips, "pnl", PNL_STATS),
        "summary": agg_all_long_short(round_trips, "pnl", SUMMARY_STATS),
        "duration": agg_all_long_short(round_trips, "duration", DURATION_STATS),
        "returns": agg_all_long_short(round_trips, returns_col, RETURN_STATS),
        "symbols": apply_custom_and_built_in_funcs(
            round_trips.groupby("symbol")[returns_col], RETURN_STATS
        ).T,
    }

    return round_trip_stats
