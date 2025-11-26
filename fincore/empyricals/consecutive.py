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

"""连续涨跌相关函数模块."""

import numpy as np
import pandas as pd
from fincore.empyricals.returns import cum_returns_final

__all__ = [
    'max_consecutive_up_days',
    'max_consecutive_down_days',
    'max_consecutive_gain',
    'max_consecutive_loss',
    'max_consecutive_up_weeks',
    'max_consecutive_down_weeks',
    'max_consecutive_up_months',
    'max_consecutive_down_months',
    'max_single_day_gain',
    'max_single_day_loss',
    'max_single_day_gain_date',
    'max_single_day_loss_date',
    'max_consecutive_up_start_date',
    'max_consecutive_up_end_date',
    'max_consecutive_down_start_date',
    'max_consecutive_down_end_date',
]


def max_consecutive_up_days(returns):
    """Determine the maximum number of consecutive days with positive returns.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative daily returns with a pandas index.

    Returns
    -------
    int or float
        Length (in days) of the longest run of strictly positive daily
        returns, or ``NaN`` if the input is empty.
    """
    if len(returns) < 1:
        return np.nan

    up_days = returns > 0

    if not up_days.any():
        return 0

    groups = (up_days != up_days.shift(1)).cumsum()
    consecutive_counts = up_days.groupby(groups).sum()

    return consecutive_counts.max()


def max_consecutive_down_days(returns):
    """Determine the maximum number of consecutive days with negative returns.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative daily returns with a pandas index.

    Returns
    -------
    int or float
        Length (in days) of the longest run of strictly negative daily
        returns, or ``NaN`` if the input is empty.
    """
    if len(returns) < 1:
        return np.nan

    down_days = returns < 0

    if not down_days.any():
        return 0

    groups = (down_days != down_days.shift(1)).cumsum()
    consecutive_counts = down_days.groupby(groups).sum()

    return consecutive_counts.max()


def max_consecutive_gain(returns):
    """Determine the maximum cumulative gain over consecutive positive days.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative daily returns with a pandas index.

    Returns
    -------
    float
        Maximum sum of returns over any contiguous block of strictly
        positive-return days, or ``NaN`` if there are no positive days.
    """
    if len(returns) < 1:
        return np.nan

    up_days = returns > 0

    if not up_days.any():
        return np.nan

    groups = (up_days != up_days.shift(1)).cumsum()
    consecutive_gains = returns.where(up_days, 0).groupby(groups).sum()

    return consecutive_gains.max()


def max_consecutive_loss(returns):
    """Determine the maximum cumulative loss over consecutive negative days.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative daily returns with a pandas index.

    Returns
    -------
    float
        Minimum (most negative) sum of returns over any contiguous block
        of strictly negative-return days, or ``NaN`` if there are no
        negative days.
    """
    if len(returns) < 1:
        return np.nan

    down_days = returns < 0

    if not down_days.any():
        return np.nan

    groups = (down_days != down_days.shift(1)).cumsum()
    consecutive_losses = returns.where(down_days, 0).groupby(groups).sum()

    return consecutive_losses.min()


def max_consecutive_up_weeks(returns):
    """Determine the maximum number of consecutive weeks with positive returns.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative returns indexed by date.

    Returns
    -------
    int or float
        Length (in weeks) of the longest run of strictly positive weekly
        returns, or ``NaN`` if the input is empty.
    """
    if len(returns) < 1:
        return np.nan

    weekly_returns = returns.resample("W").apply(lambda g: cum_returns_final(g))

    up_weeks = weekly_returns > 0

    if not up_weeks.any():
        return 0

    groups = (up_weeks != up_weeks.shift(1)).cumsum()
    consecutive_counts = up_weeks.groupby(groups).sum()

    return consecutive_counts.max()


def max_consecutive_down_weeks(returns):
    """Determine the maximum number of consecutive weeks with negative returns.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative returns indexed by date.

    Returns
    -------
    int or float
        Length (in weeks) of the longest run of strictly negative weekly
        returns, or ``NaN`` if the input is empty.
    """
    if len(returns) < 1:
        return np.nan

    weekly_returns = returns.resample("W").apply(lambda g: cum_returns_final(g))

    down_weeks = weekly_returns < 0

    if not down_weeks.any():
        return 0

    groups = (down_weeks != down_weeks.shift(1)).cumsum()
    consecutive_counts = down_weeks.groupby(groups).sum()

    return consecutive_counts.max()


def max_consecutive_up_months(returns):
    """Determine the maximum number of consecutive months with positive returns.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative returns indexed by date.

    Returns
    -------
    int or float
        Length (in months) of the longest run of strictly positive
        monthly returns, or ``NaN`` if the input is empty.
    """
    if len(returns) < 1:
        return np.nan

    monthly_returns = returns.resample("M").apply(lambda g: cum_returns_final(g))

    up_months = monthly_returns > 0

    if not up_months.any():
        return 0

    groups = (up_months != up_months.shift(1)).cumsum()
    consecutive_counts = up_months.groupby(groups).sum()

    return consecutive_counts.max()


def max_consecutive_down_months(returns):
    """Determine the maximum number of consecutive months with negative returns.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative returns indexed by date.

    Returns
    -------
    int or float
        Length (in months) of the longest run of strictly negative
        monthly returns, or ``NaN`` if the input is empty.
    """
    if len(returns) < 1:
        return np.nan

    monthly_returns = returns.resample("M").apply(lambda g: cum_returns_final(g))

    down_months = monthly_returns < 0

    if not down_months.any():
        return 0

    groups = (down_months != down_months.shift(1)).cumsum()
    consecutive_counts = down_months.groupby(groups).sum()

    return consecutive_counts.max()


def max_single_day_gain(returns):
    """Determine the maximum single-day gain."""
    if len(returns) < 1:
        return np.nan
    return returns.max()


def max_single_day_loss(returns):
    """Determine the maximum single-day loss."""
    if len(returns) < 1:
        return np.nan
    return returns.min()


def max_single_day_gain_date(returns):
    """Determine the date of the maximum single-day gain.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative returns indexed by date.

    Returns
    -------
    pd.Timestamp or None
        Date of the maximum single-day gain, or ``None`` if the input is empty.
    """
    if len(returns) < 1:
        return None
    return returns.idxmax()


def max_single_day_loss_date(returns):
    """Determine the date of the maximum single-day loss.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative returns indexed by date.

    Returns
    -------
    pd.Timestamp or None
        Date of the maximum single-day loss, or ``None`` if the input is empty.
    """
    if len(returns) < 1:
        return None
    return returns.idxmin()


def max_consecutive_up_start_date(returns):
    """Determine the start date of the longest consecutive up period.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative returns indexed by date.

    Returns
    -------
    pd.Timestamp or None
        Start date of the longest run of strictly positive returns,
        or ``None`` if there are no positive returns.
    """
    if len(returns) < 1:
        return None

    up_days = returns > 0

    if not up_days.any():
        return None

    groups = (up_days != up_days.shift(1)).cumsum()
    consecutive_counts = up_days.groupby(groups).sum()

    max_group = consecutive_counts.idxmax()

    group_mask = groups == max_group
    return returns[group_mask & up_days].index[0]


def max_consecutive_up_end_date(returns):
    """Determine the end date of the longest consecutive up period.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative returns indexed by date.

    Returns
    -------
    pd.Timestamp or None
        End date of the longest run of strictly positive returns,
        or ``None`` if there are no positive returns.
    """
    if len(returns) < 1:
        return None

    up_days = returns > 0

    if not up_days.any():
        return None

    groups = (up_days != up_days.shift(1)).cumsum()
    consecutive_counts = up_days.groupby(groups).sum()

    max_group = consecutive_counts.idxmax()

    group_mask = groups == max_group
    return returns[group_mask & up_days].index[-1]


def max_consecutive_down_start_date(returns):
    """Determine the start date of the longest consecutive down period.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative returns indexed by date.

    Returns
    -------
    pd.Timestamp or None
        Start date of the longest run of strictly negative returns,
        or ``None`` if there are no negative returns.
    """
    if len(returns) < 1:
        return None

    down_days = returns < 0

    if not down_days.any():
        return None

    groups = (down_days != down_days.shift(1)).cumsum()
    consecutive_counts = down_days.groupby(groups).sum()

    max_group = consecutive_counts.idxmax()

    group_mask = groups == max_group
    return returns[group_mask & down_days].index[0]


def max_consecutive_down_end_date(returns):
    """Determine the end date of the longest consecutive down period.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative returns indexed by date.

    Returns
    -------
    pd.Timestamp or None
        End date of the longest run of strictly negative returns,
        or ``None`` if there are no negative returns.
    """
    if len(returns) < 1:
        return None

    down_days = returns < 0

    if not down_days.any():
        return None

    groups = (down_days != down_days.shift(1)).cumsum()
    consecutive_counts = down_days.groupby(groups).sum()

    max_group = consecutive_counts.idxmax()

    group_mask = groups == max_group
    return returns[group_mask & down_days].index[-1]
