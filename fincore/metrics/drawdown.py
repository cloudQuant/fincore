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

"""回撤相关函数模块."""

import numpy as np
import pandas as pd
from fincore.utils import nanmin
from fincore.metrics.returns import cum_returns, cum_returns_final

__all__ = [
    'max_drawdown',
    'get_all_drawdowns',
    'get_all_drawdowns_detailed',
    'get_max_drawdown',
    'get_max_drawdown_underwater',
    'get_top_drawdowns',
    'gen_drawdown_table',
    'get_max_drawdown_period',
    'max_drawdown_days',
    'max_drawdown_weeks',
    'max_drawdown_months',
    'max_drawdown_recovery_days',
    'max_drawdown_recovery_weeks',
    'max_drawdown_recovery_months',
    'second_max_drawdown',
    'third_max_drawdown',
    'second_max_drawdown_days',
    'second_max_drawdown_recovery_days',
    'third_max_drawdown_days',
    'third_max_drawdown_recovery_days',
]


def max_drawdown(returns, out=None):
    """Determine the maximum drawdown of a return series.

    Maximum drawdown is defined as the minimum (most negative) percentage
    drop from a running maximum of cumulative returns.

    Parameters
    ----------
    returns : array-like or pd.Series or pd.DataFrame
        Non-cumulative simple returns.
    out : np.ndarray, optional
        pre-allocated output array. If provided, the result is
        written in-place into this array.

    Returns
    -------
    float or np.ndarray or pd.Series
        Maximum drawdown. For 1D input a scalar is returned; for 2D input
        one value is returned per column.
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns_array = np.asanyarray(returns)

    cumulative = np.empty(
        (returns.shape[0] + 1,) + returns.shape[1:],
        dtype="float64",
    )
    cumulative[0] = start = 100
    cum_returns(returns_array, starting_value=start,
                out=cumulative[1:])

    max_return = np.fmax.accumulate(cumulative, axis=0)

    nanmin((cumulative - max_return) / max_return, axis=0, out=out)
    if returns_1d:
        out = out.item()
    elif allocated_output and isinstance(returns, pd.DataFrame):
        out = pd.Series(out)

    return out


def get_all_drawdowns(returns):
    """Extract all distinct drawdown values from a return series.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative returns.

    Returns
    -------
    list of float
        List of drawdown magnitudes (negative values) for each distinct
        drawdown period.
    """
    if len(returns) < 1:
        return []

    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    # Calculate cumulative returns
    cum_ret = cum_returns(returns, starting_value=100)

    # Calculate rolling maximum
    rolling_max = cum_ret.expanding().max()

    # Calculate drawdown
    drawdown = (cum_ret - rolling_max) / rolling_max

    # Find all local minima (drawdown troughs)
    drawdown_periods = []
    in_drawdown = False
    current_min = 0

    for dd in drawdown:
        if dd < 0:
            if not in_drawdown:
                in_drawdown = True
                current_min = dd
            else:
                current_min = min(current_min, dd)
        else:
            if in_drawdown:
                drawdown_periods.append(current_min)
                in_drawdown = False
                current_min = 0

    # Handle case where series ends in drawdown
    if in_drawdown:
        drawdown_periods.append(current_min)

    return drawdown_periods


def get_all_drawdowns_detailed(returns):
    """Extract detailed information about all drawdown periods.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative returns.

    Returns
    -------
    list of dict
        List of dictionaries with keys 'value', 'duration', 'recovery_duration'
        for each distinct drawdown period.
    """
    if len(returns) < 1:
        return []

    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    # Calculate cumulative returns
    cum_ret = cum_returns(returns, starting_value=100)

    # Calculate rolling maximum
    rolling_max = cum_ret.expanding().max()

    # Calculate drawdown
    drawdown = (cum_ret - rolling_max) / rolling_max

    # Find all drawdown periods with details
    drawdown_periods = []
    in_drawdown = False
    current_min = 0
    start_idx = 0
    trough_idx = 0

    for i, dd in enumerate(drawdown):
        if dd < 0:
            if not in_drawdown:
                in_drawdown = True
                start_idx = i
                current_min = dd
                trough_idx = i
            else:
                if dd < current_min:
                    current_min = dd
                    trough_idx = i
        else:
            if in_drawdown:
                duration = trough_idx - start_idx
                recovery_duration = i - trough_idx
                drawdown_periods.append({
                    'value': current_min,
                    'duration': duration,
                    'recovery_duration': recovery_duration
                })
                in_drawdown = False
                current_min = 0

    # Handle case where series ends in drawdown
    if in_drawdown:
        duration = trough_idx - start_idx
        drawdown_periods.append({
            'value': current_min,
            'duration': duration,
            'recovery_duration': None
        })

    return drawdown_periods


def get_max_drawdown(returns):
    """Determine the maximum drawdown of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.

    Returns
    -------
    peak : datetime
        The date of the peak before the maximum drawdown.
    valley : datetime
        The date of the trough (maximum drawdown).
    recovery : datetime
        The date of recovery or NaT if not recovered.
    """
    returns = returns.copy()
    df_cum = cum_returns(returns, 1.0)
    running_max = np.maximum.accumulate(df_cum)
    underwater = df_cum / running_max - 1
    return get_max_drawdown_underwater(underwater)


def get_max_drawdown_underwater(underwater):
    """Determines peak, valley, and recovery dates given an underwater returns.

    Parameters
    ----------
    underwater : pd.Series
        Underwater returns (cumulative returns minus rolling max).

    Returns
    -------
    peak : datetime
        The date of the peak before the maximum drawdown.
    valley : datetime
        The date of the trough (maximum drawdown).
    recovery : datetime
        The date of recovery or NaT if not recovered.
    """
    valley = underwater.idxmin()  # end of the period
    # Find first 0 (peak is where underwater == 0 before valley)
    peak = underwater[:valley][underwater[:valley] == 0].index[-1]
    # Find last 0 (recovery is where underwater == 0 after valley)
    try:
        recovery = underwater[valley:][underwater[valley:] == 0].index[0]
    except IndexError:
        recovery = np.nan  # drawdown isn't recovered

    return peak, valley, recovery


def get_top_drawdowns(returns, top=10):
    """Find top drawdowns, sorted by severity.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    top : int, optional
        The amount of top drawdowns to find (default 10).

    Returns
    -------
    drawdowns : list
        List of (peak, valley, recovery) tuples.
    """
    returns = returns.copy()
    df_cum = cum_returns(returns, starting_value=1.0)
    running_max = np.maximum.accumulate(df_cum)
    underwater = df_cum / running_max - 1

    drawdowns = []
    for _ in range(top):
        peak, valley, recovery = get_max_drawdown_underwater(underwater)
        if pd.isnull(peak) or pd.isnull(valley):
            break

        # Slice out draw-down period
        if not pd.isnull(recovery):
            underwater.drop(underwater[peak:recovery].index[1:-1], inplace=True)
        else:
            # the drawdown has not ended yet
            underwater = underwater.loc[:peak]

        drawdowns.append((peak, valley, recovery))
        if (len(returns) == 0) or (len(underwater) == 0):
            break

    return drawdowns


def gen_drawdown_table(returns, top=10):
    """Place top drawdowns in a table.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    top : int, optional
        The amount of top drawdowns to find (default 10).

    Returns
    -------
    df_drawdowns : pd.DataFrame
        Information about top drawdowns.
    """
    df_cum = cum_returns(returns, starting_value=1.0)
    drawdown_periods = get_top_drawdowns(returns, top=top)
    df_drawdowns = pd.DataFrame(
        index=list(range(top)),
        columns=[
            "Net drawdown in %",
            "Peak date",
            "Valley date",
            "Recovery date",
            "Duration",
        ],
    )

    for i, (peak, valley, recovery) in enumerate(drawdown_periods):
        if pd.isnull(recovery):
            df_drawdowns.loc[i, "Duration"] = np.nan
        else:
            df_drawdowns.loc[i, "Duration"] = len(
                pd.date_range(peak, recovery, freq="B")
            )

        df_drawdowns.loc[i, "Peak date"] = pd.to_datetime(peak).strftime("%Y-%m-%d")
        df_drawdowns.loc[i, "Valley date"] = pd.to_datetime(valley).strftime("%Y-%m-%d")

        if pd.isnull(recovery):
            df_drawdowns.loc[i, "Recovery date"] = recovery
        else:
            df_drawdowns.loc[i, "Recovery date"] = pd.to_datetime(recovery).strftime("%Y-%m-%d")

        df_drawdowns.loc[i, "Net drawdown in %"] = (
            (df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[peak]
        ) * 100

    df_drawdowns["Peak date"] = pd.to_datetime(df_drawdowns["Peak date"])
    df_drawdowns["Valley date"] = pd.to_datetime(df_drawdowns["Valley date"])
    df_drawdowns["Recovery date"] = pd.to_datetime(df_drawdowns["Recovery date"])

    return df_drawdowns


def get_max_drawdown_period(returns):
    """Get the start and end dates of the maximum drawdown period.

    Parameters
    ----------
    returns : pd.Series
        Non-cumulative returns indexed by date.

    Returns
    -------
    (pd.Timestamp, pd.Timestamp)
        Tuple ``(start_date, end_date)`` of the maximum drawdown period,
        or ``(None, None)`` if it cannot be determined.
    """
    if len(returns) < 1:
        return None, None

    cum_ret = cum_returns(returns, starting_value=1)

    if not isinstance(cum_ret, pd.Series):
        return None, None

    # Calculate rolling maximum
    rolling_max = cum_ret.expanding().max()

    # Calculate drawdown
    drawdown = cum_ret / rolling_max - 1

    # Find the end date of maximum drawdown
    end_date = drawdown.idxmin()

    # Find the start date of maximum drawdown (previous peak)
    start_date = cum_ret.loc[:end_date].idxmax()

    return start_date, end_date


def max_drawdown_days(returns):
    """Calculate the duration of the maximum drawdown in days.

    Parameters
    ----------
    returns : pd.Series or array-like
        Non-cumulative returns indexed by date or position.

    Returns
    -------
    int or float
        Number of days (or periods) between the peak and trough of the
        maximum drawdown, or ``NaN`` if it cannot be determined.
    """
    if len(returns) < 1:
        return np.nan

    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    cum_ret = cum_returns(returns, starting_value=100)
    rolling_max = cum_ret.expanding().max()
    drawdown = (cum_ret - rolling_max) / rolling_max

    end_idx = drawdown.idxmin()
    start_idx = cum_ret.loc[:end_idx].idxmax()

    if isinstance(returns.index, pd.DatetimeIndex):
        start_pos = returns.index.get_loc(start_idx)
        end_pos = returns.index.get_loc(end_idx)
        return end_pos - start_pos
    else:
        start_pos = returns.index.get_loc(start_idx)
        end_pos = returns.index.get_loc(end_idx)
        return end_pos - start_pos


def max_drawdown_weeks(returns):
    """Calculate the duration of the maximum drawdown in weeks.

    Parameters
    ----------
    returns : pd.Series or array-like
        Non-cumulative returns indexed by date or position.

    Returns
    -------
    int or float
        Number of weeks between the peak and trough of the maximum
        drawdown, or ``NaN`` if it cannot be determined.
    """
    return max_drawdown_days(returns)


def max_drawdown_months(returns):
    """Calculate the duration of the maximum drawdown in months.

    Parameters
    ----------
    returns : pd.Series or array-like
        Non-cumulative returns indexed by date or position.

    Returns
    -------
    int or float
        Number of months between the peak and trough of the maximum
        drawdown, or ``NaN`` if it cannot be determined.
    """
    return max_drawdown_days(returns)


def max_drawdown_recovery_days(returns):
    """Calculate the recovery time from maximum drawdown in days.

    This computes the number of days (or periods) from the trough of the
    maximum drawdown until the portfolio value recovers to the previous
    peak.

    Parameters
    ----------
    returns : pd.Series or array-like
        Non-cumulative returns indexed by date or position.

    Returns
    -------
    int or float
        Number of days (or periods) from the trough to recovery, or
        ``NaN`` if recovery does not occur or cannot be determined.
    """
    if len(returns) < 1:
        return np.nan

    cum_ret = cum_returns(returns, starting_value=1)

    if not isinstance(cum_ret, pd.Series):
        return np.nan

    rolling_max = cum_ret.expanding().max()
    drawdown = cum_ret / rolling_max - 1

    max_dd_date = drawdown.idxmin()
    post_dd_data = cum_ret.loc[max_dd_date:]
    recovery_level = rolling_max.loc[max_dd_date]

    recovery_mask = post_dd_data >= recovery_level
    if recovery_mask.any():
        recovery_date = post_dd_data[recovery_mask].index[0]
        if hasattr(recovery_date - max_dd_date, "days"):
            return (recovery_date - max_dd_date).days
        else:
            return int(recovery_date - max_dd_date)
    else:
        return np.nan


def max_drawdown_recovery_weeks(returns):
    """Calculate the recovery time from maximum drawdown in weeks.

    Parameters
    ----------
    returns : pd.Series or array-like
        Non-cumulative returns indexed by date or position.

    Returns
    -------
    int or float
        Number of weeks from the trough to recovery, or ``NaN`` if
        recovery does not occur or cannot be determined.
    """
    return max_drawdown_recovery_days(returns)


def max_drawdown_recovery_months(returns):
    """Calculate the recovery time from maximum drawdown in months.

    Parameters
    ----------
    returns : pd.Series or array-like
        Non-cumulative returns indexed by date or position.

    Returns
    -------
    int or float
        Number of months from the trough to recovery, or ``NaN`` if
        recovery does not occur or cannot be determined.
    """
    return max_drawdown_recovery_days(returns)


def second_max_drawdown(returns):
    """Determine the second-largest drawdown of a strategy.

    This identifies all distinct drawdown periods and returns the
    second-most severe (second-most negative) drawdown.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.

    Returns
    -------
    float
        Second-largest drawdown, or ``NaN`` if there are fewer than two
        drawdown periods.
    """
    drawdown_periods = get_all_drawdowns(returns)

    if len(drawdown_periods) < 2:
        return np.nan

    sorted_drawdowns = np.sort(drawdown_periods)
    return sorted_drawdowns[-2]


def third_max_drawdown(returns):
    """Determine the third-largest drawdown of a strategy.

    This identifies all distinct drawdown periods and returns the
    third-most severe (third-most negative) drawdown.

    Parameters
    ----------
    returns : array-like or pd.Series
        Non-cumulative strategy returns.

    Returns
    -------
    float
        Third-largest drawdown, or ``NaN`` if there are fewer than three
        drawdown periods.
    """
    drawdown_periods = get_all_drawdowns(returns)

    if len(drawdown_periods) < 3:
        return np.nan

    sorted_drawdowns = np.sort(drawdown_periods)
    return sorted_drawdowns[-3]


def second_max_drawdown_days(returns):
    """Calculate the duration of the second maximum drawdown in days.

    Parameters
    ----------
    returns : pd.Series or array-like
        Non-cumulative returns indexed by date or position.

    Returns
    -------
    int or float
        Number of days (or periods) between the peak and trough of the
        second-largest drawdown, or ``NaN`` if there are fewer than two
        drawdown periods.
    """
    drawdown_periods = get_all_drawdowns_detailed(returns)

    if len(drawdown_periods) < 2:
        return np.nan

    sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x["value"])
    return sorted_drawdowns[1]["duration"]


def second_max_drawdown_recovery_days(returns):
    """Calculate the recovery time from the second maximum drawdown in days.

    Parameters
    ----------
    returns : pd.Series or array-like
        Non-cumulative returns indexed by date or position.

    Returns
    -------
    int or float
        Number of days from the trough to recovery for the second-largest
        drawdown, or ``NaN`` if recovery does not occur or there are fewer
        than two drawdown periods.
    """
    drawdown_periods = get_all_drawdowns_detailed(returns)

    if len(drawdown_periods) < 2:
        return np.nan

    sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x["value"])
    recovery_duration = sorted_drawdowns[1]["recovery_duration"]
    return recovery_duration if recovery_duration is not None else np.nan


def third_max_drawdown_days(returns):
    """Calculate the duration of the third maximum drawdown in days.

    Parameters
    ----------
    returns : pd.Series or array-like
        Non-cumulative returns indexed by date or position.

    Returns
    -------
    int or float
        Number of days (or periods) between the peak and trough of the
        third-largest drawdown, or ``NaN`` if there are fewer than three
        drawdown periods.
    """
    drawdown_periods = get_all_drawdowns_detailed(returns)

    if len(drawdown_periods) < 3:
        return np.nan

    sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x["value"])
    return sorted_drawdowns[2]["duration"]


def third_max_drawdown_recovery_days(returns):
    """Calculate the recovery time from the third maximum drawdown in days.

    Parameters
    ----------
    returns : pd.Series or array-like
        Non-cumulative returns indexed by date or position.

    Returns
    -------
    int or float
        Number of days from the trough to recovery for the third-largest
        drawdown, or ``NaN`` if recovery does not occur or there are fewer
        than three drawdown periods.
    """
    drawdown_periods = get_all_drawdowns_detailed(returns)

    if len(drawdown_periods) < 3:
        return np.nan

    sorted_drawdowns = sorted(drawdown_periods, key=lambda x: x["value"])
    recovery_duration = sorted_drawdowns[2]["recovery_duration"]
    return recovery_duration if recovery_duration is not None else np.nan
