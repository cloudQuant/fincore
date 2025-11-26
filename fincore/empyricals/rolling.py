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

"""滚动计算函数模块."""

import numpy as np
import pandas as pd
from fincore.constants import DAILY
from fincore.empyricals.basic import aligned_series
from fincore.empyricals.alpha_beta import alpha, beta
from fincore.empyricals.ratios import sharpe_ratio, up_capture, down_capture
from fincore.empyricals.drawdown import max_drawdown

__all__ = [
    'roll_alpha',
    'roll_beta',
    'roll_alpha_beta',
    'roll_sharpe_ratio',
    'roll_max_drawdown',
    'roll_up_capture',
    'roll_down_capture',
    'roll_up_down_capture',
    'rolling_volatility',
    'rolling_sharpe',
    'rolling_beta',
    'rolling_regression',
]


def roll_alpha(returns, factor_returns, window=252, risk_free=0.0, period=DAILY, annualization=None):
    """Calculate rolling alpha over a specified window."""
    returns_aligned, factor_aligned = aligned_series(returns, factor_returns)
    
    # Convert generators to lists/Series
    if not isinstance(returns_aligned, (pd.Series, np.ndarray)):
        returns_aligned = pd.Series(list(returns_aligned))
    if not isinstance(factor_aligned, (pd.Series, np.ndarray)):
        factor_aligned = pd.Series(list(factor_aligned))

    if len(returns_aligned) < window:
        return pd.Series([], dtype=float)

    rolling_alphas = []
    for i in range(window, len(returns_aligned) + 1):
        window_returns = returns_aligned.iloc[i - window: i]
        window_factor = factor_aligned.iloc[i - window: i]

        alpha_val = alpha(window_returns, window_factor, risk_free, period, annualization)
        rolling_alphas.append(alpha_val)

    if isinstance(returns_aligned, pd.Series):
        return pd.Series(rolling_alphas, index=returns_aligned.index[window - 1:])
    else:
        return pd.Series(rolling_alphas)


def roll_beta(returns, factor_returns, window=252, risk_free=0.0, period=DAILY, annualization=None):
    """Calculate rolling beta over a specified window."""
    returns_aligned, factor_aligned = aligned_series(returns, factor_returns)
    
    if not isinstance(returns_aligned, (pd.Series, np.ndarray)):
        returns_aligned = pd.Series(list(returns_aligned))
    if not isinstance(factor_aligned, (pd.Series, np.ndarray)):
        factor_aligned = pd.Series(list(factor_aligned))

    if len(returns_aligned) < window:
        return pd.Series([], dtype=float)

    rolling_betas = []
    for i in range(window, len(returns_aligned) + 1):
        window_returns = returns_aligned.iloc[i - window: i]
        window_factor = factor_aligned.iloc[i - window: i]

        beta_val = beta(window_returns, window_factor, risk_free, period, annualization)
        rolling_betas.append(beta_val)

    if isinstance(returns_aligned, pd.Series):
        return pd.Series(rolling_betas, index=returns_aligned.index[window - 1:])
    else:
        return pd.Series(rolling_betas)


def roll_alpha_beta(returns, factor_returns, window=252, risk_free=0.0, period=DAILY, annualization=None):
    """Calculate rolling alpha and beta over a specified window."""
    returns_aligned, factor_aligned = aligned_series(returns, factor_returns)
    
    if not isinstance(returns_aligned, (pd.Series, np.ndarray)):
        returns_aligned = pd.Series(list(returns_aligned))
    if not isinstance(factor_aligned, (pd.Series, np.ndarray)):
        factor_aligned = pd.Series(list(factor_aligned))

    if len(returns_aligned) < window:
        return pd.DataFrame(columns=['alpha', 'beta'])

    rolling_results = []
    for i in range(window, len(returns_aligned) + 1):
        window_returns = returns_aligned.iloc[i - window: i]
        window_factor = factor_aligned.iloc[i - window: i]

        alpha_val = alpha(window_returns, window_factor, risk_free, period, annualization)
        beta_val = beta(window_returns, window_factor, risk_free, period, annualization)
        rolling_results.append({'alpha': alpha_val, 'beta': beta_val})

    if isinstance(returns_aligned, pd.Series):
        return pd.DataFrame(rolling_results, index=returns_aligned.index[window - 1:])
    else:
        return pd.DataFrame(rolling_results)


def roll_sharpe_ratio(returns, window=252, risk_free=0.0, period=DAILY, annualization=None):
    """Calculate rolling Sharpe ratio over a specified window."""
    if len(returns) < window:
        return pd.Series([], dtype=float)

    rolling_sharpes = []
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i - window: i]
        sharpe_val = sharpe_ratio(window_returns, risk_free, period, annualization)
        rolling_sharpes.append(sharpe_val)

    if isinstance(returns, pd.Series):
        return pd.Series(rolling_sharpes, index=returns.index[window - 1:])
    else:
        return pd.Series(rolling_sharpes)


def roll_max_drawdown(returns, window=252):
    """Calculate rolling maximum drawdown over a specified window."""
    if len(returns) < window:
        return pd.Series([], dtype=float)

    rolling_mdd = []
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i - window: i]
        mdd_val = max_drawdown(window_returns)
        rolling_mdd.append(mdd_val)

    if isinstance(returns, pd.Series):
        return pd.Series(rolling_mdd, index=returns.index[window - 1:])
    else:
        return pd.Series(rolling_mdd)


def roll_up_capture(returns, factor_returns, window=252):
    """Calculate rolling up capture over a specified window."""
    returns_aligned, factor_aligned = aligned_series(returns, factor_returns)
    
    if not isinstance(returns_aligned, (pd.Series, np.ndarray)):
        returns_aligned = pd.Series(list(returns_aligned))
    if not isinstance(factor_aligned, (pd.Series, np.ndarray)):
        factor_aligned = pd.Series(list(factor_aligned))

    if len(returns_aligned) < window:
        return pd.Series([], dtype=float)

    rolling_up_caps = []
    for i in range(window, len(returns_aligned) + 1):
        window_returns = returns_aligned.iloc[i - window: i]
        window_factor = factor_aligned.iloc[i - window: i]

        up_cap_val = up_capture(window_returns, window_factor)
        rolling_up_caps.append(up_cap_val)

    if isinstance(returns_aligned, pd.Series):
        return pd.Series(rolling_up_caps, index=returns_aligned.index[window - 1:])
    else:
        return pd.Series(rolling_up_caps)


def roll_down_capture(returns, factor_returns, window=252):
    """Calculate rolling down capture over a specified window."""
    returns_aligned, factor_aligned = aligned_series(returns, factor_returns)
    
    if not isinstance(returns_aligned, (pd.Series, np.ndarray)):
        returns_aligned = pd.Series(list(returns_aligned))
    if not isinstance(factor_aligned, (pd.Series, np.ndarray)):
        factor_aligned = pd.Series(list(factor_aligned))

    if len(returns_aligned) < window:
        return pd.Series([], dtype=float)

    rolling_down_caps = []
    for i in range(window, len(returns_aligned) + 1):
        window_returns = returns_aligned.iloc[i - window: i]
        window_factor = factor_aligned.iloc[i - window: i]

        down_cap_val = down_capture(window_returns, window_factor)
        rolling_down_caps.append(down_cap_val)

    if isinstance(returns_aligned, pd.Series):
        return pd.Series(rolling_down_caps, index=returns_aligned.index[window - 1:])
    else:
        return pd.Series(rolling_down_caps)


def roll_up_down_capture(returns, factor_returns, window=252):
    """Calculate rolling up/down capture ratio over a specified window."""
    up_caps = roll_up_capture(returns, factor_returns, window)
    down_caps = roll_down_capture(returns, factor_returns, window)

    with np.errstate(divide="ignore", invalid="ignore"):
        return up_caps / down_caps


def rolling_volatility(returns, rolling_vol_window):
    """Calculate rolling volatility."""
    return returns.rolling(window=rolling_vol_window).std() * np.sqrt(252)


def rolling_sharpe(returns, rolling_sharpe_window):
    """Calculate rolling Sharpe ratio."""
    rolling_mean = returns.rolling(window=rolling_sharpe_window).mean()
    rolling_std = returns.rolling(window=rolling_sharpe_window).std()

    with np.errstate(divide="ignore", invalid="ignore"):
        return rolling_mean / rolling_std * np.sqrt(252)


def rolling_beta(returns, factor_returns, rolling_window=126):
    """Calculate rolling beta."""
    returns_aligned, factor_aligned = aligned_series(returns, factor_returns)
    
    if not isinstance(returns_aligned, (pd.Series, np.ndarray)):
        returns_aligned = pd.Series(list(returns_aligned))
    if not isinstance(factor_aligned, (pd.Series, np.ndarray)):
        factor_aligned = pd.Series(list(factor_aligned))

    if len(returns_aligned) < rolling_window:
        return pd.Series([], dtype=float)

    out = []
    for beg, end in zip(range(0, len(returns_aligned) - rolling_window + 1),
                        range(rolling_window, len(returns_aligned) + 1)):
        window_returns = returns_aligned.iloc[beg:end]
        window_factor = factor_aligned.iloc[beg:end]

        beta_val = beta(window_returns, window_factor)
        out.append(beta_val)

    if isinstance(returns_aligned, pd.Series):
        return pd.Series(out, index=returns_aligned.index[rolling_window - 1:])
    else:
        return pd.Series(out)


def rolling_regression(returns, factor_returns, rolling_window=126):
    """Calculate rolling regression alpha and beta."""
    returns_aligned, factor_aligned = aligned_series(returns, factor_returns)
    
    if not isinstance(returns_aligned, (pd.Series, np.ndarray)):
        returns_aligned = pd.Series(list(returns_aligned))
    if not isinstance(factor_aligned, (pd.Series, np.ndarray)):
        factor_aligned = pd.Series(list(factor_aligned))

    if len(returns_aligned) < rolling_window:
        return pd.DataFrame(columns=['alpha', 'beta'])

    out = []
    for beg, end in zip(range(0, len(returns_aligned) - rolling_window + 1),
                        range(rolling_window, len(returns_aligned) + 1)):
        window_returns = returns_aligned.iloc[beg:end]
        window_factor = factor_aligned.iloc[beg:end]

        alpha_val = alpha(window_returns, window_factor)
        beta_val = beta(window_returns, window_factor)
        out.append({'alpha': alpha_val, 'beta': beta_val})

    if isinstance(returns_aligned, pd.Series):
        return pd.DataFrame(out, index=returns_aligned.index[rolling_window - 1:])
    else:
        return pd.DataFrame(out)
