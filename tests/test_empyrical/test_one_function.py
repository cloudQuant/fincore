import numpy as np
import pandas as pd
from numpy.ma.testutils import assert_almost_equal


def beta_fragility_heuristic_aligned(returns, factor_returns):
    """Estimate fragility to drop in beta

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in: func:`~empyrical.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.

    Returns
    -------
    float, np.nan
        The beta fragility of the strategy.

    Note
    ----
    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.
    See also::
    `A New Heuristic Measure of Fragility and
Tail Risks: Application to Stress Testing`
        https://www.imf.org/external/pubs/ft/wp/2012/wp12216.pdf
        An IMF Working Paper describing the heuristic
    """
    if len(returns) < 3 or len(factor_returns) < 3:
        return np.nan

    # combine returns and factor returns into pairs
    returns_series = pd.Series(returns)
    factor_returns_series = pd.Series(factor_returns)
    pairs = pd.concat([returns_series, factor_returns_series], axis=1)
    pairs.columns = ['returns', 'factor_returns']

    # exclude any rows where returns are nan
    pairs = pairs.dropna()
    # sort by beta
    pairs = pairs.sort_values(by=['factor_returns'], kind='mergesort')
    print(pairs)
    # find the three vectors, using median of 3
    start_index = 0
    mid_index = int(np.around(len(pairs) / 2, 0))
    end_index = len(pairs) - 1

    (start_returns, start_factor_returns) = pairs.iloc[start_index]
    (mid_returns, mid_factor_returns) = pairs.iloc[mid_index]
    (end_returns, end_factor_returns) = pairs.iloc[end_index]

    factor_returns_range = (end_factor_returns - start_factor_returns)
    start_returns_weight = 0.5
    end_returns_weight = 0.5

    # find weights for the start and end returns
    # using a convex combination
    if not factor_returns_range == 0:
        start_returns_weight = \
            (mid_factor_returns - start_factor_returns) / \
            factor_returns_range
        end_returns_weight = \
            (end_factor_returns - mid_factor_returns) / \
            factor_returns_range

    # calculate fragility heuristic
    heuristic = (start_returns_weight * start_returns) + \
                (end_returns_weight * end_returns) - mid_returns

    return heuristic

if __name__ == '__main__':
    mixed_returns = pd.Series(
        np.array([np.nan, 1., 10., -4., 2., 3., 2., 1., -10.]) / 100,
        index=pd.date_range('2000-1-30', periods=9, freq='D'))
    simple_benchmark = pd.Series(
        np.array([0., 1., 0., 1., 0., 1., 0., 1., 0.]) / 100,
        index=pd.date_range('2000-1-30', periods=9, freq='D'))
    actual_value = beta_fragility_heuristic_aligned(mixed_returns, simple_benchmark)
    expected_value = 0.09
    assert_almost_equal(actual_value, expected_value)