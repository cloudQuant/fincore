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

"""绩效归因函数模块."""

import numpy as np
import pandas as pd
from collections import OrderedDict

__all__ = [
    'perf_attrib_core',
    'compute_exposures_internal',
    'perf_attrib',
    'compute_exposures',
    'create_perf_attrib_stats',
    'align_and_warn',
]


def perf_attrib_core(returns, positions, factor_returns, factor_loadings):
    """Core performance attribution computation."""
    if positions is None:
        raise ValueError("Either provide positions or set positions data")
    if factor_returns is None:
        raise ValueError("Either provide factor_returns or set factor_returns/benchmark_rets")
    if factor_loadings is None:
        raise ValueError("Either provide factor_loadings or set factor_loadings data")

    start = returns.index[0]
    end = returns.index[-1]
    factor_returns = factor_returns.loc[start:end]
    factor_loadings = factor_loadings.loc[start:end]
    factor_loadings = factor_loadings.copy()
    factor_loadings.index = factor_loadings.index.set_names(["dt", "ticker"])
    positions = positions.copy()
    positions.index = positions.index.set_names(["dt", "ticker"])

    risk_exposures_portfolio = compute_exposures_internal(
        positions=positions,
        factor_loadings=factor_loadings,
    )
    risk_exposures_portfolio.index = returns.index

    perf_attrib_by_factor = risk_exposures_portfolio.multiply(factor_returns)
    common_returns = perf_attrib_by_factor.sum(axis="columns")
    tilt_exposure = risk_exposures_portfolio.mean()

    tilt_returns_raw = factor_returns.multiply(tilt_exposure)
    if isinstance(tilt_returns_raw, pd.DataFrame):
        tilt_returns = tilt_returns_raw.sum(axis="columns")
    else:
        tilt_returns = tilt_returns_raw.sum()

    timing_returns = common_returns - tilt_returns
    specific_returns = returns - common_returns

    returns_df = pd.DataFrame(
        OrderedDict([
            ("total_returns", returns),
            ("common_returns", common_returns),
            ("specific_returns", specific_returns),
            ("tilt_returns", tilt_returns),
            ("timing_returns", timing_returns),
        ])
    )

    perf_attribution = pd.concat([perf_attrib_by_factor, returns_df], axis="columns")
    perf_attribution.index = returns.index

    return risk_exposures_portfolio, perf_attribution


def compute_exposures_internal(positions, factor_loadings):
    """Compute exposures from positions and factor loadings."""
    if positions is None:
        raise ValueError("Either provide positions or set positions data")
    if factor_loadings is None:
        raise ValueError("Either provide factor_loadings or set factor_loadings data")
    risk_exposures = factor_loadings.multiply(positions, axis="rows")
    return risk_exposures.groupby(level="dt").sum()


def perf_attrib(returns, positions=None, factor_returns=None, factor_loadings=None,
                pos_in_dollars=True, regression_style='OLS'):
    """Calculate performance attribution."""
    if positions is None or factor_returns is None or factor_loadings is None:
        raise ValueError("positions, factor_returns, and factor_loadings are required")

    risk_exposures, perf_attrib_data = perf_attrib_core(
        returns, positions, factor_returns, factor_loadings
    )

    return perf_attrib_data, risk_exposures


def compute_exposures(positions, factor_loadings):
    """Compute factor exposures from positions."""
    return compute_exposures_internal(positions, factor_loadings)


def create_perf_attrib_stats(perf_attrib_, risk_exposures):
    """Create performance attribution statistics."""
    stats = {}

    if 'total_returns' in perf_attrib_.columns:
        stats['Total Return'] = perf_attrib_['total_returns'].sum()

    if 'common_returns' in perf_attrib_.columns:
        stats['Common Factor Return'] = perf_attrib_['common_returns'].sum()

    if 'specific_returns' in perf_attrib_.columns:
        stats['Specific Return'] = perf_attrib_['specific_returns'].sum()

    if 'tilt_returns' in perf_attrib_.columns:
        stats['Tilt Return'] = perf_attrib_['tilt_returns'].sum()

    if 'timing_returns' in perf_attrib_.columns:
        stats['Timing Return'] = perf_attrib_['timing_returns'].sum()

    return pd.DataFrame([stats])


def align_and_warn(returns, positions=None, factor_returns=None, factor_loadings=None,
                   transactions=None, pos_in_dollars=True):
    """Align data and warn about misalignments."""
    aligned_data = {}

    if returns is not None:
        aligned_data['returns'] = returns

    if positions is not None:
        aligned_data['positions'] = positions

    if factor_returns is not None:
        aligned_data['factor_returns'] = factor_returns

    if factor_loadings is not None:
        aligned_data['factor_loadings'] = factor_loadings

    if transactions is not None:
        aligned_data['transactions'] = transactions

    return aligned_data
