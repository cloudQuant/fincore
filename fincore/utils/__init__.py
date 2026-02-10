#
# Copyright 2018 Quantopian, Inc.
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

"""
工具函数模块

从原有empyrical/utils.py重构而来的工具函数集合。
"""

from .common_utils import (
    SETTINGS,
    _1_bday_ago,
    analyze_dataframe_differences,
    analyze_series_differences,
    check_intraday,
    clip_returns_to_benchmark,
    configure_legend,
    customize,
    default_returns_func,
    detect_intraday,
    estimate_intraday,
    extract_rets_pos_txn_from_zipline,
    format_asset,
    get_month_end_freq,
    get_symbol_rets,
    get_utc_timestamp,
    make_timezone_aware,
    one_dec_places,
    percentage,
    print_table,
    register_return_func,
    sample_colormap,
    standardize_data,
    to_series,
    to_utc,
    two_dec_places,
    vectorize,
)
from .data_utils import down, roll, rolling_window, up
from .date_utils import timer
from .deprecate import deprecated
from .math_utils import (
    nanargmax,
    nanargmin,
    nanmax,
    nanmean,
    nanmin,
    nanstd,
    nansum,
)

__all__ = [
    # math_utils
    "nanmean",
    "nanstd",
    "nansum",
    "nanmax",
    "nanmin",
    "nanargmax",
    "nanargmin",
    # data_utils
    "rolling_window",
    "roll",
    "up",
    "down",
    # common_utils
    "customize",
    "analyze_dataframe_differences",
    "analyze_series_differences",
    "one_dec_places",
    "two_dec_places",
    "percentage",
    "format_asset",
    "vectorize",
    "extract_rets_pos_txn_from_zipline",
    "print_table",
    "standardize_data",
    "detect_intraday",
    "check_intraday",
    "estimate_intraday",
    "clip_returns_to_benchmark",
    "to_utc",
    "to_series",
    "get_month_end_freq",
    "make_timezone_aware",
    "SETTINGS",
    "register_return_func",
    "get_symbol_rets",
    "configure_legend",
    "sample_colormap",
    "get_utc_timestamp",
    "_1_bday_ago",
    "default_returns_func",
    # deprecate
    "deprecated",
    # date_utils
    "timer",
]
