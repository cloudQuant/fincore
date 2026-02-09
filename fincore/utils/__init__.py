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
from .math_utils import (
    nanmean, nanstd, nansum, nanmax, nanmin, nanargmax, nanargmin,
)
from .data_utils import rolling_window, roll, up, down
from .common_utils import (
    customize,
    analyze_dataframe_differences,
    analyze_series_differences,
    one_dec_places,
    two_dec_places,
    percentage,
    format_asset,
    vectorize,
    extract_rets_pos_txn_from_zipline,
    print_table,
    standardize_data,
    detect_intraday,
    check_intraday,
    estimate_intraday,
    clip_returns_to_benchmark,
    to_utc,
    to_series,
    get_month_end_freq,
    make_timezone_aware,
    SETTINGS,
    register_return_func,
    get_symbol_rets,
    configure_legend,
    sample_colormap,
    get_utc_timestamp,
    _1_bday_ago,
    default_returns_func,
)
from .deprecate import deprecated
from .date_utils import timer

__all__ = [
    # math_utils
    "nanmean", "nanstd", "nansum", "nanmax", "nanmin", "nanargmax", "nanargmin",
    # data_utils
    "rolling_window", "roll", "up", "down",
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
