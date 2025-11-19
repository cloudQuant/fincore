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
    nanmean, nanstd, nansum, nanmax, nanmin, nanargmax, nanargmin
)

from .data_utils import (
    rolling_window, up, down, roll
)

def default_returns_func(returns):
    """
    Default returns processing function.
    
    This is a simple passthrough function that returns the input as-is.
    It's used by pyfolio for returns processing compatibility.
    """
    return returns

__all__ = [
    'nanmean', 'nanstd', 'nansum', 'nanmax', 'nanmin', 'nanargmax', 'nanargmin',
    'rolling_window', 'up', 'down', 'roll', 'default_returns_func'
]
