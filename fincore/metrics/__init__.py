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

"""
Metrics - 拆分后的金融性能分析函数模块.

这个包包含从 Empyrical 类中拆分出来的各个功能模块。
"""

# 导入子模块（保留模块引用，使用别名避免被覆盖）
from fincore.metrics import basic as basic_module
from fincore.metrics import returns as returns_module
from fincore.metrics import drawdown as drawdown_module
from fincore.metrics import risk as risk_module
from fincore.metrics import ratios as ratios_module
from fincore.metrics import alpha_beta as alpha_beta_module
from fincore.metrics import stats as stats_module
from fincore.metrics import consecutive as consecutive_module
from fincore.metrics import rolling as rolling_module
from fincore.metrics import bayesian as bayesian_module
from fincore.metrics import positions as positions_module
from fincore.metrics import transactions as transactions_module
from fincore.metrics import round_trips as round_trips_module
from fincore.metrics import perf_attrib as perf_attrib_module
from fincore.metrics import perf_stats as perf_stats_module
from fincore.metrics import timing as timing_module
from fincore.metrics import yearly as yearly_module