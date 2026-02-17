### 修复import

现在已经对pyfolio进行了重构，现在需要在test_pyfolio这个文件夹中，针对这些测试用例，修复import的问题。

`from fincore import Pyfolio`

### 修复函数的使用

现在pyfolio只有一个类，所有的函数都是这个类的成员函数，所以，使用到pyfolio相关函数的测试用例，想要把pyfolio进行实例化，然后调用成员函数。

例如：

```python
pyfolio = Pyfolio(returns=simple_return)
pyfolio.plot_bayes_cone(returns_train, returns_test, preds, plot_train_len=None, ax=None)
```

### 限制

1. 测试用例不允许修改输入值和期望值，这些都是正确的，已经验证过的。
2. 做的修改需要在这个文档的下面进行记录说明。

### 验收标准
1. test_pyfolio文件夹中的所有的测试用例都能够成功运行。

### 修改记录

1. test_pyfolio/test_imports.py
   - 改为 `import fincore`，`from fincore import Pyfolio`，并从 `fincore.utils.common_utils` 导入 `HAS_IPYTHON` 等工具，验证新包结构下的导入。

2. test_pyfolio/test_tears.py
   - 改为 `from fincore import Pyfolio`，`from fincore.utils.common_utils import to_utc, to_series`。
   - 不再从 `pyfolio.tears` 导入函数，而是在用例中实例化 `Pyfolio()`，调用对应成员方法：
     `create_full_tear_sheet`、`create_simple_tear_sheet`、`create_returns_tear_sheet`、`create_position_tear_sheet`、`create_txn_tear_sheet`、`create_round_trip_tear_sheet`、`create_interesting_times_tear_sheet`。

3. test_pyfolio/test_timeseries.py
   - 将 `from pyfolio import timeseries` 替换为 `from fincore.empyrical import Empyrical`，并从 `fincore.utils.common_utils` 导入 `to_utc`、`to_series`、`get_month_end_freq`。
   - 所有 timeseries 级别计算函数改为调用 `Empyrical` 类方法：`gen_drawdown_table`、`get_max_drawdown`、`get_top_drawdowns`、`var_cov_var_normal`、`normalize`、`rolling_sharpe`、`rolling_beta`、`forecast_cone_bootstrap`、`calc_bootstrap`、`gross_lev`。

4. test_pyfolio/test_txn.py
   - 不再从 `pyfolio.txn` 导入函数，改为 `from fincore.empyrical import Empyrical`。
   - `get_turnover`、`adjust_returns_for_slippage` 改为调用 `Empyrical.get_turnover`、`Empyrical.adjust_returns_for_slippage`。
   - 差异分析工具从 `fincore.utils.common_utils` 导入 `analyze_dataframe_differences`、`analyze_series_differences`。

5. test_pyfolio/test_capacity.py
   - 不再从 `pyfolio.capacity` 导入函数，改为 `from fincore.empyrical import Empyrical`。
   - 对应函数改为调用 `Empyrical.days_to_liquidate_positions`、`Empyrical.get_max_days_to_liquidate_by_ticker`、`Empyrical.get_low_liquidity_transactions`、`Empyrical.daily_txns_with_bar_data`、`Empyrical.apply_slippage_penalty`。
   - 差异分析工具从 `fincore.utils.common_utils` 导入。

6. test_pyfolio/test_pos.py
   - 工具函数从 `fincore.utils.common_utils` 导入：`to_utc`、`to_series`、`check_intraday`、`detect_intraday`、`estimate_intraday`。
   - 头寸相关计算从 `fincore.empyrical.Empyrical` 调用：`get_percent_alloc`、`extract_pos`、`get_sector_exposures`、`get_max_median_position_concentration`。

7. test_pyfolio/test_round_trips.py
   - 不再从 `pyfolio.round_trips` 导入函数，改为从 `fincore.empyrical` 使用：`Empyrical._groupby_consecutive`、`Empyrical.extract_round_trips`、`Empyrical.add_closing_transactions`。

8. test_pyfolio/test_perf_attrib.py
   - 不再从 `pyfolio.perf_attrib` 导入函数，改为从 `fincore.empyrical` 导入 `Empyrical`，并在模块顶部定义别名：
     `perf_attrib = Empyrical.perf_attrib`，
     `create_perf_attrib_stats = Empyrical.create_perf_attrib_stats`，
     `_cumulative_returns_less_costs = Empyrical._cumulative_returns_less_costs`。

以上修改仅调整 import 和调用入口，所有测试用例中的输入数据和期望结果保持不变。
