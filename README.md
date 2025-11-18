# fincore | Quantitative Performance & Risk Analytics

<p align="center">
    <img src="https://img.shields.io/badge/version-0.1-blueviolet.svg" alt="Version 0.1" style="margin-right: 10px;"/>
    <img src="https://github.com/cloudQuant/fincore/workflows/Tests/badge.svg" alt="Tests" style="margin-right: 10px;"/>
    <img src="https://img.shields.io/badge/platform-mac%7Clinux%7Cwin-yellow.svg" alt="Supported Platforms: Mac, Linux, and Windows" style="margin-right: 10px;"/>
    <img src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-brightgreen.svg" alt="Python Versions" style="margin-right: 10px;"/>
    <img src="https://img.shields.io/badge/license-MIT-orange" alt="License: MIT"/>
</p>

**[English](#english-version) | [中文](#chinese-version)**

---

<a name="english-version"></a>
## English

### Overview

fincore is a Python library for calculating common financial risk and performance metrics. It continues the empyrical analytics stack under active maintenance by cloudQuant, providing a comprehensive toolkit for quantitative finance professionals and researchers to analyze investment returns, calculate risk metrics, and perform performance attribution.

### Features

- **Comprehensive Metrics**: Over 50 financial metrics including returns, risk, risk-adjusted returns, and market relationships
- **Rolling Calculations**: Rolling window versions of most metrics for time-series analysis
- **Flexible Input**: Supports pandas Series/DataFrame and numpy arrays
- **NaN Handling**: Robust handling of missing data throughout all calculations
- **Performance Attribution**: Factor-based performance decomposition
- **Period Flexibility**: Automatic period detection and support for daily, weekly, monthly, quarterly, and yearly data (via `fincore.empyrical.periods`)

### Implemented Indicators

The library exposes a comprehensive indicator suite (everything exported by `fincore.empyrical`). The catalogue below lists every metric grouped by theme with a short description of what it measures.

#### Returns & Growth Dynamics
- `simple_returns`, `cum_returns`, `cum_returns_final` — Convert raw prices into non-cumulative or cumulative return series.
- `aggregate_returns` — Resample daily returns into weekly, monthly, quarterly, or yearly aggregates via compounding.
- `annual_return`, `annual_return_by_year`, `annualized_cumulative_return`, `annual_active_return`, `annual_active_return_by_year`, `roll_annual_active_return`, `regression_annual_return` — Annualize total and active performance, including regression-based and rolling variants.
- `cagr` — Compute compound annual growth over multi-period investment horizons.
- `stability_of_timeseries` — R-squared style stability check against a linear trend of cumulative returns.

#### Volatility, Risk & Drawdown
- `annual_volatility`, `annual_volatility_by_year`, `roll_annual_volatility` — Annualized standard deviation for total returns on static and rolling windows.
- `annual_active_risk`, `roll_annual_active_risk` — Annualized tracking-volatility versus a benchmark, with rolling support.
- `downside_risk` — Semi-deviation that only penalizes returns below a required threshold.
- `value_at_risk`, `conditional_value_at_risk`, `var_excess_return`, `gpd_risk_estimates`, `gpd_risk_estimates_aligned` — Parametric and EVT tail-loss estimators for absolute and excess returns.
- `max_drawdown`, `max_drawdown_by_year`, `get_max_drawdown_period`, `roll_max_drawdown`, `max_drawdown_days`, `max_drawdown_weeks`, `max_drawdown_months`, `max_drawdown_recovery_days`, `max_drawdown_recovery_weeks`, `max_drawdown_recovery_months`, `second_max_drawdown`, `second_max_drawdown_days`, `second_max_drawdown_recovery_days`, `third_max_drawdown`, `third_max_drawdown_days`, `third_max_drawdown_recovery_days` — Peak-to-trough loss magnitude, timing, and recovery analytics beyond the single worst event.

#### Risk-Adjusted Performance & Efficiency
- `sharpe_ratio`, `sharpe_ratio_by_year`, `roll_sharpe_ratio`, `adjusted_sharpe_ratio`, `conditional_sharpe_ratio`, `stutzer_index` — Reward-to-variability measures with classic, adjusted, conditional, and rolling forms.
- `sortino_ratio`, `roll_sortino_ratio` — Downside-risk adjusted excess returns (static and rolling).
- `calmar_ratio`, `sterling_ratio`, `burke_ratio`, `kappa_three_ratio` — Ratios linking returns to drawdown magnitude using different denominators.
- `omega_ratio`, `tail_ratio` — Probability-weighted gain/loss and tail-balance diagnostics.
- `m_squared`, `roll_m_squared`, `treynor_ratio`, `roll_treynor_ratio` — Modigliani–Modigliani and Treynor style measures of risk-adjusted efficiency.
- `information_ratio`, `information_ratio_by_year`, `excess_sharpe`, `tracking_error`, `roll_tracking_error`, `tracking_difference` — Benchmark-relative risk/return trade-offs and tracking-quality indicators.

#### Alpha, Beta & Capture Analytics
- `alpha`, `alpha_aligned`, `alpha_beta`, `alpha_beta_aligned`, `alpha_percentile_rank`, `roll_alpha`, `roll_alpha_aligned`, `roll_alpha_beta`, `roll_alpha_beta_aligned` — Jensen-style alpha/beta estimators, their aligned-input variants, rolling windows, and cross-sectional percentile ranks.
- `beta`, `beta_aligned`, `beta_fragility_heuristic`, `beta_fragility_heuristic_aligned`, `roll_beta`, `roll_beta_aligned`, `annual_alpha`, `annual_beta`, `residual_risk` — Exposure, fragility, and residual volatility diagnostics for factor models.
- `up_alpha_beta`, `down_alpha_beta`, `capture`, `up_capture`, `down_capture`, `up_down_capture`, `roll_up_capture`, `roll_down_capture`, `roll_up_down_capture` — Conditional regressions and capture ratios for bull/bear market regimes.

#### Timing, Correlation & Market Diagnostics
- `treynor_mazuy_timing`, `henriksson_merton_timing`, `market_timing_return`, `r_cubed`, `cornell_timing` — Market-timing skill measures using non-linear and option-style frameworks.
- `stock_market_correlation`, `bond_market_correlation`, `futures_market_correlation`, `serial_correlation`, `hurst_exponent` — Serial dependence and cross-asset correlation checks.
- `win_rate`, `loss_rate`, `skewness`, `kurtosis` — Distributional moments and hit-rate style statistics.

#### Streak & Event Statistics
- `max_consecutive_up_days`, `max_consecutive_down_days`, `max_consecutive_up_weeks`, `max_consecutive_down_weeks`, `max_consecutive_up_months`, `max_consecutive_down_months` — Longest winning and losing streaks across multiple horizons.
- `max_consecutive_gain`, `max_consecutive_loss`, `max_single_day_gain`, `max_single_day_loss`, `max_single_day_gain_date`, `max_single_day_loss_date` — Magnitude and dating of extreme gain/loss episodes.
- `max_consecutive_up_start_date`, `max_consecutive_up_end_date`, `max_consecutive_down_start_date`, `max_consecutive_down_end_date` — Calendar boundaries for streak analysis.

#### Factor Attribution & Utilities
- `perf_attrib`, `compute_exposures` — Factor-based performance attribution and exposure construction helpers.

#### Constants & Tailored Variants
- `annualized_cumulative_return`, `annual_active_return_by_year`, `annual_active_risk`, `annual_active_return`, `annual_active_return_by_year` — Consolidated with the sections above but retained here for completeness of the export surface.
- `omega_ratio`, `tail_ratio`, `value_at_risk`, `conditional_value_at_risk`, `var_excess_return` — Tail and downside interpreters available to downstream users.
- Period constants `DAILY`, `WEEKLY`, `MONTHLY`, `QUARTERLY`, `YEARLY` (imported from `empyrical.periods`) support annualisation and frequency conversions.

### Installation

#### From Source (Recommended)

```bash
# For users in China
git clone https://gitee.com/yunjinqi/fincore

# For international users
git clone https://github.com/cloudQuant/fincore

cd fincore

# Windows
install_win.bat

# Linux/macOS
sh install_unix.sh
```

#### pip install 

```bash
pip install -U git+https://github.com/cloudQuant/fincore.git
```

### Quick Start

#### Basic Metrics

```python
import numpy as np
from fincore import empyrical

# Sample returns data
returns = np.array([0.01, 0.02, 0.03, -0.4, -0.06, -0.02])
benchmark_returns = np.array([0.02, 0.02, 0.03, -0.35, -0.05, -0.01])

# Calculate max drawdown
mdd = empyrical.max_drawdown(returns)
print(f"Max Drawdown: {mdd:.2%}")

# Calculate Sharpe ratio (assuming daily returns)
sharpe = empyrical.sharpe_ratio(returns, risk_free=0.02/252)
print(f"Sharpe Ratio: {sharpe:.2f}")

# Calculate alpha and beta
alpha, beta = empyrical.alpha_beta(returns, benchmark_returns)
print(f"Alpha: {alpha:.4f}, Beta: {beta:.2f}")
```

#### Rolling Metrics

```python
import pandas as pd
from fincore import empyrical

# Create time series data
dates = pd.date_range('2020-01-01', periods=100, freq='D')
returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)

# Calculate 30-day rolling Sharpe ratio
rolling_sharpe = empyrical.roll_sharpe_ratio(returns, window=30)

# Calculate 30-day rolling max drawdown
rolling_mdd = empyrical.roll_max_drawdown(returns, window=30)
```

#### Advanced Usage with DataFrames

```python
import pandas as pd
from fincore import empyrical

# Multiple strategy returns
strategies = pd.DataFrame({
    'Strategy_A': np.random.normal(0.001, 0.02, 252),
    'Strategy_B': np.random.normal(0.0015, 0.025, 252),
    'Strategy_C': np.random.normal(0.0008, 0.018, 252)
})

# Calculate metrics for all strategies at once
annual_returns = empyrical.annual_return(strategies)
annual_vols = empyrical.annual_volatility(strategies)
calmar_ratios = empyrical.calmar_ratio(strategies)

print("Annual Returns:")
print(annual_returns)
print("\nAnnual Volatilities:")
print(annual_vols)
print("\nCalmar Ratios:")
print(calmar_ratios)
```

### Period Constants

```python
from fincore.empyrical import DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY

# Use period constants for clarity
sharpe_daily = empyrical.sharpe_ratio(returns, period=DAILY)
sharpe_monthly = empyrical.sharpe_ratio(returns, period=MONTHLY)
```

### Testing

```bash
# Run all tests
pytest ./tests -n 4

# Run specific test module
pytest ./tests/test_stats.py

# Run specific test
pytest ./tests/test_stats.py::test_sharpe_ratio
```

### Development

#### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/cloudQuant/fincore
cd fincore

# Create virtual environment (using conda)
conda create -n fincore-dev python=3.11
conda activate fincore-dev

# Install dependencies
pip install -U -r requirements.txt

# Install in development mode
pip install -e .
```

#### Testing Across Python Versions

```bash
# Unix/Linux/macOS
./test_python_versions_simple.sh

# Windows
test_python_versions_simple.bat
```

### Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Make your changes and add tests
4. Run tests to ensure everything works
5. Submit a pull request

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

Originally developed by Quantopian Inc. Currently maintained by the open-source community.

---

<a name="chinese-version"></a>
## 中文

### 概述

fincore 是一个用于计算常见金融风险和绩效指标的 Python 库，它延续并增强了 empyrical 的功能，由 cloudQuant 持续维护。它为量化金融专业人士和研究人员提供了一个全面的工具包，用于分析投资回报、计算风险指标和进行绩效归因。

### 特性

- **全面的指标**：超过 50 个金融指标，包括收益、风险、风险调整收益和市场关系指标
- **滚动计算**：大多数指标都有滚动窗口版本，用于时间序列分析
- **灵活的输入**：支持 pandas Series/DataFrame 和 numpy 数组
- **NaN 处理**：在所有计算中都能稳健地处理缺失数据
- **绩效归因**：基于因子的绩效分解
- **周期灵活性**：自动周期检测，支持日、周、月、季度和年度数据

### 安装

#### 从源码安装（推荐）

```bash
# 中国用户
git clone https://gitee.com/yunjinqi/fincore

# 国际用户
git clone https://github.com/cloudQuant/fincore

cd fincore

# Windows 系统
install_win.bat

# Linux/macOS 系统
sh install_unix.sh
```

#### 从 PyPI 安装

```bash
pip install fincore
```

### 快速开始

#### 基本指标

```python
import numpy as np
from fincore import empyrical

# 示例收益数据
returns = np.array([0.01, 0.02, 0.03, -0.4, -0.06, -0.02])
benchmark_returns = np.array([0.02, 0.02, 0.03, -0.35, -0.05, -0.01])

# 计算最大回撤
mdd = empyrical.max_drawdown(returns)

print(f"最大回撤: {mdd:.2%}")

# 计算夏普比率（假设为日收益）
sharpe = empyrical.sharpe_ratio(returns, risk_free=0.02/252)

print(f"夏普比率: {sharpe:.2f}")

# 计算 alpha 和 beta
alpha, beta = empyrical.alpha_beta(returns, benchmark_returns)

print(f"Alpha: {alpha:.4f}, Beta: {beta:.2f}")
```

#### 滚动指标

```python
import pandas as pd
from fincore import empyrical

# 创建时间序列数据
dates = pd.date_range('2020-01-01', periods=100, freq='D')
returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)

# 计算 30 天滚动夏普比率
rolling_sharpe = empyrical.roll_sharpe_ratio(returns, window=30)

# 计算 30 天滚动最大回撤
rolling_mdd = empyrical.roll_max_drawdown(returns, window=30)
```

#### DataFrame 高级用法

```python
import pandas as pd
from fincore import empyrical

# 多策略收益
strategies = pd.DataFrame({
    '策略_A': np.random.normal(0.001, 0.02, 252),
    '策略_B': np.random.normal(0.0015, 0.025, 252),
    '策略_C': np.random.normal(0.0008, 0.018, 252)
})

# 一次性计算所有策略的指标
annual_returns = empyrical.annual_return(strategies)
annual_vols = empyrical.annual_volatility(strategies)
calmar_ratios = empyrical.calmar_ratio(strategies)

print("年化收益:")
print(annual_returns)
print("\n年化波动率:")
print(annual_vols)
print("\nCalmar 比率:")
print(calmar_ratios)
```

### 可用指标

#### 收益指标
- `simple_returns()` - 将价格转换为收益率
- `cum_returns()` - 累计收益
- `annual_return()` - 年化平均收益
- `cagr()` - 复合年增长率
- `aggregate_returns()` - 将收益聚合到不同频率

#### 风险指标
- `max_drawdown()` - 最大回撤
- `annual_volatility()` - 年化标准差
- `downside_risk()` - 下行风险
- `value_at_risk()` - 风险价值（VaR）
- `conditional_value_at_risk()` - 条件风险价值（CVaR/预期损失）

#### 风险调整收益指标
- `sharpe_ratio()` - 夏普比率
- `sortino_ratio()` - 索提诺比率
- `calmar_ratio()` - 卡玛比率
- `omega_ratio()` - 欧米茄比率

#### 市场关系指标
- `alpha()`, `beta()` - 詹森阿尔法和贝塔
- `up_capture()`, `down_capture()` - 捕获比率
- `tail_ratio()` - 尾部比率

#### 滚动指标
大多数指标都有以 `roll_` 为前缀的滚动版本：
- `roll_sharpe_ratio()`
- `roll_max_drawdown()`
- `roll_beta()`
- 以及更多...

### 周期常量

```python
from fincore.empyrical import DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY

# 使用周期常量以提高代码清晰度
sharpe_daily = empyrical.sharpe_ratio(returns, period=DAILY)
sharpe_monthly = empyrical.sharpe_ratio(returns, period=MONTHLY)
```

### 测试

```bash
# 运行所有测试
pytest ./tests -n 4

# 运行特定测试模块
pytest ./tests/test_stats.py

# 运行特定测试
pytest ./tests/test_stats.py::test_sharpe_ratio
```

### 开发

#### 设置开发环境

```bash
# 克隆仓库
git clone https://github.com/cloudQuant/fincore
cd fincore

# 创建虚拟环境（使用 conda）
conda create -n fincore-dev python=3.11
conda activate fincore-dev

# 安装依赖
pip install -U -r requirements.txt

# 以开发模式安装
pip install -e .
```

#### 跨 Python 版本测试

```bash
# Unix/Linux/macOS
./test_python_versions_simple.sh

# Windows
test_python_versions_simple.bat
```

### 贡献

我们欢迎贡献！请按以下步骤操作：

1. Fork 仓库
2. 创建功能分支（`git checkout -b feature-name`）
3. 进行更改并添加测试
4. 运行测试确保一切正常
5. 提交 Pull Request

### 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

### 致谢

最初由 Quantopian Inc. 开发，目前由开源社区维护。