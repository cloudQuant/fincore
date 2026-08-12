# fincore | Quantitative Performance & Risk Analytics

<p align="center">
    <img src="https://img.shields.io/badge/version-0.3.0-blueviolet.svg" alt="Version 0.3.0"/>
    <img src="https://img.shields.io/badge/compatibility-pinned%20manifests-blue.svg" alt="Pinned compatibility manifests"/>
    <img src="https://img.shields.io/badge/platform-mac%7Clinux%7Cwin-yellow.svg" alt="Platforms"/>
    <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-brightgreen.svg" alt="Python Versions"/>
    <img src="https://img.shields.io/badge/license-Apache%202.0-orange" alt="License: Apache 2.0"/>
</p>

<p align="center">
    <a href="#english">English</a> · <a href="#中文">中文</a> · <a href="https://fincore.readthedocs.io/en/latest/">Documentation</a> · <a href="CONTRIBUTING.md">Contributing</a> · <a href="CHANGELOG.md">Changelog</a> · <a href="docs/MIGRATION.md">Migration Guide</a>
</p>

---

<a name="english"></a>

## Overview

**fincore** is a Python library for quantitative finance analytics — 150+ financial metrics, portfolio optimization, Monte Carlo simulation, and performance attribution. It continues the **empyrical** stack under active maintenance by [cloudQuant](https://github.com/cloudQuant).

### Highlights

| Feature | Description |
|---------|-------------|
| **150+ Metrics** | Returns, risk, drawdown, alpha/beta, capture ratios, timing, streaks |
| **AnalysisContext** | `fincore.analyze()` — lazy, cached computation with JSON/HTML export |
| **RollingEngine** | Batch rolling metrics (sharpe, volatility, max_drawdown, beta) in one call |
| **Pluggable Viz** | Matplotlib, HTML, Plotly, Bokeh backends via `VizBackend` protocol |
| **Portfolio Optimization** | Efficient frontier, risk parity, constrained optimization |
| **Monte Carlo** | Bootstrap, scenario testing, path simulation |
| **Performance Attribution** | Brinson, Fama-French, style analysis |
| **Lazy Imports** | Heavy dependencies are loaded on first use |
| **PEP 561** | `py.typed` marker for type checker support |

### Installation

```bash
pip install fincore                  # Core
pip install "fincore[viz]"           # + matplotlib, seaborn
pip install "fincore[all]"           # Everything
pip install "fincore[dev]"           # Development tools
```

**From source:**
```bash
git clone https://github.com/cloudQuant/fincore   # International
git clone https://gitee.com/yunjinqi/fincore       # China mirror
cd fincore && pip install -e ".[dev,viz]"
```

### Quick Start

```python
import fincore
import pandas as pd

# A self-contained return series
returns = pd.Series([0.01, -0.005, 0.002, 0.004])

print(f"Sharpe: {fincore.sharpe_ratio(returns):.4f}")
print(f"Max DD: {fincore.max_drawdown(returns):.4f}")
```

**Compatibility status:** empyrical 0.6.0 and the bounded pyfolio 0.9.6
profile are pinned as targets, but fincore 0.3.0 is not yet certified as a
drop-in replacement. See the [empyrical matrix](docs/compatibility/empyrical-0.6.0.md),
[pyfolio profile](docs/compatibility/pyfolio-0.9.6.md), and
[migration guide](docs/MIGRATION.md).

**RollingEngine** (batch rolling metrics):
```python
from fincore.core.engine import RollingEngine

engine = RollingEngine(returns, factor_returns=benchmark, window=60)
results = engine.compute(['sharpe', 'volatility', 'max_drawdown', 'beta'])
```

**Portfolio Optimization:**
```python
from fincore.optimization import efficient_frontier, risk_parity, optimize

ef = efficient_frontier(returns_df, n_points=50)
rp = risk_parity(returns_df)
w = optimize(returns_df, objective="max_sharpe")
```

### Architecture

```
fincore/
├── __init__.py          # Lazy exports (Empyrical, Pyfolio, analyze)
├── empyrical.py         # Empyrical facade (150+ methods)
├── core/
│   ├── context.py       # AnalysisContext — lazy cached metrics
│   └── engine.py        # RollingEngine — batch rolling metrics
├── metrics/             # 17 metric modules (returns, risk, ratios, ...)
├── viz/                 # VizBackend protocol + backends
├── optimization/        # Efficient frontier, risk parity
├── simulation/          # Monte Carlo, bootstrap
├── attribution/         # Brinson, Fama-French, style analysis
├── risk/                # EVT, GARCH
├── report/              # HTML/PDF report generation
├── tearsheets/          # Pyfolio-style plotting
└── utils/               # Shared helpers
```

### Testing

```bash
pytest tests/ -n 4                   # Parallel
pytest tests/ --cov=fincore          # With coverage
pytest tests/test_core/              # AnalysisContext, RollingEngine, Viz
```

### License

The fincore repository declares Apache License 2.0; see [LICENSE](LICENSE).
Adapted-source provenance and unresolved upstream notice questions are tracked
in [docs/upstream-provenance.md](docs/upstream-provenance.md).

---

<a name="中文"></a>

## 概述

**fincore** 是面向量化金融的 Python 分析库 — 150+ 金融指标、组合优化、蒙特卡洛模拟和绩效归因。它延续 **empyrical** 分析栈，由 [cloudQuant](https://github.com/cloudQuant) 持续维护。

### 核心特性

| 特性 | 说明 |
|------|------|
| **150+ 指标** | 收益、风险、回撤、Alpha/Beta、捕获比率、择时、连续统计 |
| **AnalysisContext** | `fincore.analyze()` — 惰性计算、自动缓存，支持 JSON/HTML 导出 |
| **RollingEngine** | 批量滚动指标（sharpe、volatility、max_drawdown、beta）一次调用 |
| **可插拔可视化** | Matplotlib、HTML、Plotly、Bokeh 后端，基于 `VizBackend` 协议 |
| **组合优化** | 有效前沿、风险平价、约束优化 |
| **蒙特卡洛** | Bootstrap、情景测试、路径模拟 |
| **绩效归因** | Brinson、Fama-French、风格分析 |
| **惰性导入** | 重型依赖在首次使用时加载 |
| **PEP 561** | `py.typed` 标记，支持类型检查器 |

### 安装

```bash
pip install fincore                  # 核心
pip install "fincore[viz]"           # + matplotlib, seaborn
pip install "fincore[all]"           # 全部依赖
pip install "fincore[dev]"           # 开发工具
```

**从源码安装：**
```bash
git clone https://gitee.com/yunjinqi/fincore       # 中国用户
git clone https://github.com/cloudQuant/fincore     # 国际用户
cd fincore && pip install -e ".[dev,viz]"
```

### 快速开始

```python
import fincore
import pandas as pd

# 自包含的收益率序列
returns = pd.Series([0.01, -0.005, 0.002, 0.004])

print(f"夏普比率: {fincore.sharpe_ratio(returns):.4f}")
print(f"最大回撤: {fincore.max_drawdown(returns):.4f}")
```

**兼容状态：** empyrical 0.6.0 与限定的 pyfolio 0.9.6 profile 已冻结为
目标，但 fincore 0.3.0 尚未被认证为可直接替换。请参阅
[empyrical 矩阵](docs/compatibility/empyrical-0.6.0.md)、
[pyfolio profile](docs/compatibility/pyfolio-0.9.6.md) 与
[迁移指南](docs/MIGRATION.md)。

**RollingEngine**（批量滚动指标）：
```python
from fincore.core.engine import RollingEngine

engine = RollingEngine(returns, factor_returns=benchmark, window=60)
results = engine.compute(['sharpe', 'volatility', 'max_drawdown', 'beta'])
```

**组合优化：**
```python
from fincore.optimization import efficient_frontier, risk_parity, optimize

ef = efficient_frontier(returns_df, n_points=50)
rp = risk_parity(returns_df)
w = optimize(returns_df, objective="max_sharpe")
```

### 指标概览

| 类别 | 主要指标 |
|------|---------|
| **收益** | `simple_returns`, `cum_returns`, `annual_return`, `cagr`, `aggregate_returns` |
| **风险** | `max_drawdown`, `annual_volatility`, `downside_risk`, `value_at_risk`, `conditional_value_at_risk` |
| **风险调整** | `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `omega_ratio`, `information_ratio` |
| **市场关系** | `alpha`, `beta`, `up_capture`, `down_capture`, `treynor_ratio` |
| **滚动** | `roll_sharpe_ratio`, `roll_max_drawdown`, `roll_beta`, `roll_alpha` 等 |
| **择时** | `treynor_mazuy_timing`, `henriksson_merton_timing`, `market_timing_return` |
| **连续统计** | `max_consecutive_up_days`, `max_consecutive_down_days`, `max_single_day_gain` |

> 完整指标列表见 [API 文档](docs/api文档/README.md)

### 测试

```bash
pytest tests/ -n 4                   # 并行运行
pytest tests/ --cov=fincore          # 含覆盖率
pytest tests/test_core/              # AnalysisContext、RollingEngine、可视化
```

### 贡献

欢迎贡献！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

### 许可证

fincore 仓库声明采用 Apache License 2.0，详见 [LICENSE](LICENSE)。
改编来源与尚待人工确认的上游 notice 问题记录于
[docs/upstream-provenance.md](docs/upstream-provenance.md)。
