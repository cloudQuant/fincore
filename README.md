# fincore | Quantitative Performance & Risk Analytics

<p align="center">
    <img src="https://img.shields.io/badge/version-0.3.0-blueviolet.svg" alt="Version 0.3.0"/>
    <img src="https://img.shields.io/badge/status-Beta-orange.svg" alt="Status: Beta"/>
    <img src="https://img.shields.io/badge/platform-mac%7Clinux%7Cwin-yellow.svg" alt="Platforms"/>
    <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-brightgreen.svg" alt="Python Versions"/>
    <img src="https://img.shields.io/badge/license-Apache%202.0-orange" alt="License: Apache 2.0"/>
</p>

<p align="center">
    <a href="#english">English</a> · <a href="#中文">中文</a> · <a href="https://cloudquant.github.io/fincore/">Documentation</a> · <a href="CONTRIBUTING.md">Contributing</a> · <a href="CHANGELOG.md">Changelog</a> · <a href="docs/MIGRATION.md">Migration Guide</a>
</p>

---

<a name="english"></a>

## Overview

**fincore** is a Python library for quantitative finance analytics — 150+ financial metrics, portfolio optimization, Monte Carlo simulation, and performance attribution. It continues the **empyrical** stack under active maintenance by [cloudQuant](https://github.com/cloudQuant).

Current version: **0.3.0** (Beta). Python **3.11+** is required; this is a documented breaking change relative to empyrical, which supports older interpreters.

### Three API surfaces

fincore 0.3.0 exposes clearly separated surfaces. One name does not silently switch semantics between them:

| Surface | What it is | Guarantee |
|---------|------------|-----------|
| **Strict compatibility** — `fincore.empyrical` | Frozen empyrical 0.6.0 surface: 54/54 public symbols (C0), 49/49 callables (C1), core callables numerically verified (C3) | Pinned by `tests/compat/fixtures/` manifests and enforced by the `tests/compat/` gates |
| **pyfolio façade** — `fincore.pyfolio` | Frozen pyfolio 0.9.6 profile of 11 tear-sheet workflows: all entries C1, risk/returns/perf-attrib/full-sheet main chains C4 | Requires the `fincore[pyfolio]` extra |
| **Enhanced semantics** — `fincore.metrics`, flat API, `AnalysisContext` | fincore's own, documented divergences (e.g. `week_year="iso"`, explicit validation exceptions) | Recommended API; enhanced, not empyrical-identical |
| **Alphalens migration** — `fincore.alphalens` / `fincore.factor_analysis` | A source-shaped strict façade and a separate enhanced prepare/analyze/render workflow | Beta integration; use the tested APIs documented in the [migration guide](docs/MIGRATION.md), not a top-level `alphalens` import |

See the [compatibility matrix](https://cloudquant.github.io/fincore/development/compatibility/), [empyrical matrix](docs/compatibility/empyrical-0.6.0.md), and [pyfolio profile](docs/compatibility/pyfolio-0.9.6.md).

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
pip install fincore                       # Core metrics
pip install "fincore[pyfolio]"            # + Pyfolio tear sheets (matplotlib, seaborn, ipython)
pip install "fincore[factor-analysis]"    # + Compute-only enhanced factor analysis
pip install "fincore[alphalens]"          # + Factor-analysis rendering and strict Alphalens migration APIs
pip install "fincore[interactive]"        # + Plotly, Bokeh backends
pip install "fincore[report-pdf]"         # + Playwright PDF rendering
pip install "fincore[report-xlsx]"        # + XLSX report export
pip install "fincore[bayesian]"           # + Bayesian tear sheets (pymc)
pip install "fincore[data-yahoo]"         # + Yahoo Finance provider
pip install "fincore[data-pandas-datareader]"  # + pandas-datareader provider
pip install "fincore[data-alphavantage]"  # + Alpha Vantage provider
pip install "fincore[data-cn]"            # + Tushare, AkShare providers
pip install "fincore[all]"                # Everything above
pip install "fincore[dev]"                # Development tools
```

`datareader` and `viz` are 0.3.x compatibility aliases for the functional extras above.

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

The strict empyrical module is available directly:

```python
from fincore import empyrical

print(empyrical.sharpe_ratio(returns))
print(empyrical.max_drawdown(returns))
```

**AnalysisContext** (recommended stateful API — lazy, cached, exportable):

```python
index = pd.date_range("2024-01-02", periods=5, freq="B")
returns = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001], index=index)
benchmark = pd.Series([0.008, -0.003, 0.001, 0.002, 0.0], index=index)

ctx = fincore.analyze(returns, factor_returns=benchmark)
print(ctx.sharpe_ratio, ctx.max_drawdown)
ctx.to_json(path="report.json")      # write files
ctx.to_html(path="report.html")
ctx.plot(backend="matplotlib")       # -> ReportArtifacts
```

**Pyfolio main chain** (`from fincore import Pyfolio` requires the `pyfolio` extra):

```python
from fincore import Pyfolio

pyfolio = Pyfolio(returns=returns, benchmark_rets=benchmark)
pyfolio.create_returns_tear_sheet(returns, benchmark_rets=benchmark)
```

**RollingEngine** (batch rolling metrics):
```python
import numpy as np

from fincore.core.engine import RollingEngine

rng = np.random.default_rng(7)
index = pd.date_range("2024-01-02", periods=60, freq="B")
returns = pd.Series(rng.normal(0.001, 0.02, 60), index=index)
benchmark = pd.Series(rng.normal(0.0005, 0.015, 60), index=index)

engine = RollingEngine(returns, factor_returns=benchmark, window=30)
results = engine.compute(['sharpe', 'volatility', 'max_drawdown', 'beta'])
```

**Portfolio Optimization:**
```python
from fincore.optimization import efficient_frontier, risk_parity, optimize

returns_df = pd.DataFrame(
    {
        "asset_a": [0.01, -0.005, 0.004, 0.002],
        "asset_b": [0.003, 0.002, -0.001, 0.005],
    }
)
ef = efficient_frontier(returns_df, n_points=5)
rp = risk_parity(returns_df)
w = optimize(returns_df, objective="max_sharpe")
```

Every Python code block above is executed with the same data and arguments
by a matching test in [`tests/docs/test_examples.py`](tests/docs/test_examples.py).

### Alphalens migration quickstart

The repository includes an executable, deterministic migration example. It
builds local synthetic data with a fixed seed, uses no network, writes no
output files, renders only with Agg, and closes its returned figures:

```bash
pip install "fincore[alphalens]"
MPLBACKEND=Agg python examples/factor_analysis_quickstart.py
```

Use `fincore.alphalens` for source-shaped strict calls and
`fincore.factor_analysis` for new prepare/analyze/render workflows. Do not use
`import alphalens`: fincore intentionally does not install or expose a
top-level standalone-compatible package. See [the migration guide](docs/MIGRATION.md)
for the API map and limitations.

### Architecture

```
fincore/
├── __init__.py          # Lazy exports (Empyrical, Pyfolio, analyze)
├── empyrical.py         # Strict empyrical 0.6.0 compatibility facade
├── pyfolio.py           # Pyfolio 0.9.6-profile workflow facade
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

### Quality

Quality numbers are machine-generated, never hand-written into this README.
The current snapshot is [`docs/quality/current-baseline.md`](docs/quality/current-baseline.md)
(JSON: [`docs/quality/current-baseline.json`](docs/quality/current-baseline.json)),
regenerated by `scripts/collect_quality_baseline.py` on each release-gate run.
The release-candidate checklist is [`docs/quality/release-candidate-checklist.md`](docs/quality/release-candidate-checklist.md).

### Testing

```bash
pytest tests/                         # Default selector, parallel via xdist
pytest -o addopts='' tests/compat -q  # Empyrical/pyfolio compatibility gates
pytest -o addopts='' tests/docs -q    # Executable documentation examples
pytest tests/ --cov=fincore           # With coverage
```

### License

The fincore repository declares Apache License 2.0; see [LICENSE](LICENSE).
Adapted-source provenance and unresolved upstream notice questions are tracked
in [docs/upstream-provenance.md](docs/upstream-provenance.md). The required
human Alphalens license/NOTICE decision remains a release blocker; no
third-party notice or legal conclusion is implied by this integration.

---

<a name="中文"></a>

## 概述

**fincore** 是面向量化金融的 Python 分析库 — 150+ 金融指标、组合优化、蒙特卡洛模拟和绩效归因。它延续 **empyrical** 分析栈，由 [cloudQuant](https://github.com/cloudQuant) 持续维护。

当前版本 **0.3.0**（Beta）。要求 **Python 3.11+**；这是相对 empyrical（支持更老解释器）的明确 breaking change。

### 三层 API

fincore 0.3.0 暴露三个严格分离的界面，同名函数不会在界面之间静默切换语义：

| 界面 | 内容 | 保证 |
|------|------|------|
| **严格兼容** — `fincore.empyrical` | 冻结的 empyrical 0.6.0 表面：54/54 公共符号（C0）、49/49 callable（C1）、核心 callable 数值级验证（C3） | 由 `tests/compat/fixtures/` 清单固定，`tests/compat/` 门禁强制执行 |
| **pyfolio 门面** — `fincore.pyfolio` | 冻结的 pyfolio 0.9.6 profile（11 个 tear-sheet 工作流）：全部 C1，risk/returns/perf-attrib/full-sheet 主链 C4 | 需要 `fincore[pyfolio]` extra |
| **增强语义** — `fincore.metrics`、flat API、`AnalysisContext` | fincore 自有、已文档化的分歧（如 `week_year="iso"`、显式校验异常） | 推荐 API；是增强语义，不承诺与 empyrical 完全一致 |

参见[兼容矩阵](https://cloudquant.github.io/fincore/development/compatibility/)、[empyrical 矩阵](docs/compatibility/empyrical-0.6.0.md)与 [pyfolio profile](docs/compatibility/pyfolio-0.9.6.md)。

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
pip install fincore                       # 核心指标
pip install "fincore[pyfolio]"            # + Pyfolio tear sheets（matplotlib、seaborn、ipython）
pip install "fincore[interactive]"        # + Plotly、Bokeh 后端
pip install "fincore[report-pdf]"         # + Playwright PDF 渲染
pip install "fincore[report-xlsx]"        # + XLSX 报告导出
pip install "fincore[bayesian]"           # + Bayesian tear sheets（pymc）
pip install "fincore[data-yahoo]"         # + Yahoo Finance 数据源
pip install "fincore[data-pandas-datareader]"  # + pandas-datareader 数据源
pip install "fincore[data-alphavantage]"  # + Alpha Vantage 数据源
pip install "fincore[data-cn]"            # + Tushare、AkShare 数据源
pip install "fincore[all]"                # 以上全部
pip install "fincore[dev]"                # 开发工具
```

`datareader` 与 `viz` 是 0.3.x 的兼容别名，指向上述功能性 extras。

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

严格兼容的 empyrical 模块可直接导入：

```python
from fincore import empyrical

print(empyrical.sharpe_ratio(returns))
print(empyrical.max_drawdown(returns))
```

**AnalysisContext**（推荐的有状态 API — 惰性、缓存、可导出）：

```python
index = pd.date_range("2024-01-02", periods=5, freq="B")
returns = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001], index=index)
benchmark = pd.Series([0.008, -0.003, 0.001, 0.002, 0.0], index=index)

ctx = fincore.analyze(returns, factor_returns=benchmark)
print(ctx.sharpe_ratio, ctx.max_drawdown)
ctx.to_json(path="report.json")      # 写入文件
ctx.to_html(path="report.html")
ctx.plot(backend="matplotlib")       # -> ReportArtifacts
```

**Pyfolio 主链**（`from fincore import Pyfolio` 需要 `pyfolio` extra）：

```python
from fincore import Pyfolio

pyfolio = Pyfolio(returns=returns, benchmark_rets=benchmark)
pyfolio.create_returns_tear_sheet(returns, benchmark_rets=benchmark)
```

**RollingEngine**（批量滚动指标）：
```python
import numpy as np

from fincore.core.engine import RollingEngine

rng = np.random.default_rng(7)
index = pd.date_range("2024-01-02", periods=60, freq="B")
returns = pd.Series(rng.normal(0.001, 0.02, 60), index=index)
benchmark = pd.Series(rng.normal(0.0005, 0.015, 60), index=index)

engine = RollingEngine(returns, factor_returns=benchmark, window=30)
results = engine.compute(['sharpe', 'volatility', 'max_drawdown', 'beta'])
```

**组合优化：**
```python
from fincore.optimization import efficient_frontier, risk_parity, optimize

returns_df = pd.DataFrame(
    {
        "asset_a": [0.01, -0.005, 0.004, 0.002],
        "asset_b": [0.003, 0.002, -0.001, 0.005],
    }
)
ef = efficient_frontier(returns_df, n_points=5)
rp = risk_parity(returns_df)
w = optimize(returns_df, objective="max_sharpe")
```

以上每个 Python 代码块都以相同的数据和参数被
[`tests/docs/test_examples.py`](tests/docs/test_examples.py) 中对应的测试执行。

### 质量

质量数字由机器生成，绝不手写进本 README。当前快照见
[`docs/quality/current-baseline.md`](docs/quality/current-baseline.md)
（JSON: [`docs/quality/current-baseline.json`](docs/quality/current-baseline.json)），
由 `scripts/collect_quality_baseline.py` 在每次发布门禁运行时重新生成。
发布候选清单见 [`docs/quality/release-candidate-checklist.md`](docs/quality/release-candidate-checklist.md)。

### 测试

```bash
pytest tests/                         # 默认选择器，xdist 并行
pytest -o addopts='' tests/compat -q  # empyrical/pyfolio 兼容门禁
pytest -o addopts='' tests/docs -q    # 可执行文档示例
pytest tests/ --cov=fincore           # 含覆盖率
```

### 贡献

欢迎贡献！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

### 许可证

fincore 仓库声明采用 Apache License 2.0，详见 [LICENSE](LICENSE)。
改编来源与尚待人工确认的上游 notice 问题记录于
[docs/upstream-provenance.md](docs/upstream-provenance.md)。
