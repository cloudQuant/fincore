# fincore 文档索引 / Documentation Index

## English

| Document | Description |
|----------|-------------|
| [API Guide](api.md) | Stable entry points and usage patterns |
| [API Stability](API_STABILITY.md) | Versioning guarantees and deprecation policy |
| [Migration Guide](MIGRATION.md) | Migrating from empyrical to fincore |
| [Development Guide](development.md) | Environment setup, testing, linting, type checking |
| [Examples](examples.md) | Three usage patterns (Flat API, Empyrical, AnalysisContext) |
| [User Guide](user_guide.md) | Core data model, quickstart, reports |
| [Changelog](../CHANGELOG.md) | Release history |
| [Contributing](../CONTRIBUTING.md) | How to contribute |

### API Reference (API 文档)

| Document | Description |
|----------|-------------|
| [Top-Level API](api文档/01_顶层API.md) | `fincore.*` package-level functions and classes |
| [Empyrical Metrics](api文档/02_Empyrical指标.md) | 100+ performance/risk metric functions |
| [Pyfolio Visualization](api文档/03_Pyfolio可视化.md) | Tear sheets and plotting functions |
| [Risk Models](api文档/04_风险模型.md) | EVT, GARCH conditional volatility |
| [Portfolio Optimization](api文档/05_组合优化.md) | Efficient frontier, risk parity |
| [Monte Carlo Simulation](api文档/06_蒙特卡洛模拟.md) | Path simulation, bootstrap inference |
| [Performance Attribution](api文档/07_绩效归因.md) | Brinson, style analysis, factor regression |
| [Report Generation](api文档/08_报告生成.md) | HTML/PDF strategy reports |

---

## 中文

### 用户手册

| 文档 | 说明 |
|------|------|
| [安装指南](用户手册/01_安装指南.md) | 安装方式、依赖管理、环境配置 |
| [快速入门](用户手册/02_快速入门.md) | 5 分钟上手 fincore 核心功能 |
| [核心概念](用户手册/03_核心概念.md) | 数据模型、API 架构、延迟加载 |
| [数据准备](用户手册/04_数据准备.md) | 收益率序列、持仓、交易数据格式 |
| [常见工作流](用户手册/05_常见工作流.md) | 策略分析、报告生成、风险管理 |
| [FAQ](用户手册/06_FAQ.md) | 常见问题与解答 |

### 教程

| 文档 | 说明 |
|------|------|
| [入门：策略绩效分析](教程/01_入门_策略绩效分析.md) | 从零开始分析一个策略 |
| [进阶：风险管理与压力测试](教程/02_进阶_风险管理与压力测试.md) | EVT、GARCH、蒙特卡洛 |
| [进阶：组合优化](教程/03_进阶_组合优化.md) | 有效前沿、风险平价、约束优化 |
| [高级：绩效归因与因子分析](教程/04_高级_绩效归因与因子分析.md) | Brinson、风格分析、多因子 |

---

## Project Stats

- **Source**: 85 Python files, ~26,700 lines
- **Tests**: 1800 tests, 243 files, ~27,000 lines
- **Docs**: 93 markdown files, ~16,300 lines
- **Examples**: 27 Python scripts
- **Docstring coverage**: 92%
- **Python**: 3.11 / 3.12 / 3.13
