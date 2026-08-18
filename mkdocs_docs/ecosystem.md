# Ecosystem

fincore is one member of the cloudQuant quantitative-research ecosystem. The
projects below are designed to work together: fincore supplies the metrics and
performance analytics, the backtrader family covers strategy execution,
authoring tooling, and web workflows.

| Project | Focus | Description |
| --- | --- | --- |
| [backtrader](https://github.com/cloudQuant/backtrader) | Strategy execution | Professional Python algorithmic-trading framework for backtesting and live trading, actively maintained by cloudQuant. |
| [backtrader-skills](https://github.com/cloudQuant/backtrader-skills) | Authoring tooling | Offline author/review/test product for the backtrader fork: turns local datasets and typed `StrategySpec v1` into pytest strategies or three-file bundles, reviews candidates without importing them, and runs approved candidates in isolated child processes. |
| [backtrader-mcp](https://github.com/cloudQuant/backtrader-mcp) | LLM integration | Local-first MCP server for building and running reproducible backtrader strategies: immutable datasets, private strategy drafts, and bounded subprocess runs with durable status and reports. Offline and backtest-only. |
| [backtrader_web](https://github.com/cloudQuant/backtrader_web) | Web platform | "AI for Investor": a web-based full-cycle backtrader strategy management platform (Vue 3 + FastAPI) covering research, strategy generation, backtesting analysis, paper trading, live execution, and data management. |
| [backtrader-agent](https://github.com/cloudQuant/backtrader-agent) | Agent runtime | Offline-first strategy-authoring agent runtime: content-addressed data storage, canonical strategy specifications, static review, hash-bound approvals, and a fixed child-process execution profile with recoverable session provenance. |
| [fincore](https://github.com/cloudQuant/fincore) | Analytics core | This repository: unified Python toolkit for financial metrics (150+), performance analysis, backtesting support, AI-driven insights, and multi-database/data source integration. |

## How the pieces fit together

```text
fincore ────────────── metrics, risk, attribution, reports
   ▲
   │ numeric evidence
   │
backtrader ─────────── strategy execution (backtest & live)
   ▲                ▲
   │                │
backtrader-skills ──┴─ author/review/test tooling for strategies
backtrader-mcp ─────── MCP server exposing reproducible strategy workflows to LLM tools
backtrader-agent ───── offline strategy-authoring agent runtime
backtrader_web ─────── web platform connecting research, backtest, paper, and live trading
```

The backtrader family exchanges data through typed contracts (strategy
specifications, immutable datasets, bounded subprocess runs), so each project
stays independently installable and offline-first where applicable.

## Capability states

fincore's public surfaces declare their state in a machine-readable registry
(`fincore.capabilities`) rendered as `docs/quality/capability-inventory.md`.
States are `stable`, `experimental`, `provider_required`, and
`not_implemented`. Consult that inventory before depending on a surface in
 production.
