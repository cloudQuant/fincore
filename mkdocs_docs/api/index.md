# API Reference

## Package Structure

```
fincore/
├── __init__.py          # Flat API: sharpe_ratio, max_drawdown, ...
├── empyrical.py         # Empyrical facade class (150+ methods)
├── pyfolio.py           # Pyfolio tearsheet class
├── core/
│   ├── context.py       # AnalysisContext
│   └── engine.py        # RollingEngine
├── metrics/             # 17 metric modules
├── viz/                 # Visualization backends
├── optimization/        # Portfolio optimization
├── simulation/          # Monte Carlo, bootstrap
├── attribution/         # Brinson, Fama-French
├── risk/                # EVT, GARCH
└── report/              # HTML/PDF reports
```

## Module Pages

- [fincore (top-level)](fincore.md) — Flat API and main classes
- [fincore.metrics](metrics.md) — All metric functions
- [fincore.core](core.md) — AnalysisContext and RollingEngine
- [fincore.viz](viz.md) — Visualization backends
- [fincore.optimization](optimization.md) — Portfolio optimization
- [fincore.simulation](simulation.md) — Monte Carlo simulation
- [fincore.attribution](attribution.md) — Performance attribution
- [fincore.risk](risk.md) — Risk models
- [fincore.report](report.md) — Report generation
