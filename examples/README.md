# Canonical 0.5 examples

These examples demonstrate direct domain APIs only. They do not use root-level
metric exports, façade classes, or upstream-shaped compatibility imports.

| Example | Capability |
| --- | --- |
| `metrics_report.py` | direct metrics plus the portfolio-report document workflow |
| `portfolio_optimization.py` | efficient frontier, risk parity, and objective optimisation |
| `risk_validation.py` | walk-forward VaR and an immutable validation report |
| `factor_analysis_quickstart.py` | deterministic factor preparation, analysis, and optional rendering |

Run an example from the repository root with the installed 0.5 checkout. The
factor example needs `fincore[visualization]`; the other examples use core
dependencies only.
