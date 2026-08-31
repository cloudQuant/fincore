# fincore Examples

The maintained executable examples use direct 0.5 domain APIs:

| Example | Capability |
| --- | --- |
| [`examples/metrics_report.py`](../examples/metrics_report.py) | returns, ratios, drawdown, annual return |
| [`examples/portfolio_optimization.py`](../examples/portfolio_optimization.py) | efficient frontier and risk parity |
| [`examples/risk_validation.py`](../examples/risk_validation.py) | walk-forward risk validation and report model |
| [`examples/factor_analysis_quickstart.py`](../examples/factor_analysis_quickstart.py) | factor preparation, analysis, portfolio inputs, rendering |

Run an example from a source checkout after installing its declared optional
extra. For example:

```bash
pip install -e ".[visualization]"
MPLBACKEND=Agg python examples/factor_analysis_quickstart.py
```

Examples import leaf modules rather than package-root aliases. A report example
builds a `ReportDocument` first and then selects a renderer; it does not use a
stateful façade or a compatibility tear-sheet workflow.
