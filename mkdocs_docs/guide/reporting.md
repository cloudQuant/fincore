# Portfolio reporting

Build a report model once and render it explicitly. This makes report contents
testable, reusable across output formats, and independent of any UI façade.

```python
from fincore.report.portfolio.compute import build_portfolio_report
from fincore.report.renderers.html import write_html

document = build_portfolio_report(
    returns,
    benchmark_returns=benchmark_returns,
    positions=positions,
    transactions=transactions,
)
artifact = write_html(document, "portfolio-report.html")
```

`ReportDocument` contains named sections, metrics, tables, series, units, and
semantic digests. Choose `html`, `matplotlib`, `pdf`, `xlsx`, or interactive
renderers as a separate final step. PDF and XLSX renderers require their
respective extras.
