# `fincore.report`

Reports use a two-stage contract: compute an immutable `ReportDocument`, then
render it through a selected output module. The renderer never recomputes
financial metrics.

```python
from fincore.report.portfolio.compute import build_portfolio_report
from fincore.report.renderers.html import write_html

document = build_portfolio_report(returns)
artifact = write_html(document, "portfolio-report.html")
```

Available layers:

- `models`: report documents and sections;
- `portfolio.compute`, `risk`, and direct builders: canonical report calculations;
- `renderers.html`, `matplotlib`, `pdf`, `xlsx`, and `interactive`: projections;
- `runtime.artifacts`: caller-owned artifact lifecycle management.
