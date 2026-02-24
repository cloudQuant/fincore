# Visualization

fincore provides pluggable visualization backends via the `VizBackend` protocol.

## Built-in Backends

### Matplotlib
```python
from fincore.viz import get_backend
viz = get_backend("matplotlib")
viz.plot_returns(cum_returns)
viz.plot_drawdown(drawdown)
viz.plot_rolling_sharpe(rolling_sharpe)
viz.plot_monthly_heatmap(returns)
```

### HTML (no extra dependencies)
```python
from fincore.viz.html_backend import HtmlReportBuilder
builder = HtmlReportBuilder()
builder.add_title("My Report")
builder.add_stats_table(stats)
builder.save("report.html")
```

### Via AnalysisContext
```python
ctx = fincore.analyze(returns, factor_returns=benchmark)
ctx.plot(backend="matplotlib")
ctx.to_html(path="report.html")
```

## API Reference

::: fincore.viz.base.VizBackend

::: fincore.viz.html_backend.HtmlReportBuilder
