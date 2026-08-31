# Visualization

Visualisation is explicit and backend-specific. Use `fincore.viz.get_backend`
when plotting a domain result, or use report renderers for document output.

```python
from fincore.viz import get_backend

backend = get_backend("matplotlib")
figure = backend.plot_returns(cumulative_returns)
```

Install `fincore[visualization]` for matplotlib, seaborn, Plotly, and Bokeh;
install `fincore[interactive]` when only Plotly/Bokeh is required. Backends
receive already-computed data and do not own financial calculation semantics.
