# Installation

## Core package

```bash
pip install fincore
```

fincore requires Python **3.11+**. The core package includes numerical metrics,
portfolio and report models, risk, attribution, optimisation, simulation, and
the capability runtime. Install optional dependencies only for capabilities you
intend to use.

## Capability extras

```bash
pip install "fincore[factor-analysis]"       # statsmodels-backed factor inference
pip install "fincore[visualization]"         # matplotlib, seaborn, Plotly, Bokeh
pip install "fincore[interactive]"           # Plotly and Bokeh only
pip install "fincore[report-pdf]"            # PDF report renderer
pip install "fincore[report-xlsx]"           # XLSX report renderer
pip install "fincore[bayesian]"              # Bayesian analysis
pip install "fincore[data-yahoo]"            # Yahoo Finance provider
pip install "fincore[data-alphavantage]"     # Alpha Vantage provider
pip install "fincore[data-pandas-datareader]"# pandas-datareader provider
pip install "fincore[data-cn]"               # Chinese data providers
pip install "fincore[all]"                   # full functional union
```

The extras above are the complete public set. There are no compatibility alias
extras or package-profile extras in 0.5.

## From source

```bash
git clone https://github.com/cloudQuant/fincore
cd fincore
pip install -e ".[dev]"
```

`pyproject.toml` is the single source of truth for supported dependencies and
optional capability groups.
