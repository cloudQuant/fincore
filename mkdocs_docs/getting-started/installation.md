# Installation

## From PyPI (Recommended)

```bash
pip install fincore
```

## From Source

```bash
# China users
git clone https://gitee.com/yunjinqi/fincore

# International users
git clone https://github.com/cloudQuant/fincore

cd fincore
pip install -U .
```

## Optional Extras

```bash
# Pyfolio tear sheets (matplotlib, seaborn, ipython)
pip install "fincore[pyfolio]"

# Compute-only enhanced factor analysis
pip install "fincore[factor-analysis]"

# Strict Alphalens migration APIs and factor-analysis plotting
pip install "fincore[alphalens]"

# Interactive backends (plotly, bokeh)
pip install "fincore[interactive]"

# PDF report rendering
pip install "fincore[report-pdf]"

# XLSX report export
pip install "fincore[report-xlsx]"

# Bayesian analysis (pymc)
pip install "fincore[bayesian]"

# Data providers
pip install "fincore[data-yahoo]"
pip install "fincore[data-alphavantage]"
pip install "fincore[data-pandas-datareader]"
pip install "fincore[data-cn]"

# Everything
pip install "fincore[all]"

# Development (pytest, ruff, mypy, etc.)
pip install "fincore[dev]"
```

`viz` and `datareader` are 0.3.x compatibility aliases.

`fincore[alphalens]` does not install a top-level `alphalens` package. Import
the strict migration façade as `fincore.alphalens`, or use the enhanced
`fincore.factor_analysis` workflow. The former source snapshot is identified
by its pinned commit, not by its conflicting historical version strings.

## Requirements

Core dependencies (`pyproject.toml` is the single source of truth):

- Python >= 3.11 (a documented breaking change relative to empyrical)
- numpy >= 1.24.0
- pandas >= 1.5.0
- scipy >= 1.3.0
- pytz >= 2023.3
- packaging >= 21.0
