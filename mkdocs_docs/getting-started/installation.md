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

## Optional Dependencies

```bash
# Visualization (matplotlib, seaborn)
pip install "fincore[viz]"

# Bayesian analysis (pymc)
pip install "fincore[bayesian]"

# Everything
pip install "fincore[all]"

# Development (pytest, ruff, mypy, etc.)
pip install "fincore[dev]"
```

## Requirements

- Python >= 3.11
- numpy >= 1.17.0
- pandas >= 0.25.0
- scipy >= 1.3.0
