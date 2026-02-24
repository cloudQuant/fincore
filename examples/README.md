# fincore Examples

This directory contains practical examples demonstrating how to use fincore for quantitative finance analysis.

## Core Examples

### 1. Single Strategy Analysis
**File**: `01_single_strategy_analysis.py`

Demonstrates comprehensive analysis of a single trading strategy:
- Individual metric calculation
- AnalysisContext usage
- Performance statistics
- Drawdown analysis
- HTML report generation

```bash
python examples/01_single_strategy_analysis.py
```

### 2. Multi-Strategy Comparison
**File**: `02_multi_strategy_comparison.py`

Shows how to compare multiple trading strategies:
- Risk-adjusted performance metrics
- Sharpe/Sortino ratio comparison
- Rolling metrics comparison
- Correlation analysis

### 3. Factor Attribution Analysis
**File**: `03_factor_attribution.py`

Performs factor-based performance attribution:
- Alpha and beta calculation
- Rolling alpha/beta
- Capture ratios
- Performance decomposition

### 4. Risk Management Report
**File**: `04_risk_management_report.py`

Generates a comprehensive risk management report:
- Performance summary
- Drawdown analysis
- Value at Risk (VaR) and CVaR
- Distribution statistics

### 5. RollingEngine Batch Metrics
**File**: `05_rolling_engine.py`

Demonstrates high-performance batch rolling metrics calculation:
- Compute multiple rolling metrics in a single call
- Performance comparison with individual calculations
- Statistical summary and visualization

```bash
python examples/05_rolling_engine.py
```

## Advanced Examples

### 6. Advanced Backtesting Metrics
**File**: `06_advanced_backtesting_metrics.py`

Comprehensive backtesting performance analysis:
- Detailed drawdown analysis (max, average, periods)
- Streak statistics (consecutive wins/losses)
- Yearly and monthly performance breakdown
- Distribution statistics and tail risk analysis

```bash
python examples/06_advanced_backtesting_metrics.py
```

### 7. Risk Models Analysis
**File**: `07_market_timing_analysis.py`

Advanced risk modeling and analysis:
- Value at Risk (VaR) at multiple confidence levels
- Conditional VaR (CVaR/Expected Shortfall)
- Extreme Value Theory (EVT) analysis
- GARCH volatility modeling
- Stress testing scenarios

```bash
python examples/07_market_timing_analysis.py
```

### 8. Portfolio Optimization
**File**: `08_portfolio_optimization_deep_dive.py`

Portfolio optimization techniques:
- Efficient frontier calculation
- Risk parity portfolio construction
- Maximum Sharpe ratio optimization
- Strategy comparison and visualization

```bash
python examples/08_portfolio_optimization_deep_dive.py
```

### 9. Monte Carlo Simulation
**File**: `09_monte_carlo_simulation.py`

Monte Carlo simulation for risk analysis:
- Path generation using Geometric Brownian Motion
- Risk metrics (VaR, CVaR) from simulation
- Stress testing scenarios
- Probability analysis for target returns

```bash
python examples/09_monte_carlo_simulation.py
```

### 10. Market Timing Analysis
**File**: `10_market_timing_analysis.py`

Market timing ability evaluation:
- Treynor-Mazuy timing model
- Henriksson-Merton timing model
- Up/Down capture ratios
- Bull/Bear market analysis

```bash
python examples/10_market_timing_analysis.py
```

### 11. Performance Attribution
**File**: `11_performance_attribution.py`

Performance attribution and decomposition:
- Brinson attribution model
- Factor exposure analysis
- Risk attribution
- Win/Loss analysis

```bash
python examples/11_performance_attribution.py
```

### 12. Data Provider Usage
**File**: `12_data_provider_usage.py`

Data fetching from multiple providers:
- Yahoo Finance (US stocks)
- Alpha Vantage (API key required)
- Tushare (China stocks)
- AkShare (China stocks)

```bash
python examples/12_data_provider_usage.py
```

### 13. Bootstrap Statistical Analysis
**File**: `13_bootstrap_analysis.py`

Non-parametric statistical inference using resampling:
- Bootstrap confidence intervals for mean, std, Sharpe
- Significance testing
- Custom statistic bootstrap
- Sample size sensitivity analysis

```bash
python examples/13_bootstrap_analysis.py
```

### 14. Stress Testing
**File**: `14_stress_testing.py`

Stress testing and extreme scenario analysis:
- Historical extreme events analysis
- Custom scenario stress testing
- Tail risk analysis
- Correlation breakdown testing

```bash
python examples/14_stress_testing.py
```

### 15. Report Generation
**File**: `15_report_generation.py`

Strategy report generation:
- Basic report (returns only)
- Standard report (with benchmark)
- Full report (with positions and transactions)
- Custom styling options

```bash
python examples/15_report_generation.py
```

### 16. Custom Portfolio Optimization
**File**: `16_custom_optimization.py`

Constrained portfolio optimization:
- Max Sharpe ratio optimization
- Minimum variance portfolio
- Target return/volatility optimization
- Sector constraints
- Short selling constraints

```bash
python examples/16_custom_optimization.py
```

### 17. Visualization Backends
**File**: `17_visualization_backends.py`

Visualization system demonstration:
- Matplotlib backend (static plots)
- HTML backend (web-ready)
- Plotly backend (interactive)
- Bokeh backend (interactive)
- Custom styling options

```bash
python examples/17_visualization_backends.py
```

### 18. Positions Analysis
**File**: `18_positions_analysis.py`

Portfolio holdings analysis:
- Position allocation breakdown
- Long/short exposure analysis
- Leverage calculation
- Concentration metrics
- Sector exposure analysis

```bash
python examples/18_positions_analysis.py
```

### 19. Rolling Metrics Analysis
**File**: `19_rolling_metrics.py`

Rolling indicator analysis:
- Rolling Sharpe ratio
- Rolling volatility
- Rolling max drawdown
- Rolling Sortino ratio
- Rolling beta
- Window sensitivity analysis

```bash
python examples/19_rolling_metrics.py
```

### 20. Complete Workflow
**File**: `20_complete_workflow.py`

Complete quantitative analysis workflow:
1. Data preparation
2. Performance analysis
3. Rolling metrics
4. Risk assessment
5. Attribution
6. Stress testing
7. Portfolio optimization
8. Report generation
9. Comprehensive evaluation
10. Visualization

```bash
python examples/20_complete_workflow.py
```

## Running the Examples

All examples can be run with Python directly:

```bash
python examples/01_single_strategy_analysis.py
```

Some examples may require additional dependencies:

```bash
# Install with visualization support
pip install -e ".[viz]"

# Install with data providers
pip install yfinance tushare akshare

# Install with optional dependencies
pip install -e ".[all]"
```

## Output

Each example produces console output showing the analysis results. Many examples also generate visualization charts saved as PNG files.

## Example Coverage

| Example | Topic | Output |
|---------|-------|--------|
| 01 | Single Strategy Analysis | Console + HTML report |
| 02 | Multi-Strategy Comparison | Console |
| 03 | Factor Attribution | Console |
| 04 | Risk Management Report | Console |
| 05 | RollingEngine Metrics | Console + PNG |
| 06 | Advanced Backtesting | Console + PNG |
| 07 | Risk Models | Console + PNG |
| 08 | Portfolio Optimization | Console + PNG |
| 09 | Monte Carlo Simulation | Console + PNG |
| 10 | Market Timing Analysis | Console + PNG |
| 11 | Performance Attribution | Console + PNG |
| 12 | Data Provider Usage | Console |
| 13 | Bootstrap Analysis | Console + PNG |
| 14 | Stress Testing | Console + PNG |
| 15 | Report Generation | HTML + PDF |
| 16 | Custom Optimization | Console + PNG |
| 17 | Visualization Backends | HTML + PNG |
| 18 | Positions Analysis | Console + PNG |
| 19 | Rolling Metrics | Console + PNG |
| 20 | Complete Workflow | HTML + PNG |

## Example Categories

### Performance Analysis
- Examples 01-06: Strategy performance evaluation and comparison

### Risk Analysis
- Examples 07, 09, 13, 14: Risk measurement and stress testing

### Portfolio Management
- Examples 08, 11, 16: Portfolio optimization and attribution

### Data & Reporting
- Examples 12, 15, 17: Data fetching, visualization, and report generation

### Advanced Analysis
- Examples 18-20: Positions, rolling metrics, complete workflow

## More Documentation

For more detailed examples, see the documentation in `docs/`:
- User Guide: `docs/用户手册/`
- API Documentation: `docs/api文档/`
- Tutorials: `docs/教程/`
