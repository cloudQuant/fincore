# Abberation (Bollinger Band Breakout) Strategy

## Overview

Abberation is a classic trend-following strategy based on Bollinger Band breakouts:

- Go long when price breaks above the upper band.
- Go short when price breaks below the lower band.
- Exit when price reverts through the middle band.

This example uses 1x leverage sizing based on total account value.

## Strategy Mechanics

### Bollinger Bands

Bollinger Bands consist of three lines:

1. Middle band: N-period moving average (default: 200)
2. Upper band: middle band + K * standard deviation (default: K=2)
3. Lower band: middle band - K * standard deviation (default: K=2)

### Trading Rules

Long entry:

- Previous close < upper band
- Current close > upper band (upward breakout)
- Size the position with ~1x leverage

Short entry:

- Previous close > lower band
- Current close < lower band (downward breakout)
- Size the position with ~1x leverage

Exits:

- Long: close when price falls below the middle band
- Short: close when price rises above the middle band

Position sizing:

- lots = total_value / (contract_multiplier * price)

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| boll_period | int | 200 | Bollinger lookback period |
| boll_mult | float | 2 | Std-dev multiplier |

## When It Works

1. Market regime: strong directional trends
2. Instruments: medium-vol futures such as rebar
3. Timeframe: daily bars
4. Best behavior: clean breakouts rather than mean-reverting ranges

## Risks / Caveats

1. Range markets: repeated whipsaws can cause consecutive losses
2. False breakouts: price can briefly break and quickly revert
3. Leverage risk: leverage amplifies both gains and losses
4. Lag: the middle band can lag, potentially delaying exits
5. Slippage: large orders may incur meaningful slippage

## Backtest Example (Backtrader)

```python
import backtrader as bt
from backtrader.comminfo import ComminfoFuturesPercent

from run import load_rb889_data
from strategy_abberation import AbberationStrategy, RbPandasFeed

cerebro = bt.Cerebro()

df = load_rb889_data("RB889.csv")
feed = RbPandasFeed(dataname=df)
cerebro.adddata(feed, name="RB")

comm = ComminfoFuturesPercent(commission=0.0001, margin=0.10, mult=10)
cerebro.broker.addcommissioninfo(comm, name="RB")
cerebro.broker.setcash(1_000_000)

cerebro.addstrategy(AbberationStrategy, boll_period=200, boll_mult=2)
results = cerebro.run()
```

## References

1. Bollinger Bands basics
2. Standard deviation in time-series analysis
3. Trend-following strategies
