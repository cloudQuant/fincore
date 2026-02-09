"""
AbberationStrategy 策略分析脚本

读取 logs 目录下的策略运行结果，使用 fincore 进行全面的绩效分析。

使用方式:
    python analyze_strategy.py
"""
import os
import json
import pandas as pd
import numpy as np

import fincore
from fincore import Empyrical

# =========================================================================
# 1. 定位日志目录
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")

# 自动查找最新的运行目录
run_dirs = sorted(
    [d for d in os.listdir(LOGS_DIR) if os.path.isdir(os.path.join(LOGS_DIR, d))],
)
if not run_dirs:
    raise FileNotFoundError(f"No run directories found in {LOGS_DIR}")
RUN_DIR = os.path.join(LOGS_DIR, run_dirs[-1])
print(f"分析运行目录: {RUN_DIR}\n")

# =========================================================================
# 2. 读取运行信息
# =========================================================================
with open(os.path.join(RUN_DIR, "run_info.json"), "r") as f:
    run_info = json.load(f)

print("=" * 70)
print(f"策略名称:   {run_info['strategy_name']}")
print(f"运行时间:   {run_info['run_datetime']}")
print(f"运行ID:     {run_info['run_id']}")
print("=" * 70)

# =========================================================================
# 3. 读取 value.log → 计算日收益率
# =========================================================================
value_df = pd.read_csv(
    os.path.join(RUN_DIR, "value.log"),
    sep="\t",
    usecols=["dt", "value", "cash"],
)
value_df["dt"] = pd.to_datetime(value_df["dt"])

# 取每日最后一个bar的value作为日终净值
daily_value = value_df.groupby(value_df["dt"].dt.date)["value"].last()
daily_value.index = pd.to_datetime(daily_value.index)
daily_value.name = "portfolio_value"

# 计算日收益率
daily_returns = daily_value.pct_change().dropna()
daily_returns.name = "strategy"

initial_capital = value_df["value"].iloc[0]
final_value = value_df["value"].iloc[-1]

print(f"\n初始资金:   {initial_capital:>15,.2f}")
print(f"最终净值:   {final_value:>15,.2f}")
print(f"总收益:     {final_value - initial_capital:>15,.2f}")
print(f"总收益率:   {(final_value / initial_capital - 1) * 100:>14.2f}%")
print(f"交易日数:   {len(daily_returns):>15d}")
print(f"日期范围:   {daily_returns.index[0].strftime('%Y-%m-%d')} → {daily_returns.index[-1].strftime('%Y-%m-%d')}")

# =========================================================================
# 4. 读取 trade.log → 交易统计
# =========================================================================
trade_df = pd.read_csv(os.path.join(RUN_DIR, "trade.log"), sep="\t")

# 只看已关闭的交易
closed_trades = trade_df[trade_df["status"] == "Closed"].copy()
closed_trades["dtopen"] = pd.to_datetime(closed_trades["dtopen"])
closed_trades["dtclose"] = pd.to_datetime(closed_trades["dtclose"])

total_trades = len(closed_trades)
winning_trades = closed_trades[closed_trades["pnlcomm"] > 0]
losing_trades = closed_trades[closed_trades["pnlcomm"] <= 0]
long_trades = closed_trades[closed_trades["long"] == 1]
short_trades = closed_trades[closed_trades["long"] == 0]

total_commission = closed_trades["commission"].sum()
total_pnl = closed_trades["pnl"].sum()
total_pnl_after_comm = closed_trades["pnlcomm"].sum()

print(f"\n{'=' * 70}")
print("交易统计")
print(f"{'=' * 70}")
print(f"总交易数:       {total_trades}")
print(f"  多头交易:     {len(long_trades)}")
print(f"  空头交易:     {len(short_trades)}")
print(f"盈利交易:       {len(winning_trades)}  ({len(winning_trades)/total_trades*100:.1f}%)")
print(f"亏损交易:       {len(losing_trades)}  ({len(losing_trades)/total_trades*100:.1f}%)")
print(f"总盈亏:         {total_pnl:>15,.2f}")
print(f"总手续费:       {total_commission:>15,.2f}")
print(f"净盈亏:         {total_pnl_after_comm:>15,.2f}")

if len(winning_trades) > 0:
    avg_win = winning_trades["pnlcomm"].mean()
    max_win = winning_trades["pnlcomm"].max()
else:
    avg_win = max_win = 0

if len(losing_trades) > 0:
    avg_loss = losing_trades["pnlcomm"].mean()
    max_loss = losing_trades["pnlcomm"].min()
else:
    avg_loss = max_loss = 0

print(f"\n平均盈利:       {avg_win:>15,.2f}")
print(f"平均亏损:       {avg_loss:>15,.2f}")
print(f"最大单笔盈利:   {max_win:>15,.2f}")
print(f"最大单笔亏损:   {max_loss:>15,.2f}")
if avg_loss != 0:
    print(f"盈亏比:         {abs(avg_win / avg_loss):>15.2f}")

# 持仓时长统计
closed_trades["holding_bars"] = closed_trades["barlen"]
print(f"\n平均持仓K线数:  {closed_trades['holding_bars'].mean():>15.1f}")
print(f"最长持仓K线数:  {closed_trades['holding_bars'].max():>15d}")
print(f"最短持仓K线数:  {closed_trades['holding_bars'].min():>15d}")

# =========================================================================
# 5. 使用 fincore 进行绩效分析
# =========================================================================
print(f"\n{'=' * 70}")
print("fincore 绩效分析")
print(f"{'=' * 70}")

# --- 方式一: Flat API ---
print("\n--- 核心指标 (Flat API) ---")
print(f"年化收益率:     {fincore.annual_return(daily_returns):.4f}")
print(f"年化波动率:     {fincore.annual_volatility(daily_returns):.4f}")
print(f"夏普比率:       {fincore.sharpe_ratio(daily_returns):.4f}")
print(f"最大回撤:       {fincore.max_drawdown(daily_returns):.4f}")
print(f"索提诺比率:     {fincore.sortino_ratio(daily_returns):.4f}")
print(f"卡尔玛比率:     {fincore.calmar_ratio(daily_returns):.4f}")
print(f"累计收益:       {fincore.cum_returns_final(daily_returns):.4f}")
print(f"在险价值(5%):   {fincore.value_at_risk(daily_returns):.4f}")
print(f"下行风险:       {fincore.downside_risk(daily_returns):.4f}")
print(f"尾部比率:       {fincore.tail_ratio(daily_returns):.4f}")

# --- 方式二: Empyrical 类级别调用 ---
print("\n--- 扩展指标 (Empyrical 类调用) ---")
print(f"偏度:           {Empyrical.skewness(daily_returns):.4f}")
print(f"峰度:           {Empyrical.kurtosis(daily_returns):.4f}")
print(f"Omega比率:      {Empyrical.omega_ratio(daily_returns):.4f}")
print(f"Hurst指数:      {Empyrical.hurst_exponent(daily_returns):.4f}")
print(f"时序稳定性:     {Empyrical.stability_of_timeseries(daily_returns):.4f}")

# 连续涨跌
print(f"\n最大连续上涨天: {Empyrical.max_consecutive_up_days(daily_returns)}")
print(f"最大连续下跌天: {Empyrical.max_consecutive_down_days(daily_returns)}")
print(f"单日最大收益:   {Empyrical.max_single_day_gain(daily_returns):.4f}")
print(f"单日最大亏损:   {Empyrical.max_single_day_loss(daily_returns):.4f}")

# --- 方式三: Empyrical 实例调用 (自动填充 returns) ---
print("\n--- 实例方法 (自动填充 returns) ---")
emp = Empyrical(returns=daily_returns)

print(f"胜率:           {emp.win_rate():.4f}")
print(f"亏损率:         {emp.loss_rate():.4f}")
print(f"序列相关:       {emp.serial_correlation():.4f}")
print(f"常识比率:       {emp.common_sense_ratio():.4f}")
print(f"最大回撤天数:   {emp.max_drawdown_days()}")
print(f"最大回撤恢复天: {emp.max_drawdown_recovery_days()}")
print(f"第二大回撤:     {emp.second_max_drawdown():.4f}")
print(f"第三大回撤:     {emp.third_max_drawdown():.4f}")

# 斯特林比率 / 伯克比率
print(f"斯特林比率:     {emp.sterling_ratio():.4f}")
print(f"伯克比率:       {emp.burke_ratio():.4f}")
print(f"Kappa3比率:     {emp.kappa_three_ratio():.4f}")

# 按年分析
print(f"\n--- 按年统计 ---")
annual_by_year = Empyrical.annual_return_by_year(daily_returns)
sharpe_by_year = Empyrical.sharpe_ratio_by_year(daily_returns)
dd_by_year = Empyrical.max_drawdown_by_year(daily_returns)

yearly_stats = pd.DataFrame({
    "年化收益": annual_by_year,
    "夏普比率": sharpe_by_year,
    "最大回撤": dd_by_year,
})
print(yearly_stats.to_string(float_format=lambda x: f"{x:.4f}"))

# 综合统计表
print(f"\n--- 综合统计表 (perf_stats) ---")
stats = Empyrical.perf_stats(daily_returns)
print(stats.to_string(float_format=lambda x: f"{x:.4f}"))

# 回撤分析
print(f"\n--- Top 5 回撤 ---")
dd_table = Empyrical.gen_drawdown_table(daily_returns, top=5)
print(dd_table.to_string())

# 月度收益聚合
print(f"\n--- 月度收益 ---")
monthly_returns = Empyrical.aggregate_returns(daily_returns, "monthly")
print(monthly_returns.tail(12).to_string(float_format=lambda x: f"{x:.4f}"))

# =========================================================================
# 6. 汇总
# =========================================================================
print(f"\n{'=' * 70}")
print("分析完成")
print(f"{'=' * 70}")
print(f"策略: {run_info['strategy_name']}")
print(f"品种: RB (螺纹钢)")
print(f"区间: {daily_returns.index[0].strftime('%Y-%m-%d')} → {daily_returns.index[-1].strftime('%Y-%m-%d')}")
print(f"年化收益: {fincore.annual_return(daily_returns):.2%}")
print(f"夏普比率: {fincore.sharpe_ratio(daily_returns):.4f}")
print(f"最大回撤: {fincore.max_drawdown(daily_returns):.2%}")
print(f"交易次数: {total_trades}")
print(f"胜率:     {len(winning_trades)/total_trades:.2%}")
