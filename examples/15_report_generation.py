"""
策略报告生成示例

展示如何使用 fincore 生成专业的策略分析报告：
- HTML 报告生成
- 多层次报告 (基础/标准/完整)
- 自定义报告样式
- PDF 导出

适用场景：
- 策略分析报告
- 投资者汇报
- 定期绩效报告
- 研究报告生成
"""

import numpy as np
import pandas as pd
from fincore.report import create_strategy_report
from fincore import simple_returns

print("=" * 70)
print("策略报告生成示例")
print("=" * 70)

# ============================================================
# 1. 准备数据
# ============================================================
print("\n数据准备...")

# 生成模拟策略数据
np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=252*2, freq="B", tz="UTC")

# 策略收益
strategy_returns = pd.Series(
    np.random.normal(0.0008, 0.015, len(dates)),
    index=dates,
    name="strategy"
)

# 基准收益
benchmark_returns = pd.Series(
    np.random.normal(0.0005, 0.012, len(dates)),
    index=dates,
    name="benchmark"
)

# 持仓数据 (简化版)
positions = pd.DataFrame(
    np.random.uniform(-100000, 100000, (len(dates), 5)),
    index=dates,
    columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
)

# 交易数据 (简化版)
transactions_data = []
for i in range(100):
    txn_date = np.random.choice(dates)
    transactions_data.append({
        'date': txn_date,
        'symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']),
        'amount': np.random.randint(-100, 100),
        'price': np.random.uniform(100, 200)
    })
transactions = pd.DataFrame(transactions_data)

print(f"  策略收益: {len(strategy_returns)} 个交易日")
print(f"  基准收益: {len(benchmark_returns)} 个交易日")
print(f"  持仓数据: {positions.shape}")
print(f"  交易记录: {len(transactions)} 笔")

# ============================================================
# 2. 基础报告 (仅收益)
# ============================================================
print("\n" + "=" * 70)
print("1. 生成基础报告 (仅策略收益)")
print("=" * 70)

create_strategy_report(
    strategy_returns,
    title="我的量化策略 - 基础报告",
    output="report_basic.html"
)
print("\n基础报告已保存: report_basic.html")
print("  包含内容:")
print("    - 累计收益曲线")
print("    - 基本绩效指标 (收益、波动率、Sharpe、最大回撤等)")
print("    - 月度/年度收益热力图")
print("    - 收益分布统计")

# ============================================================
# 3. 标准报告 (收益 + 基准)
# ============================================================
print("\n" + "=" * 70)
print("2. 生成标准报告 (策略收益 + 基准)")
print("=" * 70)

create_strategy_report(
    strategy_returns,
    benchmark_rets=benchmark_returns,
    title="我的量化策略 - 标准报告",
    output="report_standard.html"
)
print("\n标准报告已保存: report_standard.html")
print("  包含内容:")
print("    - 所有基础报告内容")
print("    - Alpha 和 Beta 分析")
print("    - 信息比率、跟踪误差")
print("    - 滚动 Alpha/Beta")
print("    - 上下行捕获比率")

# ============================================================
# 4. 完整报告 (包含持仓和交易)
# ============================================================
print("\n" + "=" * 70)
print("3. 生成完整报告 (包含持仓和交易)")
print("=" * 70)

create_strategy_report(
    strategy_returns,
    benchmark_rets=benchmark_returns,
    positions=positions,
    transactions=transactions,
    title="我的量化策略 - 完整报告",
    output="report_full.html"
)
print("\n完整报告已保存: report_full.html")
print("  包含内容:")
print("    - 所有标准报告内容")
print("    - 持仓分析 (多头/空头暴露)")
print("    - 杠杆分析")
print("    - 持仓集中度")
print("    - 换手率分析")
print("    - 交易统计")

# ============================================================
# 5. 自定义样式报告
# ============================================================
print("\n" + "=" * 70)
print("4. 生成自定义样式报告")
print("=" * 70)

create_strategy_report(
    strategy_returns,
    benchmark_rets=benchmark_returns,
    title="我的量化策略 - 自定义样式",
    output="report_custom.html",
    # 自定义样式参数
    theme='dark'  # 使用深色主题 (如果支持)
)
print("\n自定义样式报告已保存: report_custom.html")

# ============================================================
# 6. 报告对比
# ============================================================
print("\n" + "=" * 70)
print("5. 报告类型对比")
print("=" * 70)

print("""
+------------------+----------------------+----------------------+----------------------+
| 报告内容         | 基础报告            | 标准报告            | 完整报告            |
+------------------+----------------------+----------------------+----------------------+
| 累计收益曲线     | ✓                   | ✓                   | ✓                   |
| 基本绩效指标     | ✓                   | ✓                   | ✓                   |
| 月度热力图       | ✓                   | ✓                   | ✓                   |
| 收益分布         | ✓                   | ✓                   | ✓                   |
| 回撤分析         | ✓                   | ✓                   | ✓                   |
+------------------+----------------------+----------------------+----------------------+
| Alpha/Beta       | ✗                   | ✓                   | ✓                   |
| 信息比率         | ✗                   | ✓                   | ✓                   |
| 滚动指标         | ✗                   | ✓                   | ✓                   |
| 捕获比率         | ✗                   | ✓                   | ✓                   |
+------------------+----------------------+----------------------+----------------------+
| 持仓分析         | ✗                   | ✗                   | ✓                   |
| 杠杆分析         | ✗                   | ✗                   | ✓                   |
| 换手率           | ✗                   | ✗                   | ✓                   |
| 交易统计         | ✗                   | ✗                   | ✓                   |
+------------------+----------------------+----------------------+----------------------+
| 文件大小         | ~50 KB              | ~80 KB              | ~120 KB             |
| 生成时间         | ~1 秒               | ~2 秒               | ~3 秒               |
+------------------+----------------------+----------------------+----------------------+
""")

# ============================================================
# 7. PDF 报告生成 (如果支持)
# ============================================================
print("\n" + "=" * 70)
print("6. PDF 报告生成")
print("=" * 70)

try:
    create_strategy_report(
        strategy_returns,
        benchmark_rets=benchmark_returns,
        title="我的量化策略 - PDF报告",
        output="report.pdf"
    )
    print("\nPDF 报告已保存: report.pdf")
except Exception as e:
    print(f"\nPDF 生成失败: {e}")
    print("  注意: PDF 生成需要安装额外的依赖 (playwright)")

# ============================================================
# 8. 报告查看建议
# ============================================================
print("\n" + "=" * 70)
print("7. 报告查看建议")
print("=" * 70)

print("""
查看报告:
  1. 直接在浏览器中打开 HTML 文件
  2. PDF 报告适合打印和分享
  3. HTML 报告支持交互式图表

报告定制:
  - 可以通过修改参数自定义报告样式
  - 报告标题、作者信息等可以自定义
  - 支持深色/浅色主题切换

报告用途:
  - 基础报告: 快速查看策略表现
  - 标准报告: 与基准对比分析
  - 完整报告: 全面策略分析报告
""")

# ============================================================
# 9. 报告数据来源说明
# ============================================================
print("\n" + "=" * 70)
print("8. 报告数据来源")
print("=" * 70)

print("""
报告所需数据层次:

1. 最小数据 (仅收益):
   - strategy_returns (必需): pd.Series
     日度策略收益率序列

2. 标准数据 (收益 + 基准):
   - strategy_returns (必需): 策略收益
   - benchmark_rets (可选): 基准收益 pd.Series
     用于计算相对指标

3. 完整数据 (包含持仓和交易):
   - strategy_returns (必需): 策略收益
   - benchmark_rets (可选): 基准收益
   - positions (可选): 持仓 pd.DataFrame
     每日各资产的持仓金额
   - transactions (可选): 交易 pd.DataFrame
     包含 date, symbol, amount, price 等列
   - segment_returns (可选): 分组收益
     按行业/因子等分组的收益

数据格式要求:
  - 索引: DatetimeIndex, 时区建议使用 UTC
  - 频率: 日度数据最佳
  - 缺失值: 自动处理 NaN
""")

print("\n" + "=" * 70)
print("策略报告生成示例完成！")
print("=" * 70)
print("""
生成的报告文件:
  - report_basic.html    : 基础报告
  - report_standard.html : 标准报告
  - report_full.html     : 完整报告
  - report_custom.html   : 自定义样式报告
  - report.pdf           : PDF报告 (如果支持)

建议:
  1. 根据需求选择合适的报告类型
  2. HTML 报告可在浏览器中直接查看
  3. PDF 报告适合正式文档和分享
  4. 可以将报告自动化集成到策略系统中
""")
