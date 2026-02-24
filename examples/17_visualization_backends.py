"""
可视化后端示例

展示如何使用 fincore 的可视化系统：
- 多后端支持 (Matplotlib, HTML, Plotly, Bokeh)
- 累计收益图
- 回撤图
- 滚动夏普比率图
- 月度收益热力图

适用场景：
- 策略表现可视化
- 交互式图表生成
- HTML报告嵌入
- 自动化报告系统
"""

import numpy as np
import pandas as pd
from fincore import Empyrical
from fincore.viz import get_backend

print("=" * 70)
print("可视化后端示例")
print("=" * 70)

# 生成模拟数据
np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=252, freq="B", tz="UTC")

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

print(f"\n数据概览:")
print(f"  时间范围: {dates[0].date()} 至 {dates[-1].date()}")
print(f"  策略年化收益: {strategy_returns.mean() * 252:.4f}")
print(f"  基准年化收益: {benchmark_returns.mean() * 252:.4f}")

# ============================================================
# 1. Matplotlib 后端
# ============================================================
print("\n" + "=" * 70)
print("1. Matplotlib 后端")
print("=" * 70)

try:
    mpl_backend = get_backend("matplotlib")
    print("\nMatplotlib 后端已加载")

    # 累计收益
    cum_returns_strategy = (1 + strategy_returns).cumprod()
    cum_returns_benchmark = (1 + benchmark_returns).cumprod()

    fig = mpl_backend.plot_returns(
        cum_returns_strategy,
        benchmark=cum_returns_benchmark,
        title="累计收益对比"
    )
    print("  累计收益图已生成")

    # 回撤图
    drawdown = Empyrical.max_drawdown(strategy_returns)
    # 计算回撤时间序列
    cum = cum_returns_strategy.copy()
    running_max = cum.expanding().max()
    dd = (cum - running_max) / running_max

    fig = mpl_backend.plot_drawdown(
        dd,
        title="策略回撤"
    )
    print("  回撤图已生成")

    # 滚动夏普比率
    from fincore.metrics.rolling import roll_sharpe_ratio
    rolling_sharpe = roll_sharpe_ratio(strategy_returns, window=60)
    rolling_bench_sharpe = roll_sharpe_ratio(benchmark_returns, window=60)

    fig = mpl_backend.plot_rolling_sharpe(
        rolling_sharpe,
        benchmark_sharpe=rolling_bench_sharpe,
        window=60,
        title="滚动夏普比率 (60日)"
    )
    print("  滚动夏普比率图已生成")

    # 月度收益热力图
    fig = mpl_backend.plot_monthly_heatmap(
        strategy_returns,
        title="月度收益热力图"
    )
    print("  月度热力图已生成")

    print("\n提示: 使用 plt.show() 显示图表")

except ImportError:
    print("\n未安装 matplotlib，跳过 Matplotlib 后端")
except Exception as e:
    print(f"\nMatplotlib 后端错误: {e}")

# ============================================================
# 2. HTML 后端
# ============================================================
print("\n" + "=" * 70)
print("2. HTML 后端")
print("=" * 70)

try:
    html_backend = get_backend("html")
    print("\nHTML 后端已加载")

    # 累计收益
    html_fig = html_backend.plot_returns(
        cum_returns_strategy,
        benchmark=cum_returns_benchmark,
        title="累计收益对比"
    )

    # 保存到文件
    with open("viz_returns.html", "w") as f:
        f.write(html_fig)
    print("  累计收益图已保存: viz_returns.html")

    # 回撤图
    html_fig = html_backend.plot_drawdown(
        dd,
        title="策略回撤"
    )

    with open("viz_drawdown.html", "w") as f:
        f.write(html_fig)
    print("  回撤图已保存: viz_drawdown.html")

    # 滚动夏普比率
    html_fig = html_backend.plot_rolling_sharpe(
        rolling_sharpe,
        benchmark_sharpe=rolling_bench_sharpe,
        window=60,
        title="滚动夏普比率"
    )

    with open("viz_rolling_sharpe.html", "w") as f:
        f.write(html_fig)
    print("  滚动夏普比率图已保存: viz_rolling_sharpe.html")

    # 月度热力图
    html_fig = html_backend.plot_monthly_heatmap(
        strategy_returns,
        title="月度收益热力图"
    )

    with open("viz_monthly_heatmap.html", "w") as f:
        f.write(html_fig)
    print("  月度热力图已保存: viz_monthly_heatmap.html")

    print("\n提示: 在浏览器中打开 HTML 文件查看图表")

except Exception as e:
    print(f"\nHTML 后端错误: {e}")

# ============================================================
# 3. Plotly 后端 (交互式)
# ============================================================
print("\n" + "=" * 70)
print("3. Plotly 后端 (交互式)")
print("=" * 70)

try:
    plotly_backend = get_backend("plotly")
    print("\nPlotly 后端已加载")

    # 累计收益
    fig = plotly_backend.plot_returns(
        cum_returns_strategy,
        benchmark=cum_returns_benchmark,
        title="累计收益对比"
    )
    fig.write_html("viz_returns_plotly.html")
    print("  累计收益图已保存: viz_returns_plotly.html")

    # 回撤图
    fig = plotly_backend.plot_drawdown(
        dd,
        title="策略回撤"
    )
    fig.write_html("viz_drawdown_plotly.html")
    print("  回撤图已保存: viz_drawdown_plotly.html")

    print("\n提示: Plotly 图表支持缩放、平移等交互操作")

except ImportError:
    print("\n未安装 plotly，跳过 Plotly 后端")
except Exception as e:
    print(f"\nPlotly 后端错误: {e}")

# ============================================================
# 4. Bokeh 后端 (交互式)
# ============================================================
print("\n" + "=" * 70)
print("4. Bokeh 后端 (交互式)")
print("=" * 70)

try:
    bokeh_backend = get_backend("bokeh")
    print("\nBokeh 后端已加载")

    # 累计收益
    fig = bokeh_backend.plot_returns(
        cum_returns_strategy,
        benchmark=cum_returns_benchmark,
        title="累计收益对比"
    )

    from bokeh.io import output_file, save
    output_file("viz_returns_bokeh.html")
    save(fig)
    print("  累计收益图已保存: viz_returns_bokeh.html")

    print("\n提示: Bokeh 图表支持缩放、平移等交互操作")

except ImportError:
    print("\n未安装 bokeh，跳过 Bokeh 后端")
except Exception as e:
    print(f"\nBokeh 后端错误: {e}")

# ============================================================
# 5. 多资产收益对比
# ============================================================
print("\n" + "=" * 70)
print("5. 多资产收益对比")
print("=" * 70)

# 创建多资产收益
assets_returns = pd.DataFrame({
    'Asset1': strategy_returns,
    'Asset2': benchmark_returns,
    'Asset3': pd.Series(np.random.normal(0.0006, 0.014, len(dates)), index=dates),
    'Asset4': pd.Series(np.random.normal(0.0004, 0.011, len(dates)), index=dates),
})

try:
    mpl_backend = get_backend("matplotlib")
    cum_multi = (1 + assets_returns).cumprod()

    fig = mpl_backend.plot_returns(
        cum_multi,
        title="多资产累计收益对比"
    )
    print("\n多资产收益图已生成")

except Exception as e:
    print(f"\n多资产图生成错误: {e}")

# ============================================================
# 6. 使用 AnalysisContext 可视化
# ============================================================
print("\n" + "=" * 70)
print("6. 使用 AnalysisContext 快速可视化")
print("=" * 70)

try:
    from fincore import analyze

    ctx = analyze(strategy_returns, factor_returns=benchmark_returns)

    # 使用 plot 方法
    fig = ctx.plot()
    print("  AnalysisContext.plot() 已调用")

except Exception as e:
    print(f"\nAnalysisContext 可视化错误: {e}")

# ============================================================
# 7. 自定义图表样式
# ============================================================
print("\n" + "=" * 70)
print("7. 自定义图表样式")
print("=" * 70)

try:
    mpl_backend = get_backend("matplotlib")

    # 自定义样式参数
    fig = mpl_backend.plot_returns(
        cum_returns_strategy,
        benchmark=cum_returns_benchmark,
        title="自定义样式: 累计收益",
        figsize=(10, 6),  # 图表大小
        colors=['#1f77b4', '#ff7f0e'],  # 颜色
        linewidth=2,  # 线宽
        alpha=0.8,  # 透明度
        grid=True  # 网格
    )
    print("\n自定义样式图表已生成")

except Exception as e:
    print(f"\n自定义样式错误: {e}")

# ============================================================
# 8. 后端对比总结
# ============================================================
print("\n" + "=" * 70)
print("8. 可视化后端对比")
print("=" * 70)

print("""
+------------+----------+------------+----------+
| 后端       | 交互性   | 静态输出   | 依赖     |
+------------+----------+------------+----------+
| Matplotlib | 低       | PNG/PDF    | matplotlib|
| HTML       | 低       | HTML       | 无       |
| Plotly     | 高       | HTML       | plotly   |
| Bokeh      | 高       | HTML       | bokeh    |
+------------+----------+------------+----------+

选择建议:
- Matplotlib: 静态报告、论文插图
- HTML:      简单嵌入、无依赖
- Plotly:    交互式仪表板
- Bokeh:     大数据可视化
""")

# ============================================================
# 9. 保存所有图表为图片
# ============================================================
print("\n" + "=" * 70)
print("9. 批量保存图表")
print("=" * 70)

try:
    import matplotlib.pyplot as plt

    # 创建综合图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 累计收益
    ax = axes[0, 0]
    ax.plot(cum_returns_strategy.index, cum_returns_strategy.values * 100,
            label='策略', linewidth=2)
    ax.plot(cum_returns_benchmark.index, cum_returns_benchmark.values * 100,
            label='基准', linewidth=2, alpha=0.7)
    ax.set_ylabel('累计收益 (%)')
    ax.set_title('累计收益对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 回撤
    ax = axes[0, 1]
    ax.fill_between(dd.index, dd.values * 100, 0, alpha=0.3, color='red')
    ax.set_ylabel('回撤 (%)')
    ax.set_title('策略回撤')
    ax.grid(True, alpha=0.3)

    # 3. 滚动夏普
    ax = axes[1, 0]
    ax.plot(rolling_sharpe.index, rolling_sharpe.values,
            label='策略', linewidth=2)
    ax.plot(rolling_bench_sharpe.index, rolling_bench_sharpe.values,
            label='基准', linewidth=2, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Sharpe 比率')
    ax.set_title('滚动 Sharpe (60日)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 收益分布
    ax = axes[1, 1]
    ax.hist(strategy_returns * 100, bins=50, edgecolor='black',
            alpha=0.7, label='策略')
    ax.hist(benchmark_returns * 100, bins=50, edgecolor='black',
            alpha=0.5, label='基准')
    ax.set_xlabel('日收益率 (%)')
    ax.set_ylabel('频数')
    ax.set_title('收益分布')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('viz_comprehensive.png', dpi=100, bbox_inches='tight')
    print("\n综合图表已保存: viz_comprehensive.png")

except ImportError:
    print("\n未安装 matplotlib，跳过批量保存")

print("\n" + "=" * 70)
print("可视化示例完成！")
print("=" * 70)
print("""
生成的文件:
  - viz_returns.html           : HTML 累计收益图
  - viz_drawdown.html          : HTML 回撤图
  - viz_rolling_sharpe.html    : HTML 滚动夏普图
  - viz_monthly_heatmap.html   : HTML 月度热力图
  - viz_returns_plotly.html    : Plotly 交互图
  - viz_comprehensive.png      : 综合图表

使用建议:
  1. HTML 图表适合嵌入网页
  2. PNG 图片适合文档报告
  3. Plotly/Bokeh 适合交互式仪表板
  4. Matplotlib 适合自动化批量生成
""")
