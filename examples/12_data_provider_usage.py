"""
数据获取示例

展示如何使用 fincore 的数据提供者获取金融数据：
- Yahoo Finance 数据
- Alpha Vantage 数据
- Tushare 数据 (中国市场)
- AkShare 数据 (中国市场)

适用场景：
- 自动化数据获取
- 实时行情分析
- 历史数据回测
- 多数据源整合
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("数据获取示例")
print("=" * 70)

# ============================================================
# 1. Yahoo Finance 数据
# ============================================================
print("\n" + "=" * 70)
print("1. Yahoo Finance 数据")
print("=" * 70)

try:
    from fincore.data import YahooFinanceProvider

    # 创建数据提供者
    yahoo = YahooFinanceProvider()

    # 获取单只股票数据
    print("\n获取 AAPL 股票数据...")
    aapl_data = yahoo.get_prices(
        symbols='AAPL',
        start='2023-01-01',
        end='2024-01-01'
    )

    if aapl_data is not None and not aapl_data.empty:
        print(f"\nAAPL 数据概览:")
        print(f"  数据形状: {aapl_data.shape}")
        print(f"  列: {list(aapl_data.columns)}")
        print(f"  日期范围: {aapl_data.index[0].date()} 至 {aapl_data.index[-1].date()}")
        print(f"\n最近5个交易日数据:")
        print(aapl_data.tail())

        # 计算收益
        from fincore import simple_returns
        aapl_returns = simple_returns(aapl_data['Close'])
        print(f"\nAAPL 收益统计 (2023年):")
        print(f"  年化收益: {aapl_returns.mean() * 252:.4f}")
        print(f"  年化波动: {aapl_returns.std() * np.sqrt(252):.4f}")

    # 获取多只股票数据
    print("\n\n获取多只股票数据 (AAPL, MSFT, GOOGL)...")
    multi_data = yahoo.get_prices(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        start='2023-06-01',
        end='2024-01-01'
    )

    if multi_data is not None and not multi_data.empty:
        print(f"\n多股票数据形状: {multi_data.shape}")
        print(f"列名: {list(multi_data.columns)}")

except ImportError:
    print("\n未安装 yfinance，请运行: pip install yfinance")
except Exception as e:
    print(f"\nYahoo Finance 数据获取失败: {e}")

# ============================================================
# 2. Alpha Vantage 数据
# ============================================================
print("\n" + "=" * 70)
print("2. Alpha Vantage 数据")
print("=" * 70)

try:
    from fincore.data import AlphaVantageProvider

    # 注意: 需要设置 API key
    import os
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')

    if api_key:
        av = AlphaVantageProvider(api_key=api_key)

        print("\n获取 SPY (标普500 ETF) 数据...")
        spy_data = av.get_prices(
            symbols='SPY',
            start='2023-01-01',
            end='2024-01-01'
        )

        if spy_data is not None and not spy_data.empty:
            print(f"\nSPY 数据概览:")
            print(f"  数据形状: {spy_data.shape}")
            print(f"  最近价格: ${spy_data['Close'].iloc[-1]:.2f}")
    else:
        print("\n未设置 ALPHA_VANTAGE_API_KEY 环境变量")
        print("跳过 Alpha Vantage 数据获取")

except ImportError:
    print("\n未安装 requests，请运行: pip install requests")
except Exception as e:
    print(f"\nAlpha Vantage 数据获取失败: {e}")

# ============================================================
# 3. Tushare 数据 (中国市场)
# ============================================================
print("\n" + "=" * 70)
print("3. Tushare 数据 (中国市场)")
print("=" * 70)

try:
    import tushare as ts
    import os

    # 注意: 需要设置 Tushare token
    token = os.environ.get('TUSHARE_TOKEN')

    if token:
        from fincore.data import TushareProvider

        ts_provider = TushareProvider(token=token)

        print("\n获取平安银行 (000001.SZ) 数据...")
        bank_data = ts_provider.get_prices(
            symbols='000001.SZ',
            start='2023-01-01',
            end='2024-01-01'
        )

        if bank_data is not None and not bank_data.empty:
            print(f"\n平安银行数据概览:")
            print(f"  数据形状: {bank_data.shape}")
            print(f"  最近收盘价: {bank_data['Close'].iloc[-1]:.2f} 元")
    else:
        print("\n未设置 TUSHARE_TOKEN 环境变量")
        print("跳过 Tushare 数据获取")

except ImportError:
    print("\n未安装 tushare，请运行: pip install tushare")
except Exception as e:
    print(f"\nTushare 数据获取失败: {e}")

# ============================================================
# 4. AkShare 数据 (中国市场)
# ============================================================
print("\n" + "=" * 70)
print("4. AkShare 数据 (中国市场)")
print("=" * 70)

try:
    from fincore.data import AkShareProvider

    akshare = AkShareProvider()

    print("\n获取上证指数数据...")
    index_data = akshare.get_prices(
        symbols='000001',  # 上证指数代码
        start='2023-01-01',
        end='2024-01-01'
    )

    if index_data is not None and not index_data.empty:
        print(f"\n上证指数数据概览:")
        print(f"  数据形状: {index_data.shape}")
        print(f"  最近收盘: {index_data['Close'].iloc[-1]:.2f} 点")
    else:
        print("\nAkShare 数据获取失败或返回空数据")

except ImportError:
    print("\n未安装 akshare，请运行: pip install akshare")
except Exception as e:
    print(f"\nAkShare 数据获取失败: {e}")

# ============================================================
# 5. 数据提供者统一接口
# ============================================================
print("\n" + "=" * 70)
print("5. 数据提供者使用建议")
print("=" * 70)

print("""
推荐使用方式:

1. Yahoo Finance (免费，推荐用于美股)
   - 优点: 无需注册，数据覆盖面广
   - 缺点: 有请求频率限制

2. Alpha Vantage (免费，需注册)
   - 优点: 官方 API，稳定可靠
   - 缺点: 需要申请 API key

3. Tushare (中国股市)
   - 优点: 中国股市数据全面
   - 缺点: 需要注册获取 token

4. AkShare (中国股市，免费)
   - 优点: 无需注册，开箱即用
   - 缺点: 数据来源可能有变动

API Key 设置方法:
   export ALPHA_VANTAGE_API_KEY="your_key_here"
   export TUSHARE_TOKEN="your_token_here"
""")

# ============================================================
# 6. 自定义数据源示例
# ============================================================
print("\n" + "=" * 70)
print("6. 自定义数据源")
print("=" * 70)

print("""
如果需要使用自定义数据源，可以继承 DataProvider 基类:

```python
from fincore.data import DataProvider

class MyDataProvider(DataProvider):
    def get_prices(self, symbols, start=None, end=None):
        # 实现你的数据获取逻辑
        # 返回格式: pandas.DataFrame 或 dict
        pass

    def get_returns(self, symbols, start=None, end=None):
        # 实现收益数据获取逻辑
        pass
```

然后使用:

```python
my_provider = MyDataProvider()
data = my_provider.get_prices('MY_SYMBOL', '2023-01-01', '2024-01-01')
```
""")

# ============================================================
# 7. 数据缓存建议
# ============================================================
print("\n" + "=" * 70)
print("7. 数据缓存建议")
print("=" * 70)

print("""
对于频繁访问的数据，建议实现本地缓存:

```python
import pickle
from pathlib import Path

def get_cached_data(provider, symbol, start, end, cache_dir='cache'):
    cache_file = Path(cache_dir) / f'{symbol}_{start}_{end}.pkl'

    # 尝试从缓存加载
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # 获取新数据并缓存
    data = provider.get_prices(symbol, start, end)
    cache_file.parent.mkdir(exist_ok=True)

    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

    return data
```
""")

print("\n" + "=" * 70)
print("数据获取示例完成！")
print("=" * 70)
