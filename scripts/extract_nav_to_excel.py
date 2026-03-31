from pathlib import Path

import pandas as pd

# 读取原始Excel
path = Path('/Users/yunjinqi/Documents/new_projects/fincore/2023-04-21---基金净值变化表.xlsx')
df = pd.read_excel(path, sheet_name='结算价净值表')

# 提取日期和累计净值列
cleaned = df[['日期', '累计净值']].copy()
cleaned = cleaned.dropna(subset=['日期', '累计净值'])
cleaned['日期'] = pd.to_datetime(cleaned['日期'], errors='coerce')
cleaned = cleaned.dropna(subset=['日期'])
cleaned = cleaned.sort_values('日期')
cleaned = cleaned.set_index('日期')

# 按周重采样，取每周最后一个交易日的净值
weekly_nav = cleaned['累计净值'].resample('W-FRI').last().dropna()
weekly_nav = weekly_nav.reset_index()
weekly_nav.columns = ['日期', '累计净值']
weekly_nav['日期'] = weekly_nav['日期'].dt.strftime('%Y-%m-%d')

# 保存为新的Excel
output = Path('/Users/yunjinqi/Documents/new_projects/fincore/累计净值数据_周度.xlsx')
weekly_nav.to_excel(output, index=False, sheet_name='累计净值')
print(f'输出文件: {output}')
print(f'数据行数: {len(weekly_nav)}')
print(f'日期范围: {weekly_nav["日期"].iloc[0]} -> {weekly_nav["日期"].iloc[-1]}')
print('\n前5行:')
print(weekly_nav.head().to_string(index=False))
