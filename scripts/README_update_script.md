# Global Index Data Update Script

## 概述

`update_global_index_data.py` 脚本已优化，现在能够自动将全球指数数据保存到 `tests/test_data` 目录，用于单元测试。

## 主要改进

### 1. 智能数据根目录检测

```python
def get_data_root(default_path='tests/test_data', custom_path=None):
    """
    自动检测数据根目录：
    - 默认：tests/test_data (相对于项目根目录)
    - 可选：通过命令行参数指定自定义路径
    - 自动转换为绝对路径
    """
```

### 2. 自动目录创建

```python
def ensure_directory(path):
    """
    自动创建目录（如果不存在）
    使用 pathlib.Path 确保跨平台兼容性
    """
```

### 3. 改进的文件名处理

```python
# 自动处理非法文件名字符
safe_name = name.replace('/', '_').replace('\', '_')
file_path = os.path.join(data_root, f'global_index_{safe_name}.csv')
```

### 4. 详细输出和进度跟踪

- 显示数据获取进度
- 显示保存的文件路径
- 错误处理和报告
- 成功的可视化反馈

## 使用方法

### 基本使用（默认保存到 tests/test_data）

```bash
python scripts/update_global_index_data.py
```

### 自定义保存路径

```bash
python scripts/update_global_index_data.py /path/to/custom/data_dir
```

### 在项目根目录运行

```bash
# 推荐方式
python scripts/update_global_index_data.py

# 或者
python scripts/update_global_index_data.py tests/test_data
```

## 输出示例

```
============================================================
Updating global index data
Data root: F:/f/source_code/empyrical/tests/test_data
============================================================

Fetching global index spot data...
✓ Retrieved 150 global indices
✓ Saved main index data to: F:/f/source_code/empyrical/tests/test_data/global_index_data.csv

Found 150 unique indices

Fetching historical data for each index...
------------------------------------------------------------
[1/150] ✓ 纳斯达克100
[2/150] ✓ 标普500
[3/150] ✓ 道琼斯工业平均
...
[150/150] ✓ 日经225

============================================================
✓ Update complete!
✓ Data saved to: F:/f/source_code/empyrical/tests/test_data
============================================================
```

## 保存的文件

- `global_index_data.csv` - 全球指数概况数据
- `global_index_指数名称.csv` - 各指数历史数据

## 注意事项

1. 需要安装 `akshare` 库：`pip install akshare`
2. 数据来源：东方财富网
3. 数据将保存为 CSV 格式，便于在测试中使用
4. 脚本会自动跳过保存失败的指数，并显示错误信息

## 测试

运行以下命令测试脚本：

```bash
# 检查脚本语法
python -m py_compile scripts/update_global_index_data.py

# 查看帮助
python scripts/update_global_index_data.py --help
```

## 集成到CI/CD

可以将此脚本添加到持续集成流程中，定期更新测试数据：

```yaml
# .github/workflows/update-test-data.yml
name: Update Test Data
on:
  schedule:
    - cron: '0 0 * * 0'  # 每周更新
jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Update global index data
        run: python scripts/update_global_index_data.py
```

## 版本历史

### v1.1 (修复版本)
- **修复**: 解决了 `progress` 变量作用域错误
  - 问题：变量在 `try` 块中定义，但被 `except` 块引用
  - 解决：将 `progress` 变量定义移到循环开始处，确保每次迭代都被定义
- **修复**: 移除了不必要的 `time.sleep(5)` 调用
- **优化**: 简化了文件名处理逻辑（移除对 `\` 的替换，避免转义问题）

### v1.0 (初始版本)
- 智能数据根目录检测
- 自动目录创建
- 进度跟踪
- 错误处理

## 已知问题

如果您遇到以下错误：
```
cannot access local variable 'progress' where it is not associated with a value
```

请确保使用的是最新版本 (v1.1+)。这个错误已在 v1.1 中修复。
