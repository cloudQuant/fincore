# 测试最佳实践指南 (简化版)

## 快速开始

### 运行测试

```bash
# 所有测试
pytest tests/ -v

# 仅P0测试
pytest -m p0 -v

# 性能基准
pytest tests/benchmarks/ --benchmark-only

# 边缘情况
pytest tests/test_edge_cases.py -v

# 集成测试
pytest tests/integration/ -v

# 覆盖率
pytest tests/ --cov=fincore --cov-report=html
```

## 测试组织

```
tests/
├── conftest.py              # 全局fixtures
├── test_core/              # 核心功能
├── test_empyrical/         # API测试
├── test_metrics/           # Metrics模块
├── test_edge_cases.py      # 边缘情况
├── integration/            # 集成测试
└── benchmarks/             # 性能测试
```

## 测试优先级

- **P0**: 关键指标 (sharpe_ratio, max_drawdown, alpha, beta)
- **P1**: 常用功能
- **P2**: 次要功能
- **P3**: 装饰性功能

## 编写规范

```python
class TestSharpeRatio:
    """Sharpe ratio测试"""
    
    @pytest.mark.p0
    def test_normal_case(self, returns):
        """正常情况"""
        result = sharpe_ratio(returns)
        assert np.isfinite(result)
    
    @pytest.mark.p1
    def test_edge_case(self, empty_returns):
        """边缘情况"""
        result = sharpe_ratio(empty_returns)
        assert np.isnan(result)
```

## Fixtures

```python
# 在conftest.py中定义
@pytest.fixture
def small_returns():
    """252点数据"""
    np.random.seed(42)
    return pd.Series(np.random.randn(252) * 0.01)

# 使用
def test_metric(small_returns):
    result = sharpe_ratio(small_returns)
    assert np.isfinite(result)
```

## 覆盖率目标

| 模块 | 目标 |
|------|------|
| P0指标 | 90%+ |
| P1功能 | 85%+ |
| P2功能 | 80%+ |
| 整体 | 75%+ |

## 常用命令

```bash
# 运行特定标记
pytest -m p0
pytest -m "p0 or p1"
pytest -m "not slow"

# 并行运行
pytest tests/ -n 4

# 详细输出
pytest tests/ -v --tb=short

# 仅运行失败的测试
pytest tests/ --lf

# 停止于第一个失败
pytest tests/ -x
```

## 更多信息

详见完整版文档: `docs/testing-guide-full.md`
