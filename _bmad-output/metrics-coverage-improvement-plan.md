# Metrics模块测试覆盖率提升计划

**目标**: 将Metrics模块测试覆盖率从当前的19%提升至80%+

**优先级**: P3 (持续改进)

---

## 📊 当前覆盖率状态

| 模块 | 覆盖率 | 优先级 | 状态 |
|------|--------|--------|------|
| alpha_beta.py | 68% | P0 | ⚠️ 中等 |
| basic.py | 37% | P1 | ⚠️ 低 |
| ratios.py | 20% | P0 | ❌ 需改进 |
| risk.py | 18% | P0 | ❌ 需改进 |
| returns.py | 51% | P1 | ⚠️ 中等 |
| rolling.py | 19% | P2 | ❌ 需改进 |
| yearly.py | 20% | P2 | ❌ 需改进 |
| stats.py | 10% | P1 | ❌ 低 |
| drawdown.py | 38% | P1 | ⚠️ 中等 |
| **bayesian.py** | **0%** | P2 | ❌ 未测试 |
| **consecutive.py** | **0%** | P2 | ❌ 未测试 |
| **perf_attrib.py** | **0%** | P2 | ❌ 未测试 |
| **perf_stats.py** | **0%** | P2 | ❌ 未测试 |
| **positions.py** | **0%** | P2 | ❌ 未测试 |
| **round_trips.py** | **0%** | P2 | ❌ 未测试 |
| **timing.py** | **0%** | P2 | ❌ 未测试 |
| **transactions.py** | **0%** | P2 | ❌ 未测试 |

---

## 🎯 提升策略

### 第一轮：P0模块优先 (2-3周)

#### 1. ratios.py (目标: 20% → 85%)

**当前未覆盖功能**:
- stability_of_timeseries
- simple_alpha_beta
- capture
- up_capture / down_capture

**测试计划**:
```python
# tests/test_metrics/test_ratios_complete.py

class TestStabilityOfTimeseries:
    """稳定性测试"""
    
    def test_stable_returns(self):
        """稳定收益序列"""
        returns = pd.Series([0.01] * 100)
        result = stability_of_timeseries(returns)
        assert 0 <= result <= 1
    
    def test_volatile_returns(self):
        """高波动收益"""
        returns = pd.Series(np.random.randn(100) * 0.1)
        result = stability_of_timeseries(returns)
        assert 0 <= result <= 1

class TestCaptureRatios:
    """Capture比率测试"""
    
    def test_up_capture(self):
        """上涨捕获"""
        # ... test implementation
```

**预计工作量**: 2-3天  
**新增测试**: ~30个

---

#### 2. risk.py (目标: 18% → 85%)

**当前未覆盖功能**:
- tail_ratio
- value_at_risk (完整覆盖)
- conditional_value_at_risk
- custom_value_at_risk

**测试计划**:
```python
# tests/test_metrics/test_risk_complete.py

class TestTailRatio:
    """尾比率测试"""
    
    def test_normal_distribution(self):
        """正态分布"""
        returns = pd.Series(np.random.randn(1000))
        result = tail_ratio(returns)
        assert result > 0
    
    def test_fat_tails(self):
        """肥尾分布"""
        # Heavy tail distribution test

class TestCustomVaR:
    """自定义VaR测试"""
    
    def test_historical_var(self):
        """历史模拟法"""
        # ... implementation
```

**预计工作量**: 2-3天  
**新增测试**: ~25个

---

### 第二轮：P1模块完善 (2-3周)

#### 3. basic.py (目标: 37% → 90%)

**未覆盖功能**:
- ensure_datetime_index_series
- flatten
- adjust_returns
- annualization_factor

**新增测试**: ~20个  
**工作量**: 1-2天

#### 4. stats.py (目标: 10% → 80%)

**未覆盖功能**:
- skewness
- kurtosis
- hurst_exponent
- stab

**新增测试**: ~30个  
**工作量**: 2-3天

---

### 第三轮：0%覆盖模块 (3-4周)

#### 5. positions.py (目标: 0% → 80%)

**功能**:
- exposure
- concentration
- leverage

**测试计划**:
```python
# tests/test_metrics/test_positions.py

class TestExposure:
    """持仓敞口测试"""
    
    def test_long_positions(self):
        """多头持仓"""
        positions = pd.DataFrame({
            'AAPL': [100, 110, 105],
            'GOOGL': [50, 55, 52],
        })
        result = exposure(positions)
        # ... assertions
```

**新增测试**: ~25个  
**工作量**: 2-3天

---

#### 6. transactions.py (目标: 0% → 80%)

**功能**:
- turnover
- slippage
- volume

**新增测试**: ~20个  
**工作量**: 1-2天

---

#### 7. round_trips.py (目标: 0% → 75%)

**功能**:
- extract_round_trips
- add_round_trips_stats

**新增测试**: ~15个  
**工作量**: 1-2天

---

#### 8. timing.py (目标: 0% → 75%)

**功能**:
- treynor_mazuy
- henriksson_merton

**新增测试**: ~15个  
**工作量**: 1-2天

---

#### 9. consecutive.py (目标: 0% → 80%)

**功能**:
- max_consecutive_up
- max_consecutive_down
- consecutive_wins_losses

**新增测试**: ~20个  
**工作量**: 1-2天

---

#### 10. perf_stats.py (目标: 0% → 85%)

**功能**:
- perf_stats
- bootstrap_perf_stats

**新增测试**: ~15个  
**工作量**: 1-2天

---

### 第四轮：高级模块 (2-3周)

#### 11. bayesian.py (目标: 0% → 70%)

**功能**:
- bayesian_sharpe_ratio
- bayesian_volatility

**注意**: 需要pymc依赖  
**新增测试**: ~10个  
**工作量**: 2-3天

#### 12. perf_attrib.py (目标: 0% → 75%)

**功能**:
- perf_attrib
- factor_returns
- specific_returns

**新增测试**: ~20个  
**工作量**: 2-3天

---

## 📈 覆盖率提升时间表

| 周次 | 目标模块 | 目标覆盖率 | 状态 |
|------|---------|-----------|------|
| Week 1-2 | ratios.py, risk.py | 85% | ⏳ 计划中 |
| Week 3-4 | basic.py, stats.py | 85% | ⏳ 计划中 |
| Week 5-7 | positions, transactions, round_trips | 75-80% | ⏳ 计划中 |
| Week 8-9 | timing, consecutive, perf_stats | 75-85% | ⏳ 计划中 |
| Week 10-12 | bayesian, perf_attrib | 70-75% | ⏳ 计划中 |

---

## 🎯 成功标准

### 覆盖率目标

| 指标 | 当前 | 目标 | 期限 |
|------|------|------|------|
| **整体Metrics覆盖率** | 19% | **80%** | 12周 |
| **P0模块覆盖率** | ~40% | **90%** | 6周 |
| **P1模块覆盖率** | ~30% | **85%** | 8周 |
| **P2模块覆盖率** | ~10% | **75%** | 12周 |

### 质量标准

- ✅ 所有新测试必须包含边缘情况
- ✅ 所有新测试必须有文档字符串
- ✅ 所有新测试必须通过pytest
- ✅ 性能测试必须满足阈值

---

## 📝 测试编写规范

### 1. 测试文件组织

```
tests/
├── test_metrics/
│   ├── test_ratios_complete.py    # ratios.py完整测试
│   ├── test_risk_complete.py      # risk.py完整测试
│   ├── test_positions.py          # positions.py测试
│   ├── test_transactions.py       # transactions.py测试
│   └── ...
```

### 2. 测试类命名

```python
class Test{FunctionName}:
    """测试{功能描述}"""
    
    def test_normal_case(self):
        """正常情况测试"""
        pass
    
    def test_edge_case_{type}(self):
        """边缘情况：{类型}"""
        pass
```

### 3. 测试优先级标记

```python
@pytest.mark.p0  # 核心指标
@pytest.mark.p1  # 重要功能
@pytest.mark.p2  # 次要功能
```

---

## 🔧 实施工具

### 覆盖率测量

```bash
# 单模块覆盖率
pytest tests/test_metrics/test_ratios_complete.py \
  --cov=fincore.metrics.ratios \
  --cov-report=term-missing

# 整体覆盖率
pytest tests/ \
  --cov=fincore.metrics \
  --cov-report=html
```

### 覆盖率报告

```bash
# 生成HTML报告
pytest --cov=fincore --cov-report=html
open htmlcov/index.html

# 生成缺失行报告
pytest --cov=fincore --cov-report=term-missing | grep "TOTAL"
```

---

## 📊 进度跟踪

### 里程碑

- [ ] **Milestone 1**: P0模块达到90%覆盖率 (Week 6)
- [ ] **Milestone 2**: P1模块达到85%覆盖率 (Week 8)
- [ ] **Milestone 3**: 0%模块达到75%覆盖率 (Week 10)
- [ ] **Milestone 4**: 整体达到80%覆盖率 (Week 12)

### 每周检查

- 每周一：运行覆盖率测试
- 每周三：代码审查
- 每周五：进度报告

---

## 🚀 开始执行

### 第一步：创建测试骨架

```bash
# 为所有0%模块创建测试文件
touch tests/test_metrics/test_positions.py
touch tests/test_metrics/test_transactions.py
touch tests/test_metrics/test_round_trips.py
touch tests/test_metrics/test_timing.py
touch tests/test_metrics/test_consecutive.py
touch tests/test_metrics/test_perf_stats.py
touch tests/test_metrics/test_bayesian.py
touch tests/test_metrics/test_perf_attrib.py
```

### 第二步：优先完成P0模块

```bash
# 从ratios.py开始
pytest tests/test_metrics/test_ratios_complete.py -v --cov=fincore.metrics.ratios
```

---

## 💡 注意事项

### 1. 依赖管理

- bayesian.py需要pymc (可选依赖)
- 确保测试在没有可选依赖时也能运行

### 2. 测试数据

- 使用conftest.py中的fixtures
- 保持测试数据一致性
- 避免硬编码值

### 3. 性能考虑

- 长时间运行的测试标记为@pytest.mark.slow
- 使用@pytest.mark.skipif跳过慢测试

---

## 📞 支持

如有问题，参考：
- 项目上下文: `_bmad-output/project-context.md`
- 测试指南: `CLAUDE.md`
- pytest文档: https://docs.pytest.org/

---

**计划创建日期**: 2026-03-09  
**预计完成时间**: 12周后  
**负责人**: 开发团队  
**状态**: ⏳ 计划中
