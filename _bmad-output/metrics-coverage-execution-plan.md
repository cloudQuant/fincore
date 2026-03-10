# Metrics覆盖率提升 - Week 1-2 执行方案

**创建日期**: 2026-03-09
**目标完成日期**: 2026-03-23 (Week 2 Day 7)
**目标模块**: ratios.py (P0), risk.py (P0)

---

## 🎯 核心目标

| 模块 | 当前覆盖率 | 目标覆盖率 | 新增测试 | 工作量 |
|------|-----------|-----------|---------|--------|
| ratios.py | 59% | 85%+ | ~30个 | 2-3天 |
| risk.py | 18% | 85%+ | ~25个 | 2-3天 |

---

## ✅ 核心决策矩阵

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 执行策略 | 并行开发 | 缩短总时间，风险可控 (仅2个模块) |
| 文件组织 | 新建`*_complete.py` | 保持现有测试，清晰标识目标 |
| Fixture策略 | 三级复用 (全局→模块→函数) | 最大化复用，避免重复 |
| 优先级标记 | P0/P1/P2/P3标准 | 符合`conftest.py`定义 |
| 测试数量 | ratios ~30个, risk ~25个 | 基于未覆盖功能分析 |
| 时间表 | 7个工作日 | 保守估算，包含buffer |

---

## 📅 详细时间表

### Week 1 (Day 1-5)

#### Day 1-2: 测试设计
| 人员 | 任务 | 时间 |
|------|------|------|
| Mary (BA) | 分析ratios.py/risk.py未覆盖功能 | 4h |
| Winston (Arch) | 定义fixtures策略 | 2h |
| Amelia (Dev) | 设计测试架构 | 4h |
| Quinn (QA) | 确定P0/P1/P2标准 | 2h |

**交付物**:
- [ ] 未覆盖行分析报告 (Mary)
- [ ] `conftest_metrics.py`设计稿 (Winston)
- [ ] 测试骨架文件 (Amelia)
- [ ] 优先级标准文档 (Quinn)

#### Day 3-4: 测试实现 (并行)

| Amelia (ratios.py) | Winston (risk.py) |
|-------------------|-------------------|
| • stability_of_timeseries | • tail_ratio |
| • capture ratios | • value_at_risk完整 |
| • up/down_capture | • conditional_value_at_risk |
| • 边缘情况覆盖 | • custom_value_at_risk |

**工作量**: Amelia ~25个测试 | Winston ~20个测试

#### Day 5: 代码审查
| 任务 | 负责人 | 时间 |
|------|--------|------|
| 审查Amelia的ratios测试 | Quinn (QA) | 3h |
| 审查Quinn的审查意见 | Winston (Arch) | 1h |
| 确认测试覆盖需求 | Mary (BA) | 2h |

**交付物**:
- [ ] 代码审查报告 (Quinn)
- [ ] 审查意见修复 (Amelia)
- [ ] 准备合并的feature分支

### Week 2 (Day 6-7)

| 时间 | 任务 | 状态 |
|------|------|------|
| Day 6 AM | 合并两个feature分支 | 🔄 |
| Day 6 PM | 运行集成测试 | 🧪 |
| Day 7 AM | 修复集成问题 | 🔧 |
| Day 7 PM | 达到85%覆盖率验证 | ✅ |

**每日Checklist**:
- [ ] 运行`pytest -m p0` (关键测试)
- [ ] 运行`pytest --cov` (覆盖率检查)
- [ ] 运行`ruff check` (代码质量)
- [ ] 更新进度文档

---

## 👥 团队角色职责

| 角色 | 姓名 | 主要职责 |
|------|------|---------|
| BA | Mary | 需求分析，覆盖率基线确认 |
| Arch | Winston | fixture设计，架构决策，risk模块开发 |
| Dev | Amelia | ratios模块测试开发 (~30个测试) |
| QA | Quinn | 代码审查，测试质量保证，覆盖率验证 |
| SM | Bob | 进度协调，风险监控，每日站会 |

---

## 📁 文件组织结构

```
tests/
├── conftest.py                    # ✅ 已存在 (全局fixtures)
│
├── test_metrics/                   # Metrics模块测试目录
│   ├── test_ratios_complete.py      # 🆕 ratios.py完整测试
│   ├── test_risk_complete.py        # 🆕 risk.py完整测试
│   │
│   ├── conftest_metrics.py         # 🆕 模块级fixtures (新建)
│   │   ├── returns_with_benchmark()
│   │   ├── fat_tailed_returns()
│   │   └── volatility_clustering_returns()
│   │
│   ├── test_ratios_additional.py    # ✅ 已存在 (59%覆盖)
│   ├── test_risk_more_coverage.py   # ✅ 已存在
│   └── ... (其他现有文件保持不变)
│
└── coverage_reports/               # 🆕 覆盖率报告目录
    ├── ratios_baseline.html        # 基线覆盖率 (59%)
    ├── ratios_target.html         # 目标覆盖率 (85%+)
    └── risk_baseline.html         # 基线覆盖率 (18%)
```

**为什么创建新的`*_complete.py`文件？**
1. 保持现有测试不变，避免破坏已有覆盖率
2. 清晰标识"完整覆盖"目标
3. 方便对比基线与目标
4. 便于独立运行测试: `pytest test_metrics/test_ratios_complete.py`

---

## 🔧 Fixtures复用策略

### Level 1: 全局Fixtures (`conftest.py` - 已存在)
```python
@pytest.fixture
def small_returns():
    """Generate small returns dataset for quick tests (252 points = 1 year)."""
    np.random.seed(42)
    return pd.Series(np.random.randn(252) * 0.01, index=pd.bdate_range("2020-01-01", periods=252))

@pytest.fixture
def empty_returns():
    """Empty returns series for edge case testing."""
    return pd.Series([], dtype=float)

@pytest.fixture
def all_nan_returns():
    """All NaN returns for edge case testing."""
    return pd.Series([np.nan] * 100)
```

### Level 2: 模块专用Fixtures (`test_metrics/conftest_metrics.py` - 新建)
```python
@pytest.fixture
def returns_with_benchmark():
    """带benchmark的returns (用于capture ratios)"""
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01, index=pd.bdate_range("2020-01-01", periods=252))
    benchmark = returns * 0.5
    return returns, benchmark

@pytest.fixture
def fat_tailed_returns():
    """肥尾分布returns (用于tail_ratio)"""
    np.random.seed(99)
    return pd.Series(np.random.standard_t(2, 1000) * 0.01)

@pytest.fixture
def volatility_clustering_returns():
    """波动率聚集returns (用于VaR/CVaR)"""
    vol = np.concatenate([np.random.randn(50) * v for v in [0.01, 0.02, 0.01, 0.02]])
    return pd.Series(vol)
```

### Level 3: 函数级专用Fixtures (在测试文件中定义)
```python
class TestTailRatio:
    @pytest.fixture
    def normal_returns(self):
        """正态分布 (tail_ratio测试专用)"""
        return pd.Series(np.random.randn(1000))

    @pytest.fixture
    def heavy_tail_returns(self):
        """肥尾分布 (tail_ratio测试专用)"""
        return pd.Series(np.random.standard_t(3, 1000))
```

**复用规则**:
1. **优先级**: 全局 > 模块级 > 函数级
2. **命名规范**:
   - 全局: `small_returns`, `medium_returns`
   - 模块: `{module}_returns` (ratios_returns, risk_returns)
   - 函数: `{testcase}_data`
3. **避免重复**: 新fixture先检查`conftest.py`是否已存在

---

## 🏷️ 测试优先级标记标准

### P0 - 关键 (核心业务价值)
```python
@pytest.mark.p0
def test_sharpe_ratio_normal_case():
    """正常情况下的Sharpe Ratio (核心指标)"""
    returns = small_returns()
    result = sharpe_ratio(returns)
    assert np.isfinite(result)
```

**P0判断标准**:
- ✅ 核心金融指标 (sharpe_ratio, max_drawdown, alpha, beta)
- ✅ 用户最常使用的功能
- ✅ 风险管理关键计算 (VaR, CVaR)
- ✅ 影响投资决策的指标

### P1 - 重要 (高频使用)
```python
@pytest.mark.p1
def test_sharpe_ratio_with_nan():
    """NaN处理 (边缘但常见)"""
    returns = all_nan_returns()
    result = sharpe_ratio(returns)
    assert np.isnan(result)
```

**P1判断标准**:
- ✅ 常见边缘情况 (NaN, 空数据, 零波动)
- ✅ 重要但非核心功能
- ✅ 高频使用场景

### P2 - 次要 (增强覆盖)
```python
@pytest.mark.p2
def test_sterling_ratio_specific_branch():
    """Sterling Ratio特定分支 (提高覆盖率)"""
    # 测试特定代码分支
    pass
```

**P2判断标准**:
- ✅ 罕见边缘情况
- ✅ 提高覆盖率的分支
- ✅ 内部辅助函数

### P3 - 低优先级 (装饰性)
```python
@pytest.mark.p3
def test_deprecated_function():
    """已弃用功能 (装饰性测试)"""
    pass
```

**P3判断标准**:
- ✅ 已弃用功能
- ✅ 装饰性/文档性测试
- ✅ 极罕见边缘情况

**标记应用规则**:
1. 每个测试**必须**有优先级标记
2. 默认P1，重要功能手动标记P0
3. 自动标记通过`conftest.py`的pytest钩子

---

## 🎯 成功标准

- [ ] ratios.py: 59% → 85%+ (新增65行覆盖)
- [ ] risk.py: 18% → 85%+ (新增480行覆盖)
- [ ] 所有P0测试通过 (0失败)
- [ ] 集成测试无冲突
- [ ] ruff check通过 (0 warnings)
- [ ] mypy check通过 (0 errors)

---

## ⚠️ 风险评估与缓解措施

### 🔴 高风险项

#### R1: Fixture冲突导致测试失败
**影响**: 阻塞开发，浪费时间
**概率**: 40%
**缓解**:
- ✅ Day 2: Winston统一fixture命名规范
- ✅ 使用pytest fixture scope隔离
- ✅ 建立fixture注册表 (共享文档)
- 🔄 应急: 临时命名空间 + 后续重构

#### R2: 覆盖率目标85%无法达成
**影响**: 延期，影响后续模块
**概率**: 30%
**缓解**:
- ✅ Day 1: Mary精确定义可覆盖行
- ✅ 使用`pytest-cov --cov-report=term-missing`追踪缺失行
- ✅ 如果目标>90%难达成，调整到80%+
- 🔄 应急: 扩展测试用例范围

#### R3: 代码合并冲突 (ratios与risk共享utils)
**影响**: 耗时解决冲突，破坏已有测试
**概率**: 25%
**缓解**:
- ✅ 使用feature/分支策略 (`feature/ratios-coverage`, `feature/risk-coverage`)
- ✅ Day 5: 提前同步main分支
- ✅ Winston审查共享代码修改
- 🔄 应急: 暂时隔离冲突代码

### 🟡 中风险项

#### R4: 性能测试超时
**影响**: CI/CD pipeline阻塞
**概率**: 50%
**缓解**:
- ✅ 标记慢测试 `@pytest.mark.slow`
- ✅ 使用pytest-xdist跳过慢测试 (`pytest -m "not slow"`)
- ✅ 限制`large_returns`使用
- 🔄 应急: 减少测试数据量

#### R5: NaN处理不一致
**影响**: 测试通过但实际业务失败
**概率**: 35%
**缓解**:
- ✅ 遵循`project-context.md`规则 (使用`utils.nanmean`等)
- ✅ Quinn在代码审查中重点检查NaN处理
- ✅ 对比现有测试的NaN行为
- 🔄 应急: 添加专门的NaN集成测试

#### R6: 测试运行时间过长 (>30min)
**影响**: 开发效率低
**概率**: 40%
**缓解**:
- ✅ 使用pytest-parallel (已有`-n 4`配置)
- ✅ 分离单元测试和集成测试
- ✅ 使用pytest-benchmark验证性能
- 🔄 应急: 按优先级分批测试

### 🟢 低风险项

#### R7: 文档滞后
**影响**: 后续维护困难
**概率**: 60%
**缓解**:
- ✅ 代码审查时检查docstring
- ✅ 使用Google-style docstrings (与现有一致)
- 🔄 应急: 集中补充文档

#### R8: 临时测试代码未清理
**影响**: 技术债务
**概率**: 45%
**缓解**:
- ✅ Day 7: Bob检查临时代码
- ✅ 使用git stash管理临时改动
- 🔄 应急: 创建cleanup任务

---

## 🚦 启动检查清单

### Day 1 上午
- [ ] 创建`feature/ratios-coverage`分支 (Amelia)
- [ ] 创建`feature/risk-coverage`分支 (Winston)
- [ ] 运行基线覆盖率测试 (记录59%/18%)
- [ ] 团队站会确认任务分配

### Day 1 下午
- [ ] Mary分析ratios.py/risk.py未覆盖行
- [ ] Winston设计`conftest_metrics.py` fixtures
- [ ] Quinn确认P0/P1/P2标准
- [ ] Amelia创建`test_ratios_complete.py`骨架

---

## 📊 每日监控指标

| 指标 | 基线 | 警告 | 严重 |
|------|------|------|------|
| 新增测试数/天 | ≥8个 | 5-7个 | <5个 |
| 覆盖率提升%/天 | ≥5% | 3-4% | <3% |
| 测试失败率 | <10% | 10-20% | >20% |
| 合并冲突数 | 0 | 1-2个 | >2个 |

---

## 📞 紧急联系人

- 测试框架问题: Winston
- 覆盖率疑问: Quinn
- 业务逻辑疑问: Mary
- 进度协调: Bob
- 代码质量问题: Quinn

---

## 🔄 应急预案

### 🚨 严重风险触发 (覆盖率<75%或测试失败>20%)
**立即行动**:
1. 召开团队紧急会议 (Bob负责)
2. 重新评估未覆盖功能优先级
3. 调整目标到80% (最低可接受)
4. 延长Week 2到Week 3
5. 申请额外资源

---

## 📚 参考资料

- 项目上下文: `_bmad-output/project-context.md`
- 改进计划: `_bmad-output/metrics-coverage-improvement-plan.md`
- 测试指南: `docs/testing-guide.md`
- 现有fixtures: `tests/conftest.py`
- 现有测试示例: `tests/test_metrics/test_ratios_additional.py`

---

## 🎬 下一步行动

1. Bob: 安排Day 1 9:00 AM启动会议
2. Mary: 准备ratios.py/risk.py未覆盖行分析
3. Winston: 设计`conftest_metrics.py` fixture结构
4. Amelia/Winston: 创建各自feature分支
5. 全员: 阅读project-context.md (lazy loading, NaN处理规则)

---

## 📋 决策快速参考卡

```
┌─────────────────────────────────────────────────────────────────┐
│  Metrics覆盖率提升 - Week 1-2 决策矩阵                    │
├─────────────────────────────────────────────────────────────────┤
│  执行策略:    并行开发 (ratios + risk同时进行)               │
│  文件结构:    test_ratios_complete.py, test_risk_complete.py    │
│  Fixtures:    三级复用 (全局 → 模块 → 函数)                │
│  优先级:      P0/P1/P2/P3 (与conftest.py一致)              │
│  工作量:      ratios ~30个测试, risk ~25个测试               │
│  目标覆盖率:   ratios 85%+, risk 85%+                      │
│  时间周期:    7个工作日 (Week 1-2)                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  关键风险等级                                             │
├─────────────────────────────────────────────────────────────────┤
│  🔴 高风险:  Fixture冲突, 覆盖率不达标, 合并冲突              │
│  🟡 中风险:  性能超时, NaN处理不一致, 运行时间过长             │
│  🟢 低风险:  文档滞后, 临时代码未清理                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  每日检查点                                              │
├─────────────────────────────────────────────────────────────────┤
│  9:00 AM   - 每日站会 (15分钟)                            │
│  12:00 PM  - 覆盖率检查 (pytest --cov)                     │
│  6:00 PM   - 进度更新 (#metrics-coverage频道)                 │
│  周五下午    - 周总结 (Bob主持)                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  应急触发条件                                              │
├─────────────────────────────────────────────────────────────────┤
│  ⚠️ 覆盖率<75%   - 调整目标到80%, 延长Week 3            │
│  ⚠️ 测试失败>20%  - 暂停开发, Quinn优先修复                │
│  ⚠️ 合并冲突>2个  - Winston介入, 临时隔离代码               │
│  ⚠️ 超时>30min   - 标记慢测试, 使用pytest -m "not slow"   │
└─────────────────────────────────────────────────────────────────┘
```

---

**方案创建时间**: 2026-03-09
**目标完成时间**: Week 2 Day 7 (2026-03-23)
**状态**: ⏳ 待团队审批后启动
