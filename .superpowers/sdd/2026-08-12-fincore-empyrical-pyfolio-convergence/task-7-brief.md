### Task 7: 接入统一输入验证并修正 AnalysisContext 契约

**Files:**
- Create: `fincore/contracts/validation.py`
- Modify: `fincore/validation.py`
- Modify: `fincore/exceptions.py`
- Modify: `fincore/core/context.py`
- Modify: `fincore/report/compute.py`
- Create: `fincore/report/artifacts.py`
- Modify: `fincore/empyrical.py`
- Create: `tests/contracts/test_returns_schema.py`
- Create: `tests/contracts/test_portfolio_schema.py`
- Create: `tests/test_core/test_context_cache_contract.py`
- Create: `tests/test_core/test_context_plot_contract.py`
- Create: `tests/test_core/test_context_export_contract.py`

**Step 1: 写缓存陈旧与输出契约失败测试**

```python
def test_context_takes_an_immutable_snapshot(returns) -> None:
    ctx = AnalysisContext(returns)
    before = ctx.sharpe_ratio
    returns.iloc[:] = 0.0
    assert ctx.sharpe_ratio == before


def test_replace_data_invalidates_all_cached_metrics(returns) -> None:
    ctx = AnalysisContext(returns)
    before = ctx.sharpe_ratio
    ctx.replace_data(returns=returns * -1)
    assert ctx.sharpe_ratio != before


def test_to_json_writes_when_path_is_given(returns, tmp_path) -> None:
    target = tmp_path / "metrics.json"
    payload = AnalysisContext(returns).to_json(path=target)
    assert target.read_text(encoding="utf-8") == payload


def test_plot_returns_artifacts_not_backend(returns) -> None:
    result = AnalysisContext(returns).plot(backend="matplotlib")
    assert result.backend == "matplotlib"
    assert result.figures
```

**Step 2: 验证当前失败**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' tests/contracts tests/test_core/test_context_cache_contract.py \
  tests/test_core/test_context_plot_contract.py \
  tests/test_core/test_context_export_contract.py -q --maxfail=0
```

**Step 3: 统一公共边界**

- `MetricSpec`/workflow spec 必须携带 validation profile。legacy Empyrical/Pyfolio façade 继续镜像上游对 NaN/Inf、empty、对齐和异常的行为；不能先经过增强 schema 而改变 oracle 结果。
- `enhanced`/`context` profile 的 returns/positions/transactions/factors/market-data 五类 schema 检查 index、排序、重复、时区、必需列、数值有限性、重叠区间和 cash 约定。
- validator 同时绑定 positional 和 keyword 参数；内部 kernel 不重复做昂贵验证。
- `AnalysisContext` 构造时 defensive copy，提供原子 `replace_data()` 并自动 invalidate。
- alpha/beta 缓存同一个 `alpha_beta` 结果。
- `plot()` 返回 `ReportArtifacts`；`to_json(path=None)` 同时支持返回字符串和显式写文件。
- positions/transactions 若不进入 context 输出，则从构造参数移除并走 deprecation；推荐在 `perf_stats()` 中纳入已定义的 leverage/turnover。
- `report/compute.py` 消费 context/registry，不再维护第二套指标名称和公式。

**Step 4: 运行 schema/context/report 门禁**

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' tests/contracts tests/test_core tests/test_report \
  -q --tb=short --maxfail=0
```

Expected: 0 failures；同一 validation profile 内的增强 flat API、class 和 context 对坏输入得到同一类领域异常；legacy façade 则以 frozen oracle 结果/异常为准，不强求与增强层一致。

**Step 5: Commit**

```bash
git add fincore/contracts fincore/validation.py fincore/exceptions.py \
  fincore/core/context.py fincore/report/compute.py fincore/report/artifacts.py \
  fincore/empyrical.py \
  tests/contracts tests/test_core
git commit -m "refactor: unify input and analysis context contracts"
```

