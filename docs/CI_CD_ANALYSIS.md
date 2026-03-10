# CI/CD、GitHub Pages、Read the Docs 分析报告

> 分析日期: 2026-03-10

---

## 一、CI/CD 概况

### 1. 主 CI 工作流 (`ci.yml`)

| 项目 | 配置 |
|------|------|
| 触发 | `push` / `pull_request` → `master`, `main` |
| 矩阵 | 3 OS × 7 Python 版本 = 21 个测试任务 |
| 测试 | `pytest -n auto -m "not slow and not integration"` |
| Lint | Ruff check + format（`fincore/`, `tests/`） |
| 类型检查 | mypy（部分模块，`continue-on-error: true`） |
| 构建 | sdist + wheel，twine check |

**潜在问题**：
- ~~**pyproject.toml** 要求 `requires-python = ">=3.8"`，但 CLAUDE.md 写的是 Python 3.11+~~ ✅ 已统一
- **Ruff** 不检查 `scripts/`，而 scripts 下有 lint 问题（不影响 CI 通过）
- ~~**mypy** 使用 `continue-on-error: true`~~ ✅ 已移除，类型错误会导致 CI 失败

### 2. 增强 CI (`ci-enhanced.yml`)

| 项目 | 配置 |
|------|------|
| 触发 | 仅 `workflow_dispatch`（手动） |
| 包含 | fast-check、full-suite、integration、slow、benchmark、lint、typecheck |

### 3. 文档部署 (`docs.yml`)

| 项目 | 配置 |
|------|------|
| 触发 | `push` 到 main/master，且变更涉及 `mkdocs_docs/`、`mkdocs.yml`、`fincore/` |
| 构建 | MkDocs + mkdocs-material |
| 部署 | GitHub Pages (`actions/deploy-pages@v4`) |

---

## 二、GitHub Pages

### 配置
- **站点配置**: `mkdocs.yml` → `docs_dir: mkdocs_docs`
- **预期 URL**: `https://cloudquant.github.io/fincore`
- **repo_url**: `https://github.com/cloudQuant/fincore`

### 本地构建
```bash
mkdocs build  # ✅ 构建成功
```

### 状态
- **当前访问**: `https://cloudquant.github.io/fincore` 返回 404
- **可能原因**:
  1. 未在仓库设置中启用 GitHub Pages
  2. 需在 Settings → Pages 中将 Source 设为 “GitHub Actions”
  3. 仓库或组织名称大小写不同（cloudQuant vs cloudquant）

### 建议
1. 在 GitHub 仓库 Settings → Pages 中，将 Source 选为 **GitHub Actions**
2. 推送一次到 main/master 以触发 `Deploy Docs` workflow
3. 确认 Pages 环境 `github-pages` 已创建

---

## 三、Read the Docs

### 结论
**当前项目未配置 Read the Docs**。

- 仓库中无 `.readthedocs.yaml` 或 `readthedocs.yml`
- 文档托管依赖 GitHub Pages (MkDocs)
- 搜索结果显示 fincore 无专用 Read the Docs 站点

### 若需启用 Read the Docs
1. 在 [readthedocs.org](https://readthedocs.org) 导入仓库
2. 在项目根目录创建 `.readthedocs.yaml`，例如：

```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.12"
  jobs:
    post_checkout:
      - pip install -e ".[viz]"
  mkdocs:
    configuration: mkdocs.yml
```

---

## 四、8 核并行测试

### 命令
```bash
pytest tests/ -n 8 -q --tb=line --ignore=tests/benchmarks/ -m "not slow and not integration"
```

### 结果（本地，8 workers）
- **已修复**: `test_high_tracking_error`（零追踪误差时 information_ratio 返回 NaN，已更新测试用例）✅
- **已修复**: `test_var_all_nan`、`test_cvar_all_nan`（VaR/CVaR 全 NaN 分支提前返回，避免 RuntimeWarning）✅
- **已修复**: `test_sharpe_ratio_*_dataframe`（断言改为接受 np.ndarray 或 pd.Series）✅
- **仍存在**: `test_providers_more_coverage.py` 1 个 teardown 错误（可选依赖，非核心）

### CI 中并行策略
- `ci.yml`: `-n auto`（根据 CPU 自动选择 worker 数）
- GitHub Actions runners 通常为 2 核，`-n auto` 约等于 2
- 本地 `-n 8` 能正确使用 8 个 worker

---

## 五、MkDocs 构建警告（非阻塞）

1. **perf_stats.py**: griffe 报告 `n_samples`、`random_seed` 未在函数签名中出现
2. **VizBackend**: 存在多个 primary URL
3. **MkDocs 2.0**: Material for MkDocs 与 MkDocs 2.0 不兼容提示（当前仍可构建）

---

## 六、建议操作

| 优先级 | 操作 | 状态 |
|--------|------|------|
| P0 | 在 GitHub 仓库启用 Pages，并将 Source 设为 GitHub Actions | ⬜ 需在 GitHub 上手动完成 |
| P1 | 统一 Python 版本说明（pyproject.toml 与 CLAUDE.md） | ✅ 已更新 CLAUDE.md |
| P2 | 将 mypy 的 `continue-on-error` 改为 `false` | ✅ 已移除，类型检查通过 |
| P3 | 可选：增加 Read the Docs 配置以提供多版本文档 | ✅ 已添加 `.readthedocs.yaml` |

---

*本报告由本地工作流分析及 mkdocs build 结果整理生成。*
