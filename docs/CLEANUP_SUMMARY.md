# 主目录清理说明

## 已完成的清理

### 0. Markdown 文件整理（第二次清理）
以下文件已从根目录移至 `docs/`：
- `BENCHMARKS.md` → `docs/BENCHMARKS.md`
- `CODE_QUALITY_IMPROVEMENTS.md` → `docs/CODE_QUALITY_IMPROVEMENTS.md`
- `CODE_QUALITY_REVIEW.md` → `docs/CODE_QUALITY_REVIEW.md`
- `CODE_REVIEW_REPORT.md` → `docs/CODE_REVIEW_REPORT.md`
- `IMPROVEMENTS.md` → `docs/IMPROVEMENTS.md`
- `已实现函数索引.md` → `docs/已实现函数索引.md`

根目录仅保留：`README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CLAUDE.md`

### 1. .gitignore 更新
- **AI/编辑器配置**：`.agents/`, `.claude/`, `.cursor/`, `.opencode/`, `.windsurf/`, `_bmad/`（保留 `_bmad-output`）
- **构建产物**：`site/`, `logs/`, `coverage.json`, `report.html`, `viz_returns.html`
- **缓存**：`.mypy_cache/`, `.ruff_cache/`, `.benchmarks/`
- **示例输出**：根目录 PNG 文件及 `examples/output/`

### 2. 示例脚本输出目录
- 所有 `examples/*.py` 中 `plt.savefig()` 已改为保存到 `examples/output/`
- 运行示例不会在项目根目录生成文件

### 3. 已删除的根目录文件
- `backtesting_metrics.png`, `bootstrap_analysis.png`, `market_timing_analysis.png`
- `monte_carlo_simulation.png`, `performance_attribution.png`, `portfolio_optimization.png`
- `rolling_metrics.png`, `stress_testing.png`, `risk_models.png`, `workflow_analysis.png`
- `positions_analysis.png`, `viz_comprehensive.png`, `custom_optimization.png`
- `coverage.json`, `report.html`, `viz_returns.html`

### 4. Git 历史清理
使用 `git filter-repo` 从历史中移除了以下路径（保留 `_bmad-output`）：
- `.agents/`, `.claude/`, `.cursor/`, `.windsurf/`, `.opencode/`
- `_bmad/`, `site/`, `logs/`, `.benchmarks/`
- 根目录生成的 PNG/HTML 文件（backtesting_metrics.png, coverage.json, report.html 等）

**注意**：
- 历史重写会改变所有 commit hash
- `git filter-repo` 会移除 `origin` 远程，需手动重新添加：`git remote add origin https://github.com/cloudQuant/fincore.git`
- 若已推送到远程，需要 `git push --force`。请确保团队其他成员知晓此次变更
