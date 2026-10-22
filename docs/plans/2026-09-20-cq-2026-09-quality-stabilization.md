# CQ-2026-09 质量稳态迭代计划

## 目标与边界

| 字段 | 结论 |
| --- | --- |
| 候选基线 | `master@2354bb5` 的隔离工作树；不使用原工作区的未提交治理改动。 |
| 目标 | 修复失效的本地质量门、隔离测试构建输入、建立 0.5 根命名空间的静态契约检查，并让发布流水线验证这些证据。 |
| 不变性 | 不修改 `fincore/` 运行时代码、数值结果、异常语义、公开 API、依赖解析或性能目标。 |
| 发布目标 | `0.5.1`（补丁版本）；不重新解释 0.5.0 的破坏性 API 决策。 |
| 非目标 | 兼容层恢复、旧 0.4 API fixture 重写、清理原工作区的忽略工件、改变用户拥有的未提交文件。 |

## 已验证的发现

| ID | 发现 | 状态 | 处置 |
| --- | --- | --- | --- |
| CQ-01 | pre-commit 的 MyPy 参数仍指向已删除目录，hook 退出而不是检查当前包。 | `PASS`（独立复验） | 改为一次 `mypy --ignore-missing-imports fincore`，关闭文件名自动追加。 |
| CQ-02 | 打包合同测试复制源码时会纳入仓库内 `.pytest_tmp`，可能造成递归复制/竞态。 | `PASS`（独立复验） | 两个 staging ignore 列表加入该目录，并有断言。 |
| CQ-03 | `check_quality_snapshot.py --allow-snapshot-output-commit` 在干净候选基线通过。 | `PASS` | 把它作为候选证据，不把输出型快照提交误判为陈旧。 |
| CQ-04 | `public-api-0.4.0.dev0.json` 是原子切换前的 schema-v1 历史投影；它与 0.5 源码不同是预期结果。 | `EXPECTED_HISTORICAL` | 不覆盖、不删除；发布门改用 0.5 的 schema-v2 源码/轮子契约。 |
| CQ-05 | schema-v2 扫描根包时遗漏 `__all__` 明确导出的动态 `__version__` 赋值。 | `PASS`（独立复验） | 静态解析器与回归用例已修复；不接触包运行时代码。 |
| CQ-06 | 隔离 Ruff 扫描有 37 个 C901 热点、5 个 F401 和 6 个 E402。 | `DEFERRED` | 生产源码重构必须先取得缺失的项目开发规范与逐模块行为基线。 |
| CQ-07 | `AGENTS.md` 引用的 `.joyincode/rules/backend.md`、`frontend.md` 在候选基线不存在。 | `BLOCKED` | 在规则恢复或项目负责人明确 waiver 前，不进行 `fincore/` 源码重构。 |

## 实施编排

| 阶段 | 修改范围 | 验收证据 | 状态 |
| --- | --- | --- | --- |
| Q1：本地类型门 | `.pre-commit-config.yaml` | `pre-commit run mypy --all-files` 为 0 | `PASS` |
| Q2：构建输入隔离 | 两个 `tests/packaging/` 文件 | 在仓库外 `--basetemp` 的 20 个打包合同用例通过 | `PASS` |
| Q3：根 API 静态契约 | `scripts/snapshot_public_api.py`、质量/契约测试 | 当前源根和 wheel 根快照均可生成，源/轮一致 | `PASS`（本地） |
| Q4：CI/CD 绑定 | `ci.yml`、`publish.yml` 的最小必要门 | tag 版本、静态契约、构建候选和已有 release 证据均 fail-closed | `PASS`（本地；远端待运行） |
| Q5：在线文档与版本 | `pyproject.toml`、CHANGELOG、MkDocs、README、release note | `0.5.1` 一致且 `mkdocs build --strict` 通过 | `PASS`（本地；部署待运行） |
| Q6：发布验收 | 候选 commit、tag、GitHub CI/Docs/Publish 运行、PyPI 项目页 | 所有本地与远端门 `PASS`；否则明确 `NO-GO` | `PENDING` |

## 不变性与验收矩阵

| 用例 | 命令/证据 | 通过条件 |
| --- | --- | --- |
| AC-01 | `conda run -n base … python -m pre_commit run mypy --all-files` | hook 实际检查 `fincore` 且 exit 0。 |
| AC-02 | `pytest tests/packaging/test_release_consistency.py tests/packaging/test_wheel_contents.py --basetemp <outside-repo>` | staging source 不含 `.pytest_tmp`；全部节点通过。 |
| AC-03 | `snapshot_public_api.py --source-root . --surface fincore` | 仅语法扫描即可产出 schema-v2 根契约；不导入可选依赖。 |
| AC-04 | 构建后的 `--source-root . --wheel <candidate> --surface fincore --compare` | 源码与候选 wheel 的根公开契约逐字一致。 |
| AC-05 | `pytest -m 'not integration_online' --basetemp <outside-repo>` | 非在线回归完整通过；网络型检查单独报告。 |
| AC-06 | `python -m mkdocs build --strict`、release consistency、twine、wheel-consumer | 文档与可发布工件一致。 |
| AC-07 | tag 的 GitHub CI、Docs 与 Publish-to-PyPI 运行；PyPI `0.5.1` 元数据 | 远端证据全部成功后才宣布发布。 |

## 风险控制与放行条件

1. `public-api-0.4.0.dev0.json` 仅保留为历史发现输入；不能用“更新 fixture”掩盖 0.5 公共契约变化。
2. Q3 只允许测试工具和合同文件变化；任何 `fincore/` 源码差异立即停止并重新评审。
3. Q4 不复制已有的完整 CI；只增加当前缺失且能从候选工件验证的 fail-closed 门。
4. Q6 之前必须有干净候选提交、精确 tag/版本一致性、完整 CI 与文档构建成功。PyPI trusted publishing 失败时状态为 `NO-GO`，不得声称已发布。
5. CQ-06/CQ-07 未关闭前，本迭代不把“降低复杂度”作为已完成成果；它们将作为后续、规则齐全且逐模块基线完备的源码重构迭代。
