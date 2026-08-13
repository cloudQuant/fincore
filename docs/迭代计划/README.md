# 迭代计划快照索引（Snapshot Index）

本目录（`docs/迭代计划/`）保存历史上各迭代的计划、报告与完成总结。
**全部历史目录均属历史快照（historical），不是当前发布证据（release evidence）。**

当前发布证据只来自以下两类位置：

- **机器生成的当前质量快照**：`docs/quality/current-baseline.{json,md}`；
- **冻结的兼容清单 + 可执行门禁**：`tests/compat/fixtures/` 与 `tests/compat/`；
- **公开站点源**：`mkdocs_docs/`（内部计划/证据见 `docs/plans/`）。

## 快照索引规则

1. 历史目录正文**不批量改写**：旧报告里出现的“1.0 完成”“100% 覆盖率”等表述
   只在当时语境下有参考价值，一律视为未复核的历史声明。
2. 任何新生成或实质更新的快照必须记录 `snapshot_commit`、`verified_at` 与原始命令。
3. 历史快照如需引用，必须同时引用其对应的当前证据位置，不得单独作为结论来源。

## 历史目录清单（39 个，全部为 historical）

| 编号 | 目录 | 状态 |
|---|---|---|
| 0001 | `0001_添加empyrical然后重构` | historical |
| 0002 | `0002_把常量从子项目中移动到主项目` | historical |
| 0003 | `0003_把empyrical重构成一个类的形式` | historical |
| 0004 | `0004_修复重构后的bugs` | historical |
| 0005 | `0005_修复pyfolio重构后的bug` | historical |
| 0006 | `0006_增加注释` | historical |
| 0007 | `0007_分析bug和优化点` | historical |
| 0008 | `0008_继续分析bug和优化` | historical |
| 0009 | `0009-继续分析bug和优化` | historical |
| 0010 | `0010-继续分析bug和优化` | historical |
| 0011 | `0011_继续分析bug和改进优化` | historical |
| 0012 | `0012-继续分析bug和改进优化` | historical |
| 0013 | `0013-继续分析bug和改进优化` | historical |
| 0014 | `0014-继续分析bug和改进优化` | historical |
| 0015 | `0015-继续分析bug和改进优化` | historical |
| 0016 | `0016-继续分析bug和改进优化` | historical |
| 0017 | `0017-继续分析bug和改进优化` | historical |
| 0018 | `0018-改进优化` | historical |
| 0019 | `0019-重构迭代` | historical |
| 0020 | `0020-重构优化` | historical |
| 0021 | `0021-重构优化` | historical |
| 0022 | `0022-重构优化` | historical |
| 0023 | `0023-准备上线工作` | historical |
| 0024 | `0024-优化格式` | historical |
| 0025 | `0025-校正指标计算方式` | historical |
| 0026 | `0026-优化生成的报告` | historical |
| 0027 | `0027-迈向世界一流绩效分析框架` | historical |
| 0028 | `0028-代码质量优化迭代` | historical |
| 0029 | `0029-完善hooks模块` | historical |
| 0030 | `0030-类型系统改进` | historical |
| 0031 | `0031-代码优化与清理` | historical |
| 0032 | `0032-迭代完善计划` | historical |
| 0033 | `0033-持续优化与产品化` | historical |
| 0034 | `0034-公开API稳定性与质量门禁` | historical |
| 0035 | `0035-语义正确性与测试有效性深化` | historical |
| 0036 | `0036-全面质量提升与Bug修复` | historical |
| 0037 | `0037-不断修复bug并完善项目` | historical |
| 0038 | `0038-持续不间断运行` | historical |
| 0040 | `0040-渐进式质量提升与产品化` | historical（含旧“1.0 发布准备”材料，非发布证据） |

## 当前快照（current snapshots）

| 快照 | 路径 | snapshot_commit | verified_at | 原始命令 |
|---|---|---|---|---|
| 质量基线（机器生成） | `docs/quality/current-baseline.json`、`docs/quality/current-baseline.md` | 基线内记录 source commit `53af9215`（Task 1 生成时）；文件当前随提交 `d2441f3` | 2026-08-12T16:13Z（生成时间，见基线 `generated_at`） | `python scripts/collect_quality_baseline.py`（最终验收运行会重新生成） |
| empyrical 0.6.0 冻结清单 | `tests/compat/fixtures/empyrical-0.6.0-api.json` | 上游 commit `74655e974ed2935563820c548c339731f1fe0621`；本仓库 `d2441f3` | 2026-08-13（Task 13 复核） | `python scripts/generate_compat_manifest.py --empyrical-root <root> --output tests/compat/fixtures` |
| pyfolio 0.9.6 冻结清单 | `tests/compat/fixtures/pyfolio-0.9.6-api.json` | 上游 commit `724bbd7dbed9a88bb47e1057f2ca29b3409d8e7a`；本仓库 `d2441f3` | 2026-08-13（Task 13 复核） | `python scripts/generate_compat_manifest.py --pyfolio-root <root> --output tests/compat/fixtures` |
| flat API 迁移映射 | `tests/compat/fixtures/fincore-flat-api-migrations.json` | 本仓库 `d2441f3` | 2026-08-13（Task 13 复核） | 同上生成器输出 |

注：冻结清单中的 `oracle_verification.reviewed` 当前为 `false`、`status=not_run`；
oracle 证据在人工复核前不作为 C 级结论来源。清单完整性由
`tests/compat/test_manifest_integrity.py` 与 CI job `compat` 强制执行。

## 相关位置

- 内部计划/证据：`docs/plans/`
- 公开站点源：`mkdocs_docs/`
- 发布候选清单：`docs/quality/release-candidate-checklist.md`
- 兼容矩阵（公开）：`mkdocs_docs/development/compatibility.md`
