# 0042-R2 技术验收记录

## 结论

`D-TECH: PASS`。冻结 acceptance runner 已在同一份新鲜证据根中验证全部十项必需技术门禁；测试对象是下列 **tested parent**，而不是本 evidence-only child 本身。

| 项目 | 值 |
| --- | --- |
| tested parent SHA | `b1dd88b7ca34756043539ff65d8e198989c2bc3e` |
| tested parent tree | `8ad871d540cd82a9856773a2da827bde3e4786fd` |
| final verdict 时间 | `2026-09-01T10:12:21.859533+00:00` |
| final evidence SHA-256 | `93af2dd23c8687c6800a81b439e2d07967ae413f9dc16cedcedb58195ccac23c` |
| 远端构建/矩阵 | [GitHub Actions run 33493997878](https://github.com/cloudQuant/fincore/actions/runs/33493997878) |

## 不可变输入

| 输入 | 身份 |
| --- | --- |
| 唯一候选 wheel | `fincore-0.5.0.dev0-py3-none-any.whl`; SHA-256 `0249952b38c74e76be00ff873cba201da2585817ed341a6047080315e11432b8` |
| 唯一候选 sdist | `fincore-0.5.0.dev0.tar.gz`; SHA-256 `ca91fd06fa3046afd3c77bdffa11c255aba01437e2ce0c2de043f154eed7d670` |
| D0 v23 archive | host commit `84cecf71dfdf1d9f9c0025c51b1b3eacffd20e2b`; archive SHA-256 `e83465765d038f6a4b0ab51aa632df34981b79a968d80d5b8218305b49d9ead6`; bundle digest `c619685559664eafc531bf76e9f8f9b23857ee05a345b62d65ae054366c0db01` |
| frozen tooling | commit `cf45d01d922b77283135fc82678635a662f22ddb`; tree `02a8a922b25804292f3768e331f527bb2fbde169`; runner blob SHA-256 `566ef5f3e7f2730478c64abf9df1877aa067bf2a9b514fb01f976c0aedf89dab` |

远端 Linux、macOS 与 Windows matrix cell 都绑定同一 tested parent tree、同一 wheel SHA-256、D0 bundle 和 frozen tooling，并由 matrix-aggregate 再次验证为 `PASS`。

## 已验证门禁

| 门禁 | 结果 |
| --- | --- |
| tests（slow、serial、offline integration；benchmark 由 performance 接管） | PASS |
| static（Ruff format/check、mypy、MkDocs strict） | PASS |
| package（唯一 wheel/sdist、sdist source equivalence、legacy-zero） | PASS |
| quality（fresh coverage、95% changed-lines、critical branch） | PASS |
| parity（全部 capability families，source/wheel equality） | PASS |
| architecture（LOC/duplicate reduction、legacy-zero、no cycles） | PASS |
| performance（metrics、rolling、transactions、factor、risk、report；2 warmups × 5 repeats） | PASS |
| report（Chromium、HTML、PDF、XLSX、Plotly、Bokeh） | PASS |
| installed（7 capability profiles、all providers、minimum/latest lanes） | PASS |
| matrix-aggregate（Linux/macOS/Windows × D0 support window） | PASS |

架构快照显示 legacy module/export 为零、内部 import cycle 为零；性能门禁记录 14 个 workload 的中位时间改善且无回归差异。质量门禁的总体 branch coverage 为 `75.9027949542419%`，changed-line 与 critical-module 检查均通过。

## 发布边界

本记录只确认 `D-TECH`，不确认 `D-RELEASE`。它**没有**授权或执行 master 合并、tag、PyPI 发布、GitHub Release、部署或外部公告。此提交仅为 evidence-only child；其唯一允许变更为本文件与 [0042-r2-evidence-digests.json](0042-r2-evidence-digests.json)。

外部证据的路径、artifact 摘要和每项 gate evidence 摘要记录在该 JSON 索引中。
