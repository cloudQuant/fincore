# 文档配置文件位置分析

## 结论：文档配置需保留在项目根目录

### mkdocs.yml

- **工具要求**：MkDocs 默认在项目根目录查找 `mkdocs.yml`，`mkdocs build` / `mkdocs gh-deploy` 均依赖此约定
- **引用关系**：
  - `.github/workflows/docs.yml` 的 `paths` 监听 `mkdocs.yml` 变更
  - `.readthedocs.yaml` 通过 `configuration: mkdocs.yml` 引用
- **若移到 `.github/`**：需修改 `docs.yml` 为 `mkdocs build -f .github/mkdocs.yml`，并同步更新 Read the Docs 配置，且不符合常见项目结构
- **建议**：保留在根目录

### .readthedocs.yaml

- **工具要求**：Read the Docs 要求配置必须在**仓库根目录**，不支持放在子目录
- 官方说明：<https://docs.readthedocs.io/en/stable/config-file/v2.html>
- **建议**：必须保留在根目录

## 可放在 .github 的配置

`.github/` 适合放与 GitHub 相关的配置，例如：

- `workflows/`：CI/CD 工作流
- `PAGES_SETUP.md`：Pages 说明
- `CODEOWNERS`、`dependabot.yml` 等

文档构建配置（mkdocs、readthedocs）因工具限制，应继续放在根目录。
