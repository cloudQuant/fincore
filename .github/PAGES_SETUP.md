# GitHub Pages 配置说明

## 一键启用

1. 打开仓库 **Settings** → **Pages**
2. 在 **Build and deployment** 下：
   - **Source** 选择 **GitHub Actions**
3. 保存后，推送到 `main` 或 `master` 分支会触发 docs 构建和部署

## 文档更新触发条件

当以下文件变更时会自动重建文档：
- `mkdocs_docs/**`
- `mkdocs.yml`
- `fincore/**`

也可在 **Actions** 页面手动运行 **Deploy Docs** 工作流。

## 文档地址

部署成功后访问：`https://<owner>.github.io/<repo>/`

例如：`https://cloudquant.github.io/fincore/`
