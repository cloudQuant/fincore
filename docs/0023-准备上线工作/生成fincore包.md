1. 这个包已经初步准备好了，准备现在实现ci/cd，在github上能够自动测试通过。
2. 形成一个python包，可以通过pip install fincore来安装。

## 已完成

### 1. CI/CD (GitHub Actions)

**`.github/workflows/ci.yml`** — 每次 push/PR 到 master/main 自动触发：
- **测试矩阵**: 3 OS (Ubuntu/macOS/Windows) × 5 Python (3.9–3.13) = 15 组合
- **Lint**: ruff check
- **覆盖率**: Ubuntu + Python 3.11 上生成 coverage 报告
- **构建验证**: `python -m build` + `twine check`
- **并发控制**: 同 ref 自动取消旧 workflow

**`.github/workflows/publish.yml`** — 创建 GitHub Release 时自动发布到 PyPI：
- 使用 OIDC trusted publishing（无需 API token）
- 需要在 PyPI 设置 trusted publisher:
  - Owner: `cloudQuant`, Repository: `fincore`, Workflow: `publish.yml`, Environment: `pypi`

### 2. Python 包 (pip install fincore)

**已验证**:
- `python -m build` → 成功生成 `fincore-0.1.0.tar.gz` + `fincore-0.1.0-py3-none-any.whl`
- `pip install dist/fincore-0.1.0-py3-none-any.whl` → 安装成功，`import fincore` 正常

**修复的问题**:
- 版本号统一为 `0.1.0`（`pyproject.toml` / `setup.py` / `fincore/__init__.py`）
- License 修正为 `Apache-2.0`（与 LICENSE 文件一致，原来写的 MIT）
- `pytest.ini` coverage source 从 `empyrical` 改为 `fincore`
- 新增 `MANIFEST.in` 确保 sdist 包含所有必需文件

### 发布流程

```bash
# 1. 更新版本号（三个文件）
#    pyproject.toml: version = "0.2.0"
#    setup.py:       VERSION = "0.2.0"
#    fincore/__init__.py: __version__ = "0.2.0"

# 2. 提交并打 tag
git add -A && git commit -m "release: v0.2.0"
git tag v0.2.0
git push origin master --tags

# 3. 在 GitHub 创建 Release（选择 tag v0.2.0）
#    → 自动触发 publish.yml → 发布到 PyPI

# 4. 用户即可安装
pip install fincore
```
