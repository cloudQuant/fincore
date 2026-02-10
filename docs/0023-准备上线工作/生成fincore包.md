优化一下ci/cd，确保可以在win11, linux, macos上,确保能在python3.11, python3.12, python3.13上都能够成功测试通过。

然后希望能够把这个打包上传到pypi上，然后用户就可以通过pip install fincore来安装了。

## 已完成

### 1. CI/CD (GitHub Actions)

**`.github/workflows/ci.yml`** — 每次 push/PR 到 master/main 自动触发：
- **测试矩阵**: 3 OS (Ubuntu/macOS/Windows) × 3 Python (3.11/3.12/3.13) = 9 组合
- **pip 缓存**: 使用 `setup-python` 内置 `cache: 'pip'`，自动处理跨平台缓存路径
- **Lint**: ruff check
- **覆盖率**: Ubuntu + Python 3.11 上生成 coverage 报告
- **构建验证**: `python -m build` + `twine check`
- **并发控制**: 同 ref 自动取消旧 workflow

**`.github/workflows/publish.yml`** — 创建 GitHub Release 时自动发布到 PyPI：
- 使用 OIDC trusted publishing（无需 API token）
- 需要在 PyPI 设置 trusted publisher:
  - Owner: `cloudQuant`, Repository: `fincore`, Workflow: `publish.yml`, Environment: `pypi`

### 2. 配置更新

- **`pyproject.toml`** / **`setup.py`**: `requires-python >= 3.11`，classifiers 仅保留 3.11/3.12/3.13
- **`pyproject.toml`**: ruff `target-version = "py311"`, mypy `python_version = "3.11"`
- **`requirements-test.txt`**: 移除 `nose`（Python 3.12+ 不兼容）、`six`（未使用）、`pandas-datareader`（未使用）

### 3. 源码修复（测试全部通过 1299/1299）

- **rolling.py**: 修复空 DatetimeIndex dtype 不匹配（`datetime64[s]` vs `datetime64[us]`）
- **transactions.py**: 移除已废弃的 `infer_objects(copy=False)` 参数
- **perf_attrib.py**: 添加 `sort=False` 到 `pd.concat` 消除 Pandas4Warning

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
