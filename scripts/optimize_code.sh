#!/bin/bash
# Fincore 代码优化脚本
# 使用 pyupgrade, ruff 等工具优化代码风格和格式
# 配置参考: pyproject.toml (line-length=120, target-version=py311)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=========================================="
echo "Fincore 代码优化工具"
echo "=========================================="
echo ""

# 检查必要的工具
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ 错误: 未找到 $1"
        echo "请运行: pip install $2"
        exit 1
    fi
}

echo "📋 检查依赖工具..."
check_tool "python" "python3"
check_tool "ruff" "ruff"
python -c "import pyupgrade" 2>/dev/null || { echo "⚠️  缺少 pyupgrade, 正在安装..."; pip install pyupgrade; }
echo "✅ 所有依赖工具已安装"
echo ""

# 步骤 1: 使用 pyupgrade 升级 Python 语法到 3.11+
echo "🔧 步骤 1: 使用 pyupgrade 升级 Python 语法 (--py311-plus)..."
find fincore -name "*.py" -type f -exec python -m pyupgrade --py311-plus {} + 2>/dev/null || true
find tests -name "*.py" -type f -exec python -m pyupgrade --py311-plus {} + 2>/dev/null || true
echo "✅ pyupgrade 完成"
echo ""

# 步骤 2: 使用 ruff 格式化代码 (line-length=120)
# 注意: 格式化在 lint --fix 之前执行，确保格式一致
echo "🔧 步骤 2: 使用 ruff 格式化代码..."
ruff format fincore/ tests/
echo "✅ ruff format 完成"
echo ""

# 步骤 3: 使用 ruff 自动修复 lint 问题 (排除 F401/F811 避免删除动态使用的符号)
# 本项目大量使用 __getattr__ 和 registry 动态访问符号，F401 自动修复会误删
echo "🔧 步骤 3: 使用 ruff 进行 linting 并自动修复..."
ruff check fincore/ tests/ --fix --exit-zero --ignore F401,F811
echo "✅ ruff check 完成"
echo ""

# 步骤 4: 更新安装 fincore
echo "📦 步骤 4: 更新安装 fincore..."
pip install -U .
echo "✅ fincore 更新完成"
echo ""

# 步骤 5: 运行全部测试验证
echo "🧪 步骤 5: 运行全部测试验证代码完整性..."
if [ -d "tests" ]; then
    python -m pytest tests -n 8 --tb=short -q
    echo "✅ 所有测试通过"
else
    echo "⚠️  未找到测试目录"
fi
echo ""

echo "=========================================="
echo "✅ 代码优化完成！"
echo "=========================================="
