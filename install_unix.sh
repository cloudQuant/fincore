#!/bin/bash

# 定义变量
BUILD_DIR="build"
EGG_INFO_DIR="empyrical.egg-info"
BENCHMARKS_DIR=".benchmarks"



# 切换到上一级目录
cd ..

# 安装 requirements.txt 中的依赖
pip install -U -r ./empyrical/requirements.txt

# 安装 empyrical 包
pip install -U --no-build-isolation ./empyrical

# 运行 empyrical 的测试用例，使用 4 个进程并行测试
pytest ./empyrical/tests -n 4

# 切换到 empyrical 目录
cd ./empyrical



# 删除中间构建和 egg-info 目录
echo "Deleting intermediate files..."
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
    echo "Deleted $BUILD_DIR directory."
fi

if [ -d "$EGG_INFO_DIR" ]; then
    rm -rf "$EGG_INFO_DIR"
    echo "Deleted $EGG_INFO_DIR directory."
fi

# 删除 .benchmarks 目录
if [ -d "$BENCHMARKS_DIR" ]; then
    rm -rf "$BENCHMARKS_DIR"
    echo "Deleted $BENCHMARKS_DIR directory."
fi

# 删除所有 .log 文件
echo "Deleting all .log files..."
find . -type f -name "*.log" -exec rm -f {} \;
echo "All .log files deleted."



