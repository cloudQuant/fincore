@echo off
:: 安装 requirements.txt 中的依赖
pip install -U -r requirements.txt

SET BUILD_DIR=build
SET EGG_INFO_DIR=empyrical.egg-info
SET BENCHMARKS_DIR=.benchmarks

:: 切换到脚本所在目录的上一级目录，确保相对路径正确
cd /d "%~dp0.."

:: 安装 empyrical 包
:: pip install -U --no-build-isolation ./empyrical
pip install -U ./empyrical

:: 运行测试用例，使用 4 个进程并行测试
pytest ./empyrical/tests -n 4

cd ./empyrical
:: 删除中间构建和 egg-info 目录
echo Deleting intermediate files...
IF EXIST %BUILD_DIR% (
    rmdir /s /q %BUILD_DIR%
    echo Deleted %BUILD_DIR% directory.
)
IF EXIST %EGG_INFO_DIR% (
    rmdir /s /q %EGG_INFO_DIR%
    echo Deleted %EGG_INFO_DIR% directory.
)
:: 删除 pytest 生成的 .benchmarks 目录
IF EXIST %BENCHMARKS_DIR% (
    rmdir /s /q %BENCHMARKS_DIR%
    echo Deleted %BENCHMARKS_DIR% directory.
)

:: 删除所有 .log 文件
echo Deleting all .log files...
del /s /q *.log
echo All .log files deleted.

:: 暂停脚本，以便查看输出结果
:: pause
