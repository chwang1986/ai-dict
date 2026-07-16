#!/bin/bash
# AI 词典启动脚本

set -e

# 检查 Python3
if ! command -v python3 &> /dev/null; then
    echo "错误：未找到 python3，请先安装 Python 3"
    exit 1
fi

echo "Python3 版本：$(python3 --version)"

# 检查 pip
if ! command -v pip3 &> /dev/null; then
    echo "错误：未找到 pip3，请先安装 pip"
    exit 1
fi

# 安装依赖
echo "正在安装依赖..."
pip3 install -r requirements.txt

# 初始化数据库（如果不存在）
if [ ! -f dict.db ]; then
    echo "正在初始化数据库..."
    python3 seed.py
else
    echo "数据库已存在，跳过初始化"
fi

# 启动 Flask 开发服务器
echo "正在启动 AI 词典服务器（端口 5001）..."
python3 app.py