#!/usr/bin/env bash
# 环境准备：依赖 + 数据 + 基座模型。建议在魔搭实例的 /mnt/workspace 下 clone 本项目后运行。
set -e

echo "== 1. 安装依赖 =="
pip install -r requirements.txt

echo "== 2. 下载数据与基座（内网约 2-5 分钟）=="
modelscope download --dataset AI-MO/NuminaMath-CoT --local_dir data/numina
modelscope download --dataset evalscope/cmath      --local_dir data/cmath
modelscope download --model Qwen/Qwen3-1.7B-Instruct-2507 --local_dir models/qwen3-1.7b

echo "== 3. 构建训练/评测数据 =="
python scripts/build_dataset.py --n-train 5000 --n-val 500

echo "== 4. 链路自检 =="
python scripts/train_sft.py --smoke
echo "== setup 完成。下一步：python scripts/train_grpo.py --smoke =="
