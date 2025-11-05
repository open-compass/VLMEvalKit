#!/bin/bash
set -x
export USE_COT="1"
export PRED_FORMAT=tsv
export LMUData=/mnt/shared-storage-user/agent4review-share/linjunming/LMUData

model_name="Qwen3-VL-32B-Thinking"
dataset="MMBench_DEV_EN"

# 创建日志目录
mkdir -p log_eval

# 生成日期时间戳
current_date=$(date +"%Y%m%d_%H%M%S")

# 构建日志文件名
log_file="log_eval/${model_name}_${current_date}.log"

# 使用nohup运行Python脚本并获取PID
CUDA_VISIBLE_DEVICES=5,6 nohup python run.py --model ${model_name} --data ${dataset} --mode infer > ${log_file} 2>&1 &
PID=$!

echo "实验已在后台运行，PID: $PID"
echo "日志文件: ${log_file}"
echo "可以使用以下命令查看日志:"
echo "tail -f ${log_file}"
echo "可以使用以下命令终止进程:"
echo "kill $PID"