#!/bin/bash
set -x
export USE_COT="1"
export PRED_FORMAT=tsv
export VIDEOMME_PATH=/mnt/shared-storage-user/agent4review-share/linjunming/LMUData/VMME
# export RANK=0
# export WORLD_SIZE=1
# export LOCAL_RANK=0
# export TP=1
# 设置参数


model_name="Qwen3-VL-30B-A3B-Thinking_api"
dataset="Video-MME_64frame"

# 创建日志目录
mkdir -p log_eval

# 生成日期时间戳
current_date=$(date +"%Y%m%d_%H%M%S")

# 构建日志文件名
log_file="log_eval/${model_name}_${current_date}.log"

# 使用nohup运行Python脚本
nohup python3 run.py --data ${dataset} --model ${model_name} --verbose --reuse --api-nproc 256 > qwen3vl_thinking_videomme.log 2>&1 &

echo "实验已在后台运行，日志文件: ${log_file}"
echo "可以使用以下命令查看日志:"
echo "tail -f ${log_file}"