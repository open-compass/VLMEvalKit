source /fs-computility/mllm1/fangxinyu/miniconda3/bin/activate
cd /fs-computility/mllm1/fangxinyu/model_root_path/VLMEvalKit
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/fs-computility/mllm1/shared
export HOME=/fs-computility/mllm1/fangxinyu
conda activate vlmevalkit
torchrun --nproc-per-node=8 run.py --model SmolVLM2 --data MMBench_Video_64frame_nopack MVBench_64frame MLVU_64frame Video-MME_64frame TempCompass_64frame --work-dir ./outputs/video_lb_update --verbose








