source /fs-computility/mllm1/fangxinyu/miniconda3/bin/activate
cd /fs-computility/mllm1/fangxinyu/model_root_path/VLMEvalKit
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/fs-computility/mllm1/shared
export HOME=/fs-computility/mllm1/fangxinyu
conda activate qwen2vl
torchrun --nproc-per-node=8 run.py --model Qwen2.5-VL-7B-Instruct-FOR-Video --data MMBench_Video_2fps_nopack MVBench_MP4_2fps Video-MME_2fps MLVU_2fps TempCompass_2fps --work-dir ./outputs/video_lb_update --verbose








