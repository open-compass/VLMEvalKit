source /fs-computility/mllm1/fangxinyu/miniconda3/bin/activate
cd /fs-computility/mllm1/fangxinyu/model_root_path/VLMEvalKit
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/fs-computility/mllm1/shared
export HOME=/fs-computility/mllm1/fangxinyu
conda activate qwen2vl
torchrun --nproc-per-node=8 run.py --model Qwen2-VL-7B-Instruct --data MMSci_DEV_Captioning_image_only MMSci_DEV_Captioning_with_abs --verbose --reuse --api-nproc 20








