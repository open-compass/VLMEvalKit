source /fs-computility/mllm1/fangxinyu/miniconda3/bin/activate
cd /fs-computility/mllm1/fangxinyu/model_root_path/VLMEvalKit
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/fs-computility/llm/shared/mllm
export HOME=/fs-computility/mllm1/fangxinyu
export PROXY_ENDPOINT_2=http://100.68.170.107:3128
export http_proxy=$PROXY_ENDPOINT_2
export https_proxy=$PROXY_ENDPOINT_2
export HTTP_PROXY=$PROXY_ENDPOINT_2
export HTTPS_PROXY=$PROXY_ENDPOINT_2
conda activate vlmevalkit-v2
torchrun --nproc-per-node=8 run.py --model InternVL2-8B --data MMSci_DEV_Captioning_image_only MMSci_DEV_Captioning_with_abs --verbose --reuse --api-nproc 20








