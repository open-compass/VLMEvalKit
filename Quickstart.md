# Quickstart

Before running the evaluation script, you need to **configure** the VLMs and set the model_paths properly.

After that, you can use a single script `run.py` to inference and evaluate multiple VLMs and benchmarks at a same time.

## Step 0. Installation & Setup essential keys

**Installation.**

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

**Setup Keys.**

To infer with API models (GPT-4v, Gemini-Pro-V, etc.) or use LLM APIs as the **judge or choice extractor**, you need to first setup API keys. VLMEvalKit will use an judge **LLM** to extract answer from the output if you set the key, otherwise it uses the **exact matching** mode (find "Yes", "No", "A", "B", "C"... in the output strings). **The exact matching can only be applied to the Yes-or-No tasks and the Multi-choice tasks.**
- You can place the required keys in `$VLMEvalKit/.env` or directly set them as the environment variable. If you choose to create a `.env` file, its content will look like:

  ```bash
  # The .env file, place it under $VLMEvalKit
  # Alles-apin-token, for intra-org use only
  ALLES=
  # API Keys of Proprietary VLMs
  DASHSCOPE_API_KEY=
  GOOGLE_API_KEY=
  OPENAI_API_KEY=
  OPENAI_API_BASE=
  STEPAI_API_KEY=
  ```

- Fill the blanks with your API keys (if necessary). Those API keys will be automatically loaded when doing the inference and evaluation.
## Step 1. Configuration

**VLM Configuration**: All VLMs are configured in `vlmeval/config.py`, for some VLMs, you need to configure the code root (MiniGPT-4, PandaGPT, etc.) or the model_weight root (LLaVA-v1-7B, etc.) before conducting the evaluation. During evaluation, you should use the model name specified in `supported_VLM` in `vlmeval/config.py` to select the VLM. For MiniGPT-4 and InstructBLIP, you also need to modify the config files in `vlmeval/vlm/misc` to configure LLM path and ckpt path.

Following VLMs require the configuration step:

**Code Preparation & Installation**: InstructBLIP ([LAVIS](https://github.com/salesforce/LAVIS)), LLaVA ([LLaVA](https://github.com/haotian-liu/LLaVA)), MiniGPT-4 ([MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)), mPLUG-Owl2 ([mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2)), OpenFlamingo-v2 ([OpenFlamingo](https://github.com/mlfoundations/open_flamingo)), PandaGPT-13B ([PandaGPT](https://github.com/yxuansu/PandaGPT)), TransCore-M ([TransCore-M](https://github.com/PCIResearch/TransCore-M)).

**Manual Weight Preparation & Configuration**: InstructBLIP, LLaVA-v1-7B, MiniGPT-4, PandaGPT-13B

## Step 2. Evaluation

We use `run.py` for evaluation. To use the script, you can use `$VLMEvalKit/run.py` or create a soft-link of the script (to use the script anywhere):

**Arguments**

- `--data (list[str])`: Set the dataset names that are supported in VLMEvalKit (defined in `vlmeval/utils/dataset_config.py`).
- `--model (list[str])`: Set the VLM names that are supported in VLMEvalKit (defined in `supported_VLM` in `vlmeval/config.py`).
- `--mode (str, default to 'all', choices are ['all', 'infer'])`: When `mode` set to "all", will perform both inference and evaluation; when set to "infer", will only perform the inference.
- `--nproc (int, default to 4)`: The number of threads for OpenAI API calling.

**Command**

You can run the script with `python` or `torchrun`:

```bash
# When running with `python`, only one VLM instance is instantiated, and it might use multiple GPUs (depending on its default behavior).
# That is recommended for evaluating very large VLMs (like IDEFICS-80B-Instruct).

# IDEFICS-80B-Instruct on MMBench_DEV_EN, MME, and SEEDBench_IMG, Inference and Evalution
python run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_80b_instruct --verbose
# IDEFICS-80B-Instruct on MMBench_DEV_EN, MME, and SEEDBench_IMG, Inference only
python run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_80b_instruct --verbose --mode infer

# When running with `torchrun`, one VLM instance is instantiated on each GPU. It can speed up the inference.
# However, that is only suitable for VLMs that consume small amounts of GPU memory.

# IDEFICS-9B-Instruct, Qwen-VL-Chat, mPLUG-Owl2 on MMBench_DEV_EN, MME, and SEEDBench_IMG. On a node with 8 GPU. Inference and Evaluation.
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_80b_instruct qwen_chat mPLUG-Owl2 --verbose
# Qwen-VL-Chat on MME. On a node with 2 GPU. Inference and Evaluation.
torchrun --nproc-per-node=2 run.py --data MME --model qwen_chat --verbose
```

The evaluation results will be printed as logs, besides. **Result Files** will also be generated in the directory `$YOUR_WORKING_DIRECTORY/{model_name}`. Files ending with `.csv` contain the evaluated metrics.

## Deploy a local language model as the judge / choice extractor
The default setting mentioned above uses OpenAI's GPT as the judge LLM. However, you can also deploy a local judge LLM with [LMDeploy](https://github.com/InternLM/lmdeploy).

First install:
```
pip install lmdeploy openai
```

And then deploy a local judge LLM with the single line of code. LMDeploy will automatically download the model from Huggingface. Assuming we use internlm2-chat-1_8b as the judge, port 23333, and the key sk-123456 (the key must start with "sk-" and follow with any number you like):
```
lmdeploy serve api_server internlm/internlm2-chat-1_8b --server-port 23333
```

You need to get the model name registered by LMDeploy with the following python code:
```
from openai import OpenAI
client = OpenAI(
    api_key='sk-123456',
    base_url="http://0.0.0.0:23333/v1"
)
model_name = client.models.list().data[0].id
```

Now set some environment variables to tell VLMEvalKit how to use the local judge LLM. As mentioned above, you can also set them in `$VLMEvalKit/.env` file:
```
OPENAI_API_KEY=sk-123456
OPENAI_API_BASE=http://0.0.0.0:23333/v1/chat/completions
LOCAL_LLM=<model_name you get>
```

Finally, you can run the commands in step 2 to evaluate your VLM with the local judge LLM.

Note that

- If you hope to deploy the judge LLM in a single GPU and evaluate your VLM on other GPUs because of limited GPU memory, try `CUDA_VISIBLE_DEVICES=x` like
```
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server internlm/internlm2-chat-1_8b --server-port 23333
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc-per-node=3 run.py --data HallusionBench  --model qwen_chat --verbose
```
- If the local judge LLM is not good enough in following the instructions, the evaluation may fail. Please report such failures (e.g., by issues).
- It's possible to deploy the judge LLM in different ways, e.g., use a private LLM (not from HuggingFace) or use a quantized LLM. Please refer to the [LMDeploy doc](https://lmdeploy.readthedocs.io/en/latest/serving/api_server.html). You can use any other deployment framework if they support OpenAI API.
