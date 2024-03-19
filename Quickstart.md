# Quickstart

Before running the evaluation script, you need to **configure** the VLMs and set the model_paths properly.

After that, you can use a single script `run.py` to inference and evaluate multiple VLMs and benchmarks at a same time.

## Step0. Installation

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

## Step1. Configuration

**VLM Configuration**: All VLMs are configured in `vlmeval/config.py`, for some VLMs, you need to configure the code root (MiniGPT-4, PandaGPT, etc.) or the model_weight root (LLaVA-v1-7B, etc.) before conducting the evaluation. During evaluation, you should use the model name specified in `supported_VLM` in `vlmeval/config.py` to select the VLM. For MiniGPT-4 and InstructBLIP, you also need to modify the config files in `vlmeval/vlm/misc` to configure LLM path and ckpt path.

Following VLMs require the configuration step:

**Code Preparation & Installation**: InstructBLIP ([LAVIS](https://github.com/salesforce/LAVIS)), LLaVA ([LLaVA](https://github.com/haotian-liu/LLaVA)), MiniGPT-4 ([MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)), mPLUG-Owl2 ([mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2)), OpenFlamingo-v2 ([OpenFlamingo](https://github.com/mlfoundations/open_flamingo)), PandaGPT-13B ([PandaGPT](https://github.com/yxuansu/PandaGPT)), TransCore-M ([TransCore-M](https://github.com/PCIResearch/TransCore-M)).

**Manual Weight Preparation & Configuration**: InstructBLIP, LLaVA-v1-7B, MiniGPT-4, PandaGPT-13B

## Step2. Evaluation

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
