# Using LMDeploy to Accelerate Evaluation and Inference

VLMEvalKit supports testing VLM models deployed by LMDeploy. Below, we use InternVL2-8B as an example to show how to test the model.

## Step 0: Install LMDeploy

```bash
pip install lmdeploy
```
For other installation methods, you can refer to LMDeploy's [documentation](https://github.com/InternLM/lmdeploy).

## Step 1: Start the Inference Service

```bash
lmdeploy serve api_server OpenGVLab/InternVL2-8B --model-name InternVL2-8B --server-port 23333
```
> [!IMPORTANT]
> Since models in VLMEvalKit may have custom behaviors when building prompts for different datasets, such as InternVL2's handling of HallusionBench, it is necessary to specify `--model-name` when starting the server. This allows the VLMEvalKit to select appropriate prompt construction strategy based on the name when using the LMDeploy API.
>
> The evaluation command below connects to `http://localhost:23333/v1`. If you choose a different server port, update `--base-url` to match.


## Step 2: Evaluation

```bash
python run.py --api-mode --data MMStar --model InternVL2-8B --base-url http://localhost:23333/v1 --model-class LMDeployAPI --verbose --api-nproc 64
```
