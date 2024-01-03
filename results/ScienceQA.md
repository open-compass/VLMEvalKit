# ScienceQA Evaluation Results

> We benchmark the **image** subset of ScienceQA validation and test set, and report the Top-1 accuracy. 
>
> During evaluation, we use `GPT-3.5-Turbo-0613` as the choice extractor for all VLMs if the choice can not be extracted via heuristic matching. **Zero-shot** inference is adopted. 

## ScienceQA Accuracy

| Model                         | ScienceQA-Image Val | ScienceQA-Image Test |
| :---------------------------- | ------------------: | -------------------: |
| InternLM-XComposer-VL         |                  88 |                 89.8 |
| Human Performance             |                     |                 87.5 |
| GPT-4v (detail: low)          |                84.6 |                 82.1 |
| GeminiProVision               |                80.1 |                 81.4 |
| TransCore-M                   |                68.9 |                 72.1 |
| LLaVA-v1.5-13B                |                69.2 |                   72 |
| LLaVA-v1.5-13B (LoRA, XTuner) |                68.9 |                 70.3 |
| mPLUG-Owl2                    |                69.5 |                 69.5 |
| ShareGPT4V-7B                 |                68.1 |                 69.4 |
| LLaVA-v1.5-7B                 |                66.6 |                 68.9 |
| Qwen-VL-Chat                  |                65.5 |                 68.8 |
| LLaVA-v1.5-7B (LoRA, XTuner)  |                68.8 |                 68.7 |
| LLaVA-InternLM-7B (LoRA)      |                65.3 |                 68.4 |
| PandaGPT-13B                  |                60.9 |                 63.2 |
| IDEFICS-80B-Instruct          |                59.9 |                 61.8 |
| Qwen-VL                       |                57.7 |                 61.1 |
| LLaVA-v1-7B                   |                59.9 |                 60.5 |
| InstructBLIP-13B              |                53.3 |                 58.3 |
| VisualGLM                     |                53.4 |                 56.1 |
| MiniGPT-4-v2                  |                54.1 |                 54.7 |
| InstructBLIP-7B               |                54.7 |                 54.1 |
| IDEFICS-9B-Instruct           |                51.6 |                 53.5 |
| MiniGPT-4-v1-13B              |                44.3 |                   46 |
| OpenFlamingo v2               |                45.7 |                 44.8 |
| MiniGPT-4-v1-7B               |                  39 |                 39.6 |