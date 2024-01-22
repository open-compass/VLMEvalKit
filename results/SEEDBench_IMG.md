# SEEDBench_IMG Evaluation Results

> LLaVA, mPLUG-Owl2, XComposer use specific prompts (defined in the official repo) for multiple-choice questions. 

### SEEDBench_IMG Scores (Vanilla / ChatGPT Answer Extraction / Official Leaderboard)

- **Acc w/o. ChatGPT Extraction**: The accuracy when using exact matching for evaluation. 
- **Acc w. ChatGPT Extraction**: The overall accuracy across all questions with **ChatGPT answer matching**.
- **Official**: SEEDBench_IMG acc on the official leaderboard (if applicable). 

| Model                       |   Acc w/o. ChatGPT Extraction |   Acc w. ChatGPT Extraction | [**Official (Eval Method)**](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard)   |
|:----------------------------|------------------------------:|----------------------------:|:-----------------------------------------------------------------------------------------------|
| GPT-4v (detail: low)        |                         70.65 |                       71.59 | 69.1 (Gen)                                                                                     |
| TransCore-M                 |                         71.22 |                       71.22 | N/A                                                                                            |
| LLaVA-InternLM2-7B (QLoRA)  |                         71.2  |                       71.2  | N/A                                                                                            |
| GeminiProVision             |                         70.65 |                       70.74 | N/A                                                                                            |
| ShareGPT4V-13B              |                         70.66 |                       70.66 | 70.8 (Gen)                                                                                     |
| LLaVA-InternLM2-20B (QLoRA) |                         70.24 |                       70.24 | N/A                                                                                            |
| ShareGPT4V-7B               |                         69.25 |                       69.25 | 69.7 (Gen)                                                                                     |
| EMU2-Chat                   |                         68.35 |                       68.89 | N/A                                                                                            |
| Monkey-Chat                 |                         68.89 |                       68.89 | N/A                                                                                            |
| CogVLM-17B-Chat             |                         68.71 |                       68.76 | N/A                                                                                            |
| LLaVA-v1.5-13B              |                         68.11 |                       68.11 | 68.2 (Gen)                                                                                     |
| LLaVA-v1.5-13B (QLoRA)      |                         67.95 |                       67.95 | N/A                                                                                            |
| LLaVA-v1.5-7B (QLoRA)       |                         66.39 |                       66.39 | N/A                                                                                            |
| InternLM-XComposer-VL       |                         66.07 |                       66.07 | 66.9 (PPL)                                                                                     |
| LLaVA-InternLM-7B (QLoRA)   |                         65.75 |                       65.75 | N/A                                                                                            |
| QwenVLPlus                  |                         55.27 |                       65.73 | N/A                                                                                            |
| LLaVA-v1.5-7B               |                         65.59 |                       65.59 | N/A                                                                                            |
| Qwen-VL-Chat                |                         64.08 |                       64.83 | 65.4 (PPL)                                                                                     |
| mPLUG-Owl2                  |                         64.52 |                       64.52 | 64.1 (Not Given)                                                                               |
| Monkey                      |                         64.3  |                       64.3  | N/A                                                                                            |
| SharedCaptioner             |                         54.71 |                       61.22 | N/A                                                                                            |
| Qwen-VL                     |                         52.31 |                       52.53 | 62.3 (PPL)                                                                                     |
| IDEFICS-80B-Instruct        |                         51.88 |                       51.96 | 53.2 (Not Given)                                                                               |
| LLaVA-v1-7B                 |                         41.41 |                       49.48 | N/A                                                                                            |
| PandaGPT-13B                |                         39.71 |                       47.63 | N/A                                                                                            |
| InstructBLIP-13B            |                         47.06 |                       47.26 | N/A                                                                                            |
| VisualGLM                   |                         35.34 |                       47.02 | N/A                                                                                            |
| IDEFICS-9B-Instruct         |                         44.75 |                       45    | 44.5 (Not Given)                                                                               |
| InstructBLIP-7B             |                         43.84 |                       44.51 | 58.8 (PPL)                                                                                     |
| MiniGPT-4-v1-13B            |                         26.66 |                       34.91 | N/A                                                                                            |
| MiniGPT-4-v1-7B             |                         23.29 |                       31.56 | 47.4 (PPL)                                                                                     |
| MiniGPT-4-v2                |                         25.89 |                       29.38 | N/A                                                                                            |
| OpenFlamingo v2             |                         28.79 |                       28.84 | 42.7 (PPL)                                                                                     |

### Comments

For models with limited instruction following capabilities (including qwen_base, MiniGPT-4, InstructBLIP, flamingov2), the performance gap between generation-based evaluation and PPL-based evaluation is significant. 

