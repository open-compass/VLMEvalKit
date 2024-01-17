# SEEDBench_IMG Evaluation Results

> LLaVA, mPLUG-Owl2, XComposer use specific prompts (defined in the official repo) for multiple-choice questions. 

### SEEDBench_IMG Scores (Vanilla / ChatGPT Answer Extraction / Official Leaderboard)

- **ExactMatchRate**: The success rate of extracting the option label with heuristic matching. 
- **MatchedAcc:** The overall accuracy across questions with predictions successfully matched, **with exact matching**. 
- **ExactMatchAcc:** The overall accuracy across all questions with **exact matching** (if prediction not successfully matched with an option, count it as wrong). 
- **LLMMatchAcc:** The overall accuracy across all questions with **ChatGPT answer matching**.
- **OfficialAcc**: SEEDBench_IMG acc on the official leaderboard (if applicable). 

| Model                         | ExactMatchRate | MatchedAcc | ExactMatchAcc | LLMMatchAcc | [**Official Leaderboard (Eval Method)**](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) |
| :---------------------------- | -------------: | ---------: | ------------: | ----------: | ------------------------------------------------------------ |
| GPT-4v (detail: low)          |          95.58 |      73.92 |         70.65 |       71.59 | 69.1 (Gen)                                                   |
| GeminiProVision               |          99.38 |      71.09 |         70.65 |       70.74 | N/A                                                          |
| ShareGPT4V-7B                 |            100 |      69.25 |         69.25 |       69.25 | 69.7 (Gen)                                                   |
| CogVLM-17B-Chat               |            99.93 |        68.76 |           68.71 |         68.76 |
| LLaVA-v1.5-13B                |            100 |      68.11 |         68.11 |       68.11 | 68.2 (Gen)                                                   |
| LLaVA-v1.5-13B (QLoRA, XTuner) |            100 |      67.95 |         67.95 |       67.95 | N/A                                                          |
| TransCore-M                   |            100 |      66.77 |         66.77 |       66.77 | N/A                                                          |
| LLaVA-v1.5-7B (QLoRA, XTuner)  |            100 |      66.39 |         66.39 |       66.39 | N/A                                                          |
| InternLM-XComposer-VL         |            100 |      66.07 |         66.07 |       66.07 | 66.9 (PPL)                                                   |
| LLaVA-InternLM-7B (QLoRA)      |            100 |      65.75 |         65.75 |       65.75 | N/A                                                          |
| LLaVA-v1.5-7B                 |            100 |      65.59 |         65.59 |       65.59 | N/A                                                          |
| Qwen-VL-Chat                  |          96.21 |      66.61 |         64.08 |       64.83 | 65.4 (PPL)                                                   |
| mPLUG-Owl2                    |            100 |      64.52 |         64.52 |       64.52 | 64.1 (Not Given)                                             |
| SharedCaptioner               |            84.57 |        64.7  |           54.71 |         61.22 |
| Qwen-VL                       |          99.28 |      52.69 |         52.31 |       52.53 | 62.3 (PPL)                                                   |
| IDEFICS-80B-Instruct          |          99.84 |      51.96 |         51.88 |       51.96 | 53.2 (Not Given)                                             |
| LLaVA-v1-7B                   |          82.51 |      50.18 |         41.41 |       49.48 | N/A                                                          |
| PandaGPT_13B                  |          82.02 |      48.41 |         39.71 |       47.63 | N/A                                                          |
| InstructBLIP_13B              |          99.07 |       47.5 |         47.06 |       47.26 | N/A                                                          |
| VisualGLM                     |          74.15 |      47.66 |         35.34 |       47.02 | N/A                                                          |
| IDEFICS-9B-Instruct           |          99.52 |      44.97 |         44.75 |          45 | 44.5 (Not Given)                                             |
| InstructBLIP_7B               |          88.35 |      49.63 |         43.84 |       44.51 | 58.8 (PPL)                                                   |
| MiniGPT-4-v1-13B              |          67.71 |      39.37 |         26.66 |       34.91 | N/A                                                          |
| MiniGPT-4-v1-7B               |          69.25 |      33.62 |         23.29 |       31.56 | 47.4 (PPL)                                                   |
| MiniGPT-4-v2                  |           81.4 |      31.81 |         25.89 |       29.38 | N/A                                                          |
| OpenFlamingo v2               |          99.84 |      28.83 |         28.79 |       28.84 | 42.7 (PPL)                                                   |

### Comments

For models with limited instruction following capabilities (including qwen_base, MiniGPT-4, InstructBLIP, flamingov2), the performance gap between generation-based evaluation and PPL-based evaluation is significant. 

