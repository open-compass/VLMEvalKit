# SEEDBench_IMG Evaluation Results

> LLaVA, mPLUG-Owl2, XComposer use specific prompts (defined in the official repo) for multiple-choice questions. 

### SEEDBench_IMG Scores (Vanilla / ChatGPT Answer Extraction / Official Leaderboard)

- **ExactMatchRate**: The success rate of extracting the option label with heuristic matching. 
- **MatchedAcc:** The overall accuracy across questions with predictions successfully matched, **with exact matching**. 
- **ExactMatchAcc:** The overall accuracy across all questions with **exact matching** (if prediction not successfully matched with an option, count it as wrong). 
- **LLMMatchAcc:** The overall accuracy across all questions with **ChatGPT answer matching**.
- **OfficialAcc**: SEEDBench_IMG acc on the official leaderboard (if applicable). 

| Model                | ExactMatchRate | MatchedAcc | ExactMatchAcc | LLMMatchAcc | Official Leaderboard |
| :------------------- | -------------: | ---------: | ------------: | ----------: | -------------------: |
| llava_v1.5_13b       |            100 |      68.11 |         68.11 |       68.11 |                 68.2 |
| XComposer            |            100 |      66.07 |         66.07 |       66.07 |                 66.9 |
| llava_v1.5_7b        |            100 |      65.59 |         65.59 |       65.59 |                  N/A |
| qwen_chat            |          96.21 |      66.61 |         64.08 |       64.83 |                 65.4 |
| mPLUG-Owl2           |            100 |      64.52 |         64.52 |       64.52 |                 64.1 |
| qwen_base            |          99.28 |      52.69 |         52.31 |       52.53 |                 62.3 |
| idefics_80b_instruct |          99.84 |      51.96 |         51.88 |       51.96 |                 53.2 |
| llava_v1_7b          |          82.51 |      50.18 |         41.41 |       49.48 |                  N/A |
| PandaGPT_13B         |          82.02 |      48.41 |         39.71 |       47.63 |                  N/A |
| instructblip_13b     |          99.07 |       47.5 |         47.06 |       47.26 |                  N/A |
| VisualGLM_6b         |          74.15 |      47.66 |         35.34 |       47.02 |                  N/A |
| idefics_9b_instruct  |          99.52 |      44.97 |         44.75 |          45 |                 44.5 |
| instructblip_7b      |          88.35 |      49.63 |         43.84 |       44.51 |                 58.8 |
| MiniGPT-4-v1-13B     |          67.71 |      39.37 |         26.66 |       34.91 |                  N/A |
| MiniGPT-4-v1-7B      |          69.25 |      33.62 |         23.29 |       31.56 |                 47.4 |
| MiniGPT-4-v2         |           81.4 |      31.81 |         25.89 |       29.38 |                  N/A |
| flamingov2           |          99.84 |      28.83 |         28.79 |       28.84 |                 42.7 |

### Comments

For models with limited instruction following capabilities (including qwen_base, MiniGPT-4, InstructBLIP, flamingov2), the performance gap between generation-based evaluation and PPL-based evaluation is significant. 

