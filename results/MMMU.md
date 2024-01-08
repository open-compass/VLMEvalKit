# MMMU Evaluation Results

> - For MMMU, we support the evaluation of the `dev` (150 samples) and `validation` (900 samples) set. Here we only report the results on the `validation` set. 
> - **Answer Inference:**
>   - For models with `interleave_generate` interface (accept interleaved images & texts as inputs), all testing samples can be inferred. **`interleave_generate` is adopted for inference.**
>   - For models without `interleave_generate` interface, samples with more than one images are skipped (42 out of 1050, directly count as wrong). **`generate` is adopted for inference.**
> - **Evaluation**:
>   - MMMU include two types of questions: **multi-choice questions** & **open-ended QA**. 
>   - For **open-ended QA (62/1050)**, we re-formulate it as multi-choice questions: `{'question': 'QQQ', 'answer': 'AAA'} -> {'question': 'QQQ', 'A': 'AAA', 'B': 'Other Answers', 'answer': 'A'}`, and then adopt the same evaluation paradigm for **multi-choice questions**. 
>   - For **multi-choice questions (988/1050)**, we use **GPT-3.5-Turbo-0613** for matching prediction with options if heuristic matching does not work. 

### MMMU Scores

| Model                         |   Overall |   Art & Design |   Business |   Science |   Health & Medicine |   Humanities & Social Science |   Tech & Engineering |
|:------------------------------|----------:|---------------:|-----------:|----------:|--------------------:|------------------------------:|---------------------:|
| GPT-4v (detail: low)          |      53.8 |           67.5 |       59.3 |      46   |                54.7 |                          70.8 |                 37.1 |
| GeminiProVision               |      48.9 |           59.2 |       36.7 |      42.7 |                52   |                          66.7 |                 43.8 |
| CogVLM-17B-Chat               |      37.3 |           51.7 |       34   |      36   |                35.3 |                          41.7 |                 31.4 |
| Qwen-VL-Chat                  |      37   |           49.2 |       35.3 |      28   |                31.3 |                          54.2 |                 31.9 |
| LLaVA-InternLM-7B (LoRA)      |      36.9 |           44.2 |       32   |      29.3 |                38.7 |                          46.7 |                 34.8 |
| LLaVA-v1.5-13B                |      36.9 |           49.2 |       24   |      37.3 |                33.3 |                          50.8 |                 33.3 |
| TransCore-M                   |      36.9 |           54.2 |       32.7 |      28   |                32   |                          48.3 |                 33.3 |
| ShareGPT4V-7B                 |      36.6 |           50   |       28.7 |      26   |                37.3 |                          49.2 |                 34.3 |
| SharedCaptioner               |      36.3 |           44.2 |       28.7 |      29.3 |                37.3 |                          45.8 |                 36.2 |
| LLaVA-v1.5-7B                 |      36.2 |           45.8 |       26   |      34   |                32.7 |                          47.5 |                 35.7 |
| InternLM-XComposer-VL         |      35.6 |           45.8 |       28.7 |      22.7 |                30.7 |                          52.5 |                 37.6 |
| LLaVA-v1.5-13B (LoRA, XTuner) |      35.2 |           40.8 |       30.7 |      27.3 |                35.3 |                          44.2 |                 35.7 |
| mPLUG-Owl2                    |      34.7 |           47.5 |       26   |      21.3 |                38   |                          50   |                 31.9 |
| LLaVA-v1.5-7B (LoRA, XTuner)  |      33.7 |           48.3 |       23.3 |      30.7 |                32.7 |                          45.8 |                 28.6 |
| InstructBLIP-13B              |      33.2 |           37.5 |       30   |      32.7 |                30   |                          36.7 |                 33.8 |
| PandaGPT-13B                  |      32.9 |           42.5 |       36   |      30.7 |                30   |                          43.3 |                 22.9 |
| LLaVA-v1-7B                   |      32.3 |           31.7 |       26   |      31.3 |                32.7 |                          35.8 |                 35.7 |
| InstructBLIP-7B               |      30.6 |           38.3 |       28.7 |      22   |                30.7 |                          39.2 |                 28.6 |
| VisualGLM                     |      29.9 |           30.8 |       27.3 |      28.7 |                29.3 |                          40.8 |                 26.2 |
| Qwen-VL                       |      29.6 |           45   |       18.7 |      26.7 |                32.7 |                          42.5 |                 21   |
| OpenFlamingo v2               |      28.8 |           27.5 |       30.7 |      29.3 |                28.7 |                          33.3 |                 25.2 |
| MiniGPT-4-v1-13B              |      26.3 |           31.7 |       20.7 |      28   |                25.3 |                          35   |                 21.9 |
| Frequent Choice               |      25.8 |           26.7 |       28.4 |      24   |                24.4 |                          25.2 |                 26.5 |
| MiniGPT-4-v2                  |      25   |           27.5 |       23.3 |      22   |                27.3 |                          32.5 |                 21   |
| IDEFICS-80B-Instruct          |      24   |           39.2 |       18   |      20   |                22   |                          46.7 |                 11   |
| MiniGPT-4-v1-7B               |      23.6 |           33.3 |       28.7 |      19.3 |                18   |                          15   |                 26.2 |
| IDEFICS-9B-Instruct           |      18.4 |           22.5 |       11.3 |      17.3 |                21.3 |                          30   |                 13.3 |