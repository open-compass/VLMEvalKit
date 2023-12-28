# MMMU Evaluation Results

> - In MMMU Evaluation, we evaluate the `dev` (150 samples) and `validation` (900 samples) set of MMMU. 
> - **Answer Inference:**
>   - For models with `interleave_generate` interface (accept interleaved images & texts as inputs), all testing samples can be inferred. **`interleave_generate` is adopted for inference.**
>   - For models without `interleave_generate` interface, samples with more than one images are skipped (42 out of 1050, directly count as wrong). **`generate` is adopted for inference.**
> - **Evaluation**:
>   - MMMU include two types of questions: **multi-choice questions** & **open-ended QA**. 
>   - For **open-ended QA (62/1050)**, we re-formulate it as multi-choice questions: `{'question': 'QQQ', 'answer': 'AAA'} -> {'question': 'QQQ', 'A': 'AAA', 'B': 'Other Answers', 'answer': 'A'}`, and then adopt the same evaluation paradigm for **multi-choice questions**. 
>   - For **multi-choice questions (988/1050)**, we use **GPT-3.5-Turbo-0613** for matching prediction with options if heuristic matching does not work. 

### MMMU Scores

| Model                |   Overall<br>(Val) |   Art & Design<br>(Val) |   Business<br>(Val) |   Science<br>(Val) |   Health & Medicine<br>(Val) |   Humanities & Social Science<br>(Val) |   Tech & Engineering<br>(Val) |   Overall<br>(Dev) |
|:---------------------|-------------------:|------------------------:|--------------------:|-------------------:|-----------------------------:|---------------------------------------:|------------------------------:|-------------------:|
| GPT-4v         |               53.8 |                    66.7 |                60   |               46   |                         54.7 |                                   71.7 |                          36.7 |               52.7 |
| GeminiProVision      |               48.4 |                    59.2 |                36   |               42   |                         52   |                                   66.7 |                          42.9 |               54   |
| qwen_chat            |               37.6 |                    49.2 |                36   |               28   |                         32.7 |                                   55.8 |                          31.9 |               30   |
| llava_v1.5_13b       |               36.8 |                    49.2 |                23.3 |               36   |                         34   |                                   51.7 |                          33.3 |               42   |
| sharegpt4v_7b        |               36.7 |                    50   |                27.3 |               26.7 |                         37.3 |                                   50   |                          34.8 |               30   |
| TransCore_M          |               36.6 |                    54.2 |                32   |               27.3 |                         32   |                                   49.2 |                          32.4 |               38.7 |
| llava_v1.5_7b        |               36.1 |                    45.8 |                25.3 |               34   |                         32   |                                   48.3 |                          35.7 |               38.7 |
| XComposer            |               35.7 |                    45.8 |                28.7 |               22.7 |                         30.7 |                                   53.3 |                          37.6 |               36.7 |
| mPLUG-Owl2           |               34.6 |                    47.5 |                26   |               21.3 |                         37.3 |                                   50   |                          31.9 |               40.7 |
| instructblip_13b     |               32.9 |                    37.5 |                29.3 |               32   |                         28.7 |                                   37.5 |                          33.8 |               30   |
| PandaGPT_13B         |               32.7 |                    42.5 |                35.3 |               30   |                         29.3 |                                   45.8 |                          21.9 |               26.7 |
| llava_v1_7b          |               32.1 |                    31.7 |                24.7 |               31.3 |                         32   |                                   37.5 |                          35.2 |               33.3 |
| instructblip_7b      |               30.4 |                    38.3 |                28   |               22   |                         30.7 |                                   39.2 |                          28.6 |               24   |
| VisualGLM_6b         |               28.9 |                    30   |                24   |               28   |                         28   |                                   40.8 |                          26.2 |               28.7 |
| qwen_base            |               28.8 |                    43.3 |                18.7 |               25.3 |                         32.7 |                                   42.5 |                          19.5 |               29.3 |
| flamingov2           |               28.2 |                    27.5 |                30   |               28.7 |                         28   |                                   33.3 |                          24.3 |               21.3 |
| Frequent Choice | 26.8 |  |  |  |  |  |  |  |
| MiniGPT-4-v1-13B     |               26.2 |                    33.3 |                19.3 |               28.7 |                         26   |                                   34.2 |                          21   |               23.3 |
| idefics_80b_instruct |               25.1 |                    39.2 |                17.3 |               23.3 |                         24   |                                   48.3 |                          11.4 |               23.3 |
| MiniGPT-4-v2         |               24.6 |                    27.5 |                22.7 |               21.3 |                         28   |                                   33.3 |                          19   |               32   |
| MiniGPT-4-v1-7B      |               23   |                    32.5 |                27.3 |               18.7 |                         17.3 |                                   15   |                          26.2 |               19.3 |
| Random Choice | 22.1 |  |  |  |  |  |  |  |
| idefics_9b_instruct  |               19.6 |                    22.5 |                11.3 |               20.7 |                         23.3 |                                   31.7 |                          13.3 |               20   |