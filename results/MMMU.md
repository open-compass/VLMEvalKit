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

| Model                | Overall<br>(Val) | Overall<br>(Dev) | Art & Design<br>(Val) | Business<br>(Val) | Science<br>(Val) | Health & Medicine<br>(Val) | Humanities & Social Science<br>(Val) | Tech & Engineering<br>(Val) |
| :------------------- | ---------------: | ---------------: | --------------------: | ----------------: | ---------------: | -------------------------: | -----------------------------------: | --------------------------: |
| qwen_chat            |             37.6 |               30 |                  49.2 |                36 |               28 |                       32.7 |                                 55.8 |                        31.9 |
| llava_v1.5_13b       |             36.8 |               42 |                  49.2 |              23.3 |               36 |                         34 |                                 51.7 |                        33.3 |
| sharegpt4v_7b        |             36.7 |               30 |                    50 |              27.3 |             26.7 |                       37.3 |                                   50 |                        34.8 |
| TransCore_M          |             36.6 |             38.7 |                  54.2 |                32 |             27.3 |                         32 |                                 49.2 |                        32.4 |
| llava_v1.5_7b        |             36.1 |             38.7 |                  45.8 |              25.3 |               34 |                         32 |                                 48.3 |                        35.7 |
| instructblip_13b     |             32.9 |               30 |                  37.5 |              29.3 |               32 |                       28.7 |                                 37.5 |                        33.8 |
| PandaGPT_13B         |             32.7 |             26.7 |                  42.5 |              35.3 |               30 |                       29.3 |                                 45.8 |                        21.9 |
| llava_v1_7b          |             32.1 |             33.3 |                  31.7 |              24.7 |             31.3 |                         32 |                                 37.5 |                        35.2 |
| instructblip_7b      |             30.4 |               24 |                  38.3 |                28 |               22 |                       30.7 |                                 39.2 |                        28.6 |
| VisualGLM_6b         |             28.9 |             28.7 |                    30 |                24 |               28 |                         28 |                                 40.8 |                        26.2 |
| qwen_base            |             28.8 |             29.3 |                  43.3 |              18.7 |             25.3 |                       32.7 |                                 42.5 |                        19.5 |
| flamingov2           |             28.2 |             21.3 |                  27.5 |                30 |             28.7 |                         28 |                                 33.3 |                        24.3 |
| **Frequent Choice**  |         **26.8** |                  |                       |                   |                  |                            |                                      |                             |
| MiniGPT-4-v1-13B     |             26.2 |             23.3 |                  33.3 |              19.3 |             28.7 |                         26 |                                 34.2 |                          21 |
| idefics_80b_instruct |             25.1 |             23.3 |                  39.2 |              17.3 |             23.3 |                         24 |                                 48.3 |                        11.4 |
| MiniGPT-4-v2         |             24.6 |               32 |                  27.5 |              22.7 |             21.3 |                         28 |                                 33.3 |                          19 |
| MiniGPT-4-v1-7B      |               23 |             19.3 |                  32.5 |              27.3 |             18.7 |                       17.3 |                                   15 |                        26.2 |
| **Random Choice**    |         **22.1** |                  |                       |                   |                  |                            |                                      |                             |
| idefics_9b_instruct  |             19.6 |               20 |                  22.5 |              11.3 |             20.7 |                       23.3 |                                 31.7 |                        13.3 |