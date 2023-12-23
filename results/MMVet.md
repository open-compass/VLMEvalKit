# MMVet Evaluation Results

> - In MMVet Evaluation, we use GPT-4-Turbo (gpt-4-1106-preview) as the judge LLM to assign scores to the VLM outputs. We only perform the evaluation once due to the limited variance among results of multiple evaluation pass originally reported. 
> - No specific prompt template adopted for **ALL VLMs**.

### MMVet Scores

| Model                     | Overall | [Overall (Official Leaderboard)](https://paperswithcode.com/sota/visual-question-answering-on-mm-vet) |  ocr | math | spat |  rec | know |  gen |
| :------------------------ | ------: | -----------------------------------------------------------: | ---: | ---: | ---: | ---: | ---: | ---: |
| GeminiProVision           |    59.2 |                                                     64.3±0.4 | 63.6 | 41.5 | 61.2 | 59.8 | 51.0 | 48.0 |
| GPT-4v (detail: low)      |    56.8 |                                                     60.2±0.3 | 59.4 | 61.2 | 52.5 | 59.7 | 48.0 | 46.5 |
| qwen_chat                 |    47.3 |                                                          N/A | 37.2 | 22.3 | 42.8 | 52.5 | 45.4 | 40.3 |
| idefics_80b_instruct      |    39.7 |                                                          N/A | 29.9 |   15 | 30.7 | 45.6 | 38.6 | 37.1 |
| llava_v1.5_13b            |    38.3 |                                                     36.3±0.2 | 28.8 | 11.5 | 31.5 |   42 | 23.1 |   23 |
| mPLUG-Owl2                |    35.7 |                                                     36.3±0.1 | 29.5 |  7.7 | 32.1 | 47.3 | 23.8 | 20.9 |
| XComposer                 |    35.2 |                                                          N/A | 21.8 |  3.8 | 24.7 | 43.1 | 28.9 | 27.5 |
| sharegpt4v_7b             |    34.7 |                                                         37.6 | 30.2 | 18.5 |   30 | 36.1 | 20.2 | 18.1 |
| TransCore_M               |    33.9 |                                                          N/A | 27.3 | 15.4 | 32.7 | 36.7 |   23 | 23.5 |
| instructblip_7b           |    33.1 |                                                     26.2±0.2 | 25.5 | 11.5 | 23.5 | 39.3 | 24.3 | 23.6 |
| llava_v1.5_7b             |    32.7 |                                                     31.1±0.2 |   25 |  7.7 | 26.3 | 36.9 |   22 | 21.5 |
| instructblip_13b          |    30.1 |                                                     25.6±0.3 | 25.4 | 11.2 | 26.9 | 33.4 |   19 | 18.2 |
| idefics_9b_instruct       |      30 |                                                          N/A | 21.7 | 11.5 | 22.4 | 34.6 | 27.4 | 26.9 |
| llava_v1_7b (vicuna-v1.1) |    27.4 |                                                     23.8±0.6 |   19 | 11.5 | 25.6 | 31.4 | 18.1 | 16.2 |
| flamingov2                |    23.3 |                                                     24.8±0.2 | 19.5 |  7.7 | 21.7 | 24.7 | 21.7 |   19 |
| PandaGPT_13B              |    19.6 |                                                          N/A |  6.8 |  6.5 | 16.5 | 26.3 | 13.7 | 13.9 |
| MiniGPT-4-v1-13B          |    16.9 |                                                     24.4±0.4 | 10.3 |  7.7 | 12.5 | 19.9 | 14.9 | 13.8 |
| MiniGPT-4-v1-7B           |    15.6 |                                                     22.1±0.1 |  9.2 |  3.8 | 10.1 | 19.4 | 13.3 | 12.5 |
| VisualGLM_6b              |    14.8 |                                                          N/A |  8.5 |  6.5 |  9.1 |   18 |  8.1 |  7.1 |
| qwen_base                 |      13 |                                                          N/A |  7.4 |    0 |  3.9 | 16.5 | 18.6 | 18.1 |
| MiniGPT-4-v2              |    10.5 |                                                          N/A |  7.1 |  7.3 |  9.6 | 12.2 |  9.2 |    8 |