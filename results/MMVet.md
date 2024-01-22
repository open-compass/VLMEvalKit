# MMVet Evaluation Results

> - In MMVet Evaluation, we use GPT-4-Turbo (gpt-4-1106-preview) as the judge LLM to assign scores to the VLM outputs. We only perform the evaluation once due to the limited variance among results of multiple evaluation pass originally reported. 
> - No specific prompt template adopted for **ALL VLMs**.
> - We also provide performance on the [**Official Leaderboard**](https://paperswithcode.com/sota/visual-question-answering-on-mm-vet) for models that are applicable. Those results are obtained with GPT-4-0314 evaluator (which has been deperacted for new users).  

### MMVet Scores


| Model                       |   ocr |   math |   spat |   rec |   know |   gen |   Overall (GPT-4-Turbo) | Overall (Official)   |
|:----------------------------|------:|-------:|-------:|------:|-------:|------:|------------------------:|:---------------------|
| GeminiProVision             |  63.6 |   41.5 |   61.2 |  59.8 |   51   |  48   |                    59.2 | 64.3±0.4             |
| GPT-4v (detail: low)        |  59.4 |   61.2 |   52.5 |  59.7 |   48   |  46.5 |                    56.8 | 60.2±0.3             |
| QwenVLPlus                  |  59   |   45.8 |   48.7 |  58.4 |   49.2 |  49.3 |                    55.7 | N/A                  |
| CogVLM-17B-Chat             |  46.4 |   10.8 |   46.1 |  64.7 |   52.4 |  50.6 |                    54.5 | N/A                  |
| Qwen-VL-Chat                |  37.2 |   22.3 |   42.8 |  52.5 |   45.4 |  40.3 |                    47.3 | N/A                  |
| IDEFICS-80B-Instruct        |  29.9 |   15   |   30.7 |  45.6 |   38.6 |  37.1 |                    39.7 | N/A                  |
| ShareGPT4V-13B              |  37.3 |   18.8 |   39.1 |  45.2 |   23.7 |  22.4 |                    39.2 | 43.1                 |
| TransCore-M                 |  35.8 |   11.2 |   36.8 |  47.3 |   26.1 |  27.3 |                    38.8 | N/A                  |
| LLaVA-v1.5-13B              |  28.8 |   11.5 |   31.5 |  42   |   23.1 |  23   |                    38.3 | 36.3±0.2             |
| Monkey                      |  30.9 |    7.7 |   33.6 |  50.3 |   21.4 |  16.5 |                    37.5 | N/A                  |
| LLaVA-InternLM2-20B (QLoRA) |  30.6 |    7.7 |   32.4 |  49.1 |   29.2 |  31.9 |                    37.2 | N/A                  |
| LLaVA-v1.5-13B (QLoRA)      |  31.3 |   15   |   28   |  46.3 |   25.6 |  27.3 |                    35.9 | N/A                  |
| mPLUG-Owl2                  |  29.5 |    7.7 |   32.1 |  47.3 |   23.8 |  20.9 |                    35.7 | 36.3±0.1             |
| InternLM-XComposer-VL       |  21.8 |    3.8 |   24.7 |  43.1 |   28.9 |  27.5 |                    35.2 | N/A                  |
| ShareGPT4V-7B               |  30.2 |   18.5 |   30   |  36.1 |   20.2 |  18.1 |                    34.7 | 37.6                 |
| InstructBLIP-7B             |  25.5 |   11.5 |   23.5 |  39.3 |   24.3 |  23.6 |                    33.1 | 26.2±0.2             |
| Monkey-Chat                 |  26.9 |    3.8 |   28   |  44.5 |   17.3 |  13   |                    33   | N/A                  |
| LLaVA-v1.5-7B               |  25   |    7.7 |   26.3 |  36.9 |   22   |  21.5 |                    32.7 | 31.1±0.2             |
| LLaVA-InternLM-7B (QLoRA)   |  29.2 |    7.7 |   27.5 |  41.1 |   21.7 |  18.5 |                    32.4 | N/A                  |
| LLaVA-v1.5-7B (QLoRA)       |  28.2 |   11.5 |   26.8 |  41.1 |   21.7 |  17   |                    32.2 | N/A                  |
| EMU2-Chat                   |  30.1 |   11.5 |   31.7 |  38.4 |   16   |  11.7 |                    31   | 48.5                 |
| SharedCaptioner             |  26   |   11.2 |   31.1 |  39.5 |   17.1 |  12   |                    30.1 | N/A                  |
| InstructBLIP-13B            |  25.4 |   11.2 |   26.9 |  33.4 |   19   |  18.2 |                    30.1 | 25.6±0.3             |
| IDEFICS-9B-Instruct         |  21.7 |   11.5 |   22.4 |  34.6 |   27.4 |  26.9 |                    30   | N/A                  |
| LLaVA-v1-7B                 |  19   |   11.5 |   25.6 |  31.4 |   18.1 |  16.2 |                    27.4 | 23.8±0.6             |
| OpenFlamingo v2             |  19.5 |    7.7 |   21.7 |  24.7 |   21.7 |  19   |                    23.3 | 24.8±0.2             |
| PandaGPT-13B                |   6.8 |    6.5 |   16.5 |  26.3 |   13.7 |  13.9 |                    19.6 | N/A                  |
| MiniGPT-4-v1-13B            |  10.3 |    7.7 |   12.5 |  19.9 |   14.9 |  13.8 |                    16.9 | 24.4±0.4             |
| MiniGPT-4-v1-7B             |   9.2 |    3.8 |   10.1 |  19.4 |   13.3 |  12.5 |                    15.6 | 22.1±0.1             |
| VisualGLM                   |   8.5 |    6.5 |    9.1 |  18   |    8.1 |   7.1 |                    14.8 | N/A                  |
| Qwen-VL                     |   7.4 |    0   |    3.9 |  16.5 |   18.6 |  18.1 |                    13   | N/A                  |
| MiniGPT-4-v2                |   7.1 |    7.3 |    9.6 |  12.2 |    9.2 |   8   |                    10.5 | N/A                  |
