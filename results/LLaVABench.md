# LLaVABench Evaluation Results

> - In LLaVABench Evaluation, we use GPT-4-Turbo (gpt-4-1106-preview) as the judge LLM to assign scores to the VLM outputs. We only perform the evaluation once due to the limited variance among results of multiple evaluation pass originally reported. 
> - No specific prompt template adopted for **ALL VLMs**.
> - We also include the official results (obtained by gpt-4-0314) for applicable models. 


## Results

| Model                       |   complex |   detail |   conv |   overall | overall (GPT-4-0314)   |
|:----------------------------|----------:|---------:|-------:|----------:|:-----------------------|
| GPT-4v (detail: low)        |     102.3 |     90.1 |   82.2 |      93.1 | N/A                    |
| GeminiProVision             |      78.8 |     67.3 |   90.8 |      79.9 | N/A                    |
| CogVLM-17B-Chat             |      77.6 |     67.3 |   73.6 |      73.9 | N/A                    |
| QwenVLPlus                  |      77.1 |     65.7 |   74.3 |      73.7 | N/A                    |
| Qwen-VL-Chat                |      71.6 |     54.9 |   71.5 |      67.7 | N/A                    |
| ShareGPT4V-13B              |      70.1 |     58.3 |   66.9 |      66.6 | 72.6                   |
| ShareGPT4V-7B               |      69.5 |     56   |   64.5 |      64.9 | N/A                    |
| LLaVA-v1.5-13B              |      70.3 |     52.5 |   66.2 |      64.6 | 70.7                   |
| LLaVA-InternLM2-20B (QLoRA) |      70.5 |     50.8 |   63.6 |      63.7 | N/A                    |
| LLaVA-v1.5-13B (QLoRA)      |      77   |     52.9 |   53.8 |      63.6 | N/A                    |
| TransCore-M                 |      60.6 |     57.4 |   66   |      61.7 | N/A                    |
| LLaVA-v1.5-7B               |      68.6 |     51.7 |   56   |      60.7 | 63.4                   |
| InstructBLIP-7B             |      59.3 |     48.2 |   69.3 |      59.8 | 60.9                   |
| LLaVA-InternLM-7B (QLoRA)   |      61.3 |     52.8 |   62.7 |      59.7 | N/A                    |
| LLaVA-v1-7B                 |      67.6 |     43.8 |   58.7 |      58.9 | N/A                    |
| LLaVA-v1.5-7B (QLoRA)       |      64.4 |     45.2 |   56.2 |      57.2 | N/A                    |
| IDEFICS-80B-Instruct        |      57.7 |     49.6 |   61.7 |      56.9 | N/A                    |
| EMU2-Chat                   |      48.5 |     38.4 |   82.2 |      56.4 | N/A                    |
| InternLM-XComposer-VL       |      61.7 |     52.5 |   44.3 |      53.8 | N/A                    |
| InstructBLIP-13B            |      57.3 |     41.7 |   56.9 |      53.5 | 58.2                   |
| SharedCaptioner             |      38.3 |     44.2 |   62.1 |      47.4 | N/A                    |
| MiniGPT-4-v1-13B            |      57.3 |     44.4 |   32.5 |      46.2 | N/A                    |
| MiniGPT-4-v1-7B             |      48   |     44.4 |   41.4 |      45.1 | N/A                    |
| IDEFICS-9B-Instruct         |      49.3 |     45.2 |   38.6 |      45   | N/A                    |
| Monkey                      |      30.3 |     29.1 |   74.5 |      43.1 | N/A                    |
| VisualGLM                   |      43.6 |     36.6 |   28.4 |      37.3 | N/A                    |
| PandaGPT-13B                |      46.2 |     29.2 |   31   |      37.1 | N/A                    |
| OpenFlamingo v2             |      26.1 |     29.6 |   50.3 |      34.2 | N/A                    |
| Monkey-Chat                 |      19.5 |     18.9 |   67.3 |      33.5 | N/A                    |
| MiniGPT-4-v2                |      36.8 |     15.5 |   28.9 |      28.8 | N/A                    |
| mPLUG-Owl2                  |      13   |     17.1 |   51   |      25   | N/A                    |
| Qwen-VL                     |       5.6 |     26.5 |   13.7 |      12.9 | N/A                    |