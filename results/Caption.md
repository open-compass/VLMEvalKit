# Caption Results

## COCO Caption

> By default, we evaluate COCO Caption Validation set (5000 samples), and report the following metrics: `BLEU-1, BLEU-4, CIDEr, ROUGE-L
>
> We use the following prompt to evaluate all VLMs: `Please describe this image in general. Directly provide the description, do not include prefix like "This image depicts". `
>
> **No specific prompt is adopted for all VLMs.**

### Evaluation Results

| Model                         | BLEU-4 | BLEU-1 | ROUGE-L | CIDEr |
| :---------------------------- | -----: | -----: | ------: | ----: |
| Qwen-VL-Chat                  |     34 |   75.8 |    54.9 |  98.9 |
| IDEFICS-80B-Instruct          |   32.5 |   76.1 |    54.1 |  94.9 |
| IDEFICS-9B-Instruct           |   29.4 |   72.7 |    53.4 |  90.4 |
| InstructBLIP-7B               |   20.9 |   56.8 |    39.9 |  58.1 |
| InstructBLIP-13B              |   16.9 |     50 |      37 |  52.4 |
| InternLM-XComposer-VL         |   12.4 |   38.3 |    37.9 |    41 |
| TransCore-M                   |    8.8 |   30.3 |    36.1 |  34.7 |
| GeminiProVision               |    8.4 |   33.2 |    31.2 |   9.7 |
| LLaVA-v1.5-7B (LoRA, XTuner)  |    7.2 |     25 |    36.6 |  43.2 |
| mPLUG-Owl2                    |    7.1 |   25.8 |    33.6 |    35 |
| LLaVA-v1-7B                   |    6.7 |   27.3 |    26.7 |   6.1 |
| VisualGLM                     |    5.4 |   28.6 |    23.6 |   0.2 |
| LLaVA-v1.5-13B (LoRA, XTuner) |    5.3 |   19.6 |    25.8 |  17.8 |
| LLaVA-v1.5-13B                |    5.1 |   20.7 |    21.2 |   0.3 |
| LLaVA-v1.5-7B                 |    4.6 |   19.6 |    19.9 |   0.1 |
| PandaGPT-13B                  |    4.6 |   19.9 |    19.3 |   0.1 |
| MiniGPT-4-v1-13B              |    4.4 |     20 |    19.8 |   1.3 |
| MiniGPT-4-v1-7B               |    4.3 |   19.6 |    17.5 |   0.8 |
| LLaVA-InternLM-7B (LoRA)      |      4 |   17.3 |    17.2 |   0.1 |
| Qwen-VL                       |    3.5 |   11.6 |      30 |  41.1 |
| GPT-4v (detail: low)          |    3.3 |     18 |    18.1 |     0 |
| ShareGPT4V-7B                 |    1.4 |    9.7 |    10.6 |   0.1 |
| MiniGPT-4-v2                  |    1.4 |   12.6 |    13.3 |   0.1 |
| OpenFlamingo v2               |    1.3 |    6.4 |    15.8 |  14.9 |

### Error Analysis & Case Study

TBD. 