# Caption Results

## COCO Caption

> By default, we evaluate COCO Caption Validation set (5000 samples), and report the following metrics: `BLEU-1, BLEU-4, CIDEr, ROUGE-L
>
> We use the following prompt to evaluate all VLMs: `Please describe this image in general. Directly provide the description, do not include prefix like "This image depicts". `
>
> **No specific prompt is adopted for all VLMs.**

### Evaluation Results

| Model                         |   BLEU-4 |   BLEU-1 |   ROUGE-L |   CIDEr |   Word_cnt mean. |   Word_cnt std. |
|:------------------------------|---------:|---------:|----------:|--------:|-----------------:|----------------:|
| Qwen-VL-Chat                  |     34   |     75.8 |      54.9 |    98.9 |             10   |             1.7 |
| IDEFICS-80B-Instruct          |     32.5 |     76.1 |      54.1 |    94.9 |              9.7 |             3.2 |
| IDEFICS-9B-Instruct           |     29.4 |     72.7 |      53.4 |    90.4 |             10.5 |             4.4 |
| InstructBLIP-7B               |     20.9 |     56.8 |      39.9 |    58.1 |             11.6 |             5.9 |
| InstructBLIP-13B              |     16.9 |     50   |      37   |    52.4 |             11.8 |            12.8 |
| InternLM-XComposer-VL         |     12.4 |     38.3 |      37.9 |    41   |             26.3 |            22.2 |
| TransCore-M                   |      8.8 |     30.3 |      36.1 |    34.7 |             39.9 |            27.9 |
| GeminiProVision               |      8.4 |     33.2 |      31.2 |     9.7 |             35.2 |            15.7 |
| LLaVA-v1.5-7B (LoRA, XTuner)  |      7.2 |     25   |      36.6 |    43.2 |             48.8 |            42.9 |
| mPLUG-Owl2                    |      7.1 |     25.8 |      33.6 |    35   |             45.8 |            32.1 |
| LLaVA-v1-7B                   |      6.7 |     27.3 |      26.7 |     6.1 |             40.9 |            16.1 |
| VisualGLM                     |      5.4 |     28.6 |      23.6 |     0.2 |             41.5 |            11.5 |
| LLaVA-v1.5-13B (LoRA, XTuner) |      5.3 |     19.6 |      25.8 |    17.8 |             72.2 |            39.4 |
| LLaVA-v1.5-13B                |      5.1 |     20.7 |      21.2 |     0.3 |             70.6 |            22.3 |
| LLaVA-v1.5-7B                 |      4.6 |     19.6 |      19.9 |     0.1 |             72.5 |            21.7 |
| PandaGPT-13B                  |      4.6 |     19.9 |      19.3 |     0.1 |             65.4 |            16.6 |
| MiniGPT-4-v1-13B              |      4.4 |     20   |      19.8 |     1.3 |             64.4 |            30.5 |
| MiniGPT-4-v1-7B               |      4.3 |     19.6 |      17.5 |     0.8 |             61.9 |            30.6 |
| LLaVA-InternLM-7B (LoRA)      |      4   |     17.3 |      17.2 |     0.1 |             82.3 |            21   |
| CogVLM-17B-Chat               |      3.6 |     21.3 |      20   |     0.1 |             56.2 |            13.7 |
| Qwen-VL                       |      3.5 |     11.6 |      30   |    41.1 |             46.6 |           105.2 |
| GPT-4v (detail: low)          |      3.3 |     18   |      18.1 |     0   |             77.8 |            20.4 |
| ShareGPT4V-7B                 |      1.4 |      9.7 |      10.6 |     0.1 |            147.9 |            45.4 |
| MiniGPT-4-v2                  |      1.4 |     12.6 |      13.3 |     0.1 |             83   |            27.1 |
| OpenFlamingo v2               |      1.3 |      6.4 |      15.8 |    14.9 |             60   |            81.9 |
| SharedCaptioner               |      1   |      8.8 |       9.2 |     0   |            164.2 |            31.6 |

We noticed that, VLMs that generate long image descriptions tend to achieve inferior scores under different caption metrics.

### Error Analysis & Case Study

TBD. 