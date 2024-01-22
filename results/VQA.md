# VQA Results

> **Warning:** Currently, we only provide the **preliminary results**, obtained by checking if the prediction of a VLM ***exactly* matches** some reference answers. The accuracy numbers may not faithfully reflect the real performance of the VLM on the corresponding task. We will further provide more results based on ChatGPT matching, stay tuned. 
>
> - **Zero-Shot** Inference is adopted for all VLMs.

## **Supported Datasets:**

1. **OCRVQA**:	
   1. **OCRVQA_TESTCORE**: 3072 QAs sampled from the OCRVQA test set. There exist 32 categories (`Arts & Photography`, `Children's Books`, `Computers & Technology`, etc.), and we sample 96 cases from each category. 
   2. **OCRVQA_TEST**: The test set of OCRVQA, contains approximately 100k QAs. VLMEvalKit supports evaluating this dataset, but due to the large scale, we do not provide comprehensive evaluation results. 
2. **TextVQA_VAL**: The validation set of TextVQA, which includes around 5000 QAs.
3. **ChartVQA_VALTEST_HUMAN**: The validation and test sets of **ChartVQA**: 960 QAs for validation, 1250 QAs for testing. 

## The Preliminary Results:

| Model                         |   OCRVQA_TESTCORE |   TextVQA_VAL |   ChartQA_VAL |   ChartQA_TEST |
|:------------------------------|------------------:|--------------:|--------------:|---------------:|
| InternLM-XComposer-VL         |              54.5 |          38.5 |           9.1 |            9.5 |
| SharedCaptioner               |              50.2 |          38   |          12.3 |           10.6 |
| Qwen-VL                       |              63.5 |          12.7 |          14.6 |           14.6 |
| Qwen-VL-Chat                  |              57.7 |          10.5 |          16.2 |           14.6 |
| InstructBLIP-13B              |              53   |          32   |           6.6 |            6.5 |
| InstructBLIP-7B               |              52.9 |          32.1 |           6.1 |            6   |
| GeminiProVision               |              12.3 |          36.9 |          13   |           14.5 |
| TransCore-M                   |              58.1 |           3.3 |           0   |            0.2 |
| LLaVA-v1.5-13B (QLoRA, XTuner) |              51.5 |           3.1 |           0.1 |            0.2 |
| LLaVA-v1.5-13B                |              38.7 |           2.1 |           0.1 |            0.2 |
| LLaVA-InternLM-7B (QLoRA)      |              39   |           1.3 |           0.3 |            0.4 |
| LLaVA-v1.5-7B (QLoRA, XTuner)  |              36.7 |           2.5 |           0   |            0.2 |
| OpenFlamingo v2               |               8.9 |          17.8 |           5.9 |            4.9 |
| ShareGPT4V-7B                 |              34.7 |           0.9 |           0   |            0.1 |
| LLaVA-v1.5-7B                 |              32.7 |           1   |           0   |            0   |
| mPLUG-Owl2                    |              18.7 |           0   |           0.1 |            0.2 |
| IDEFICS-9B-Instruct           |               0   |           1   |           0.1 |            0.2 |
| MiniGPT-4-v1-13B              |               0.5 |           0.5 |           0.2 |            0.1 |
| MiniGPT-4-v1-7B               |               0.3 |           0.6 |           0.1 |            0.1 |
| IDEFICS-80B-Instruct          |               0   |           0   |           0   |            0.1 |
| CogVLM-17B-Chat               |               0   |           0   |           0   |            0   |
| GPT-4v (detail: low)          |               0   |           0   |           0   |            0   |
| LLaVA-v1-7B                   |               0   |           0   |           0   |            0   |
| VisualGLM                     |               0   |           0   |           0   |            0   |
| MiniGPT-4-v2                  |               0   |           0   |           0   |            0   |
| PandaGPT-13B                  |               0   |           0   |           0   |            0   |


