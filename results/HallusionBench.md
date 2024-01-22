# HallusionBench Results

[**HallusionBench**](https://github.com/tianyi-lab/HallusionBench) is a benchmark to evaluate hallucination of VLMs. It asks a set of visual questions with one original image and one modified image (the answers for a question can be different, considering the image content). 

**Examples in HallusionBench:**

| Original Figure                                              | Modified Figure                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](https://github-production-user-asset-6210df.s3.amazonaws.com/34324155/293858612-f7f378db-d8d7-47ec-a53a-37ec9649a321.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240103%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240103T075209Z&X-Amz-Expires=300&X-Amz-Signature=6410617bdaf21fc8bebf42382bd1ff73a1534b1a1d5da35cba8dd55f3878d172&X-Amz-SignedHeaders=host&actor_id=34324155&key_id=0&repo_id=477074140) | ![](https://github-production-user-asset-6210df.s3.amazonaws.com/34324155/293858628-f6b8a0d4-0cf0-4f8a-8a18-b6ad45dd792f.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240103%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240103T075259Z&X-Amz-Expires=300&X-Amz-Signature=b53d8554054592d47994d485d20adb7f61888600be60c01786a25168e3a07fa2&X-Amz-SignedHeaders=host&actor_id=34324155&key_id=0&repo_id=477074140) |
| **Q1.** Is the right orange circle the same size as the left orange circle? **A1. Yes** | **Q1.** Is the right orange circle the same size as the left orange circle? **A1. No** |
| **Q2.** Is the right orange circle larger than the left orange circle? **A2. No** | **Q2.** Is the right orange circle larger than the left orange circle? **A2. Yes** |
| **Q3.** Is the right orange circle smaller than the left orange circle? **A3. No** | **Q3.** Is the right orange circle smaller than the left orange circle? **A3. No** |

**Metrics: **

>-  aAcc: The overall accuracy of **all** atomic questions. 
>
>- qAcc: The mean accuracy of unique **questions**. One question can be asked multiple times with different figures, we consider VLM correctly solved a unique question only if it succeeds in all <question, figure> pairs for this unique question.
>- fAcc: The mean accuracy of all **figures**. One figure is associated with multiple questions, we consider VLM correct on a figure only if it succeeds to solve all questions of this figure. 

**Evaluation Setting: **

> 1. **No-visual** Questions (questions asked without the associated figure) in HallusionBench are **skipped** during evaluation.
> 2. When we failed to extract Yes / No from the VLM prediction, we adopt **GPT-3.5-Turbo-0613** as the answer extractor.
> 3. We report aAcc, qAcc, and fAcc for all evaluated VLMs. 

## Evaluation Results

> Models are sorted by the **descending order of qAcc.** 


| Model                       |   aAcc |   fAcc |   qAcc |
|:----------------------------|-------:|-------:|-------:|
| GPT-4v (detail: low)        |   65.8 |   38.4 |   35.2 |
| GeminiProVision             |   63.9 |   37.3 |   34.3 |
| Monkey-Chat                 |   58.4 |   30.6 |   29   |
| Qwen-VL-Chat                |   56.4 |   27.7 |   26.4 |
| MiniGPT-4-v1-7B             |   52.4 |   17.3 |   25.9 |
| Monkey                      |   55.1 |   24   |   25.5 |
| CogVLM-17B-Chat             |   55.1 |   26.3 |   24.8 |
| MiniGPT-4-v1-13B            |   51.3 |   16.2 |   24.6 |
| InternLM-XComposer-VL       |   57   |   26.3 |   24.6 |
| SharedCaptioner             |   55.6 |   22.8 |   24.2 |
| MiniGPT-4-v2                |   52.6 |   16.5 |   21.1 |
| InstructBLIP-7B             |   53.6 |   20.2 |   19.8 |
| Qwen-VL                     |   57.6 |   12.4 |   19.6 |
| OpenFlamingo v2             |   52.7 |   17.6 |   18   |
| EMU2-Chat                   |   49.4 |   22.3 |   16.9 |
| mPLUG-Owl2                  |   48.9 |   22.5 |   16.7 |
| ShareGPT4V-13B              |   49.8 |   21.7 |   16.7 |
| VisualGLM                   |   47.2 |   11.3 |   16.5 |
| TransCore-M                 |   49.7 |   21.4 |   15.8 |
| IDEFICS-9B-Instruct         |   50.1 |   16.2 |   15.6 |
| ShareGPT4V-7B               |   48.2 |   21.7 |   15.6 |
| LLaVA-InternLM-7B (QLoRA)   |   49.1 |   22.3 |   15.4 |
| InstructBLIP-13B            |   47.9 |   17.3 |   15.2 |
| LLaVA-InternLM2-20B (QLoRA) |   47.7 |   17.1 |   14.3 |
| LLaVA-v1.5-13B (QLoRA)      |   46.9 |   17.6 |   14.1 |
| LLaVA-v1.5-7B               |   48.3 |   19.9 |   14.1 |
| LLaVA-v1.5-7B (QLoRA)       |   46.2 |   16.2 |   13.2 |
| LLaVA-v1.5-13B              |   46.7 |   17.3 |   13   |
| IDEFICS-80B-Instruct        |   46.1 |   13.3 |   11   |
| LLaVA-v1-7B                 |   44.1 |   13.6 |    9.5 |
| PandaGPT-13B                |   43.1 |    9.2 |    7.7 |