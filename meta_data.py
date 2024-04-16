# CONSTANTS-URL
URL = "http://opencompass.openxlab.space/utils/OpenVLM.json"
VLMEVALKIT_README = 'https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/README.md'
# CONSTANTS-CITATION
CITATION_BUTTON_TEXT = r"""@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}"""
CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
# CONSTANTS-TEXT
LEADERBORAD_INTRODUCTION = """# OpenVLM Leaderboard
### Welcome to the OpenVLM Leaderboard! On this leaderboard we share the evaluation results of VLMs obtained by the OpenSource Framework:
### [*VLMEvalKit*: A Toolkit for Evaluating Large Vision-Language Models](https://github.com/open-compass/VLMEvalKit) 🏆
### Currently, OpenVLM Leaderboard covers {} different VLMs (including GPT-4v, Gemini, QwenVLPlus, LLaVA, etc.) and {} different multi-modal benchmarks. 

This leaderboard was last updated: {}. 
"""
# CONSTANTS-FIELDS
META_FIELDS = ['Method', 'Parameters (B)', 'Language Model', 'Vision Model', 'OpenSource', 'Verified']
MAIN_FIELDS = [
    'MMBench_TEST_EN', 'MMBench_TEST_CN', 'MMStar', 'MME'
    'MMMU_VAL', 'MathVista', 'HallusionBench', 'AI2D', 'OCRBench',
    'CCBench', 'SEEDBench_IMG', 'MMVet', 'LLaVABench'
]
MMBENCH_FIELDS = ['MMBench_TEST_EN', 'MMBench_DEV_EN', 'MMBench_TEST_CN', 'MMBench_DEV_CN', 'CCBench']
MODEL_SIZE = ['<10B', '10B-20B', '20B-40B', '>40B', 'Unknown']
MODEL_TYPE = ['API', 'OpenSource', 'Proprietary']

# The README file for each benchmark
LEADERBOARD_MD = {}

LEADERBOARD_MD['MAIN'] = """
## Main Evaluation Results

- Avg Score: The average score on all VLM Benchmarks (normalized to 0 - 100, the higher the better). 
- Avg Rank: The average rank on all VLM Benchmarks (the lower the better). 
- The overall evaluation results on 10 VLM benchmarks, sorted by the ascending order of Avg Rank. 
"""

LEADERBOARD_MD['SEEDBench_IMG'] = """
## SEEDBench_IMG Scores (ChatGPT Answer Extraction / Official Leaderboard)

- **Overall**: The overall accuracy across all questions with **ChatGPT answer matching**.
- **Overall (official)**: SEEDBench_IMG acc on the official leaderboard (if applicable). 
"""

LEADERBOARD_MD['MMVet'] = """
## MMVet Evaluation Results

- In MMVet Evaluation, we use GPT-4-Turbo (gpt-4-1106-preview) as the judge LLM to assign scores to the VLM outputs. We only perform the evaluation once due to the limited variance among results of multiple evaluation pass originally reported. 
- No specific prompt template adopted for **ALL VLMs**.
- We also provide performance on the [**Official Leaderboard**](https://paperswithcode.com/sota/visual-question-answering-on-mm-vet) for models that are applicable. Those results are obtained with GPT-4-0314 evaluator (which has been deperacted for new users).  
"""

LEADERBOARD_MD['MMMU_VAL'] = """
## MMMU Validation Evaluation Results

- For MMMU, we support the evaluation of the `dev` (150 samples) and `validation` (900 samples) set. Here we only report the results on the `validation` set. 
- **Answer Inference:**
  - For models with `interleave_generate` interface (accept interleaved images & texts as inputs), all testing samples can be inferred. **`interleave_generate` is adopted for inference.**
  - For models without `interleave_generate` interface, samples with more than one images are skipped (42 out of 1050, directly count as wrong). **`generate` is adopted for inference.**
- **Evaluation**:
  - MMMU include two types of questions: **multi-choice questions** & **open-ended QA**. 
  - For **open-ended QA (62/1050)**, we re-formulate it as multi-choice questions: `{'question': 'QQQ', 'answer': 'AAA'} -> {'question': 'QQQ', 'A': 'AAA', 'B': 'Other Answers', 'answer': 'A'}`, and then adopt the same evaluation paradigm for **multi-choice questions**. 
  - For **multi-choice questions (988/1050)**, we use **GPT-3.5-Turbo-0613** for matching prediction with options if heuristic matching does not work. 
"""

LEADERBOARD_MD['MathVista'] = """
## MMMU TestMini Evaluation Results

- We report the evaluation results on MathVista **TestMini**, which include 1000 test samples. 
- We adopt `GPT-4-Turbo (1106)` as the answer extractor when we failed to extract the answer with heuristic matching. 
- The performance of **Human  (High school)** and **Random Choice** are copied from the official leaderboard. 
**Category Definitions:** **FQA:** figure QA, **GPS:** geometry problem solving, **MWP:** math word problem, **TQA:** textbook QA, **VQA:** visual QA, **ALG:** algebraic, **ARI:** arithmetic, **GEO:** geometry, **LOG:** logical , **NUM:** numeric, **SCI:** scientific, **STA:** statistical.
"""

LEADERBOARD_MD['HallusionBench'] = """
[**HallusionBench**](https://github.com/tianyi-lab/HallusionBench) is a benchmark to evaluate hallucination of VLMs. It asks a set of visual questions with one original image and one modified image (the answers for a question can be different, considering the image content). 

**Examples in HallusionBench:**

| Original Figure                                              | Modified Figure                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](http://opencompass.openxlab.space/utils/Hallu0.png) | ![](http://opencompass.openxlab.space/utils/Hallu1.png) |
| **Q1.** Is the right orange circle the same size as the left orange circle? **A1. Yes** | **Q1.** Is the right orange circle the same size as the left orange circle? **A1. No** |
| **Q2.** Is the right orange circle larger than the left orange circle? **A2. No** | **Q2.** Is the right orange circle larger than the left orange circle? **A2. Yes** |
| **Q3.** Is the right orange circle smaller than the left orange circle? **A3. No** | **Q3.** Is the right orange circle smaller than the left orange circle? **A3. No** |

**Metrics**:

>-  aAcc: The overall accuracy of **all** atomic questions. 
>
>- qAcc: The mean accuracy of unique **questions**. One question can be asked multiple times with different figures, we consider VLM correctly solved a unique question only if it succeeds in all <question, figure> pairs for this unique question.
>- fAcc: The mean accuracy of all **figures**. One figure is associated with multiple questions, we consider VLM correct on a figure only if it succeeds to solve all questions of this figure. 

**Evaluation Setting**:

> 1. **No-visual** Questions (questions asked without the associated figure) in HallusionBench are **skipped** during evaluation.
> 2. When we failed to extract Yes / No from the VLM prediction, we adopt **GPT-3.5-Turbo-0613** as the answer extractor.
> 3. We report aAcc, qAcc, and fAcc for all evaluated VLMs. 

## HallusionBench Evaluation Results
"""

LEADERBOARD_MD['LLaVABench'] = """
## LLaVABench Evaluation Results

- In LLaVABench Evaluation, we use GPT-4-Turbo (gpt-4-1106-preview) as the judge LLM to assign scores to the VLM outputs. We only perform the evaluation once due to the limited variance among results of multiple evaluation pass originally reported. 
- No specific prompt template adopted for **ALL VLMs**.
- We also include the official results (obtained by gpt-4-0314) for applicable models. 
"""

LEADERBOARD_MD['COCO_VAL'] = """
## COCO Caption Results

-  By default, we evaluate COCO Caption Validation set (5000 samples), and report the following metrics: `BLEU-1, BLEU-4, CIDEr, ROUGE-L
-  We use the following prompt to evaluate all VLMs: `Please describe this image in general. Directly provide the description, do not include prefix like "This image depicts". `
- **No specific prompt is adopted for all VLMs.**
"""

LEADERBOARD_MD['ScienceQA_VAL'] = """
## ScienceQA Evaluation Results

- We benchmark the **image** subset of ScienceQA validation and test set, and report the Top-1 accuracy. 
- During evaluation, we use `GPT-3.5-Turbo-0613` as the choice extractor for all VLMs if the choice can not be extracted via heuristic matching. **Zero-shot** inference is adopted. 
"""

LEADERBOARD_MD['ScienceQA_TEST'] = LEADERBOARD_MD['ScienceQA_VAL']

LEADERBOARD_MD['OCRBench'] = """
## OCRBench Evaluation Results

- The evaluation of OCRBench is implemented by the official team: https://github.com/Yuliang-Liu/MultimodalOCR. 
- The performance of GPT4V might be underestimated: GPT4V rejects to answer 12 percent of the questions due to the policy of OpenAI. For those questions, the returned answer is "Your input image may contain content that is not allowed by our safety system."
"""