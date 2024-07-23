# CONSTANTS-URL
URL = "http://opencompass.openxlab.space/assets/OpenVLM.json"
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

OpenVLM Leaderboard only includes open-source VLMs or API models that are publicly available. To add your own model to the leaderboard, please create a PR in [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to support your VLM and then we will help with the evaluation and updating the leaderboard. For any questions or concerns, please feel free to contact us at [opencompass, duanhaodong]@pjlab.org.cn.
"""
# CONSTANTS-FIELDS
META_FIELDS = ['Method', 'Param (B)', 'Language Model', 'Vision Model', 'OpenSource', 'Verified']
MAIN_FIELDS = [
    'MMBench_V11', 'MMStar', 'MME',
    'MMMU_VAL', 'MathVista', 'OCRBench', 'AI2D', 
    'HallusionBench', 'SEEDBench_IMG', 'MMVet', 
    'LLaVABench', 'CCBench', 'RealWorldQA', 'POPE', 'ScienceQA_TEST',
    'SEEDBench2_Plus', 'MMT-Bench_VAL', 'BLINK'
]
DEFAULT_BENCH = [
    'MMBench_V11', 'MMStar', 'MMMU_VAL', 'MathVista', 'OCRBench', 'AI2D', 
    'HallusionBench', 'MMVet'
]
MMBENCH_FIELDS = ['MMBench_TEST_EN_V11', 'MMBench_TEST_CN_V11', 'MMBench_TEST_EN', 'MMBench_TEST_CN', 'CCBench']
MODEL_SIZE = ['<4B', '4B-10B', '10B-20B', '20B-40B', '>40B', 'Unknown']
MODEL_TYPE = ['API', 'OpenSource', 'Proprietary']

# The README file for each benchmark
LEADERBOARD_MD = {}

LEADERBOARD_MD['MAIN'] = f"""
## Main Evaluation Results

- Metrics:
  - Avg Score: The average score on all VLM Benchmarks (normalized to 0 - 100, the higher the better). 
  - Avg Rank: The average rank on all VLM Benchmarks (the lower the better). 
  - Avg Score & Rank are calculated based on selected benchmark. **When results for some selected benchmarks are missing, Avg Score / Rank will be None!!!** 
- By default, we present the overall evaluation results based on {len(DEFAULT_BENCH)} VLM benchmarks, sorted by the descending order of Avg Score. 
  - The following datasets are included in the main results: {', '.join(DEFAULT_BENCH)}. 
  - Detailed evaluation results for each dataset (included or not included in main) are provided in the consequent tabs. 
"""

for dataset in ['MMBench_DEV_CN', 'MMBench_TEST_CN', 'MMBench_DEV_EN', 'MMBench_TEST_EN', 'CCBench']:
    LEADERBOARD_MD[dataset] = f"""
## {dataset.replace('_', ' ')} Evaluation Results  

- We adopt Circular Eval for benchmarks in MMBench series, you can check https://arxiv.org/pdf/2307.06281.pdf for the detailed definition of Circular Eval. 
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

-  By default, we evaluate COCO Caption Validation set (5000 samples), and report the following metrics: BLEU-1, BLEU-4, CIDEr, ROUGE-L (default sorted by CIDEr).
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

LEADERBOARD_MD['MMStar'] = """
## MMStar Evaluation Results

- MMStar is an elite vision-indispensable multi-modal benchmark, including 1,500 challenging samples meticulously selected by humans.
- During the evaluation of MMStar, we find that some API models may reject to answer some of the questions. Currently, we treat such cases as wrong answers when reporting the results. 
"""

LEADERBOARD_MD['RealWorldQA'] = """
## RealWorldQA Evaluation Results

- RealWorldQA is a benchmark designed to evaluate the real-world spatial understanding capabilities of multimodal AI models, contributed by XAI. It assesses how well these models comprehend physical environments. The benchmark consists of 700+ images, each accompanied by a question and a verifiable answer. These images are drawn from real-world scenarios, including those captured from vehicles. The goal is to advance AI models' understanding of our physical world.
"""

LEADERBOARD_MD['TextVQA_VAL'] = """
## TextVQA Evaluation Results

- TextVQA is a dataset to benchmark visual reasoning based on text in images. TextVQA requires models to read and reason about text in images to answer questions about them. Specifically, models need to incorporate a new modality of text present in the images and reason over it to answer TextVQA questions.
- Note that some models may not be able to generate standardized responses based on the prompt. We currently do not have reports for these models.
"""

LEADERBOARD_MD['ChartQA_TEST'] = """
## ChartQA Evaluation Results

- ChartQA is a benchmark for question answering about charts with visual and logical reasoning. 
- Note that some models may not be able to generate standardized responses based on the prompt. We currently do not have reports for these models.
"""

LEADERBOARD_MD['OCRVQA_TESTCORE'] = """
## OCRVQA Evaluation Results

- OCRVQA is a benchmark for visual question answering by reading text in images. It presents a large-scale dataset, OCR-VQA-200K, comprising over 200,000 images of book covers. The study combines techniques from the Optical Character Recognition (OCR) and Visual Question Answering (VQA) domains to address the challenges associated with this new task and dataset. 
- Note that some models may not be able to generate standardized responses based on the prompt. We currently do not have reports for these models.
"""

LEADERBOARD_MD['POPE'] = """
## POPE Evaluation Results

- POPE is a benchmark for object hallucination evaluation. It includes three tracks of object hallucination: random, popular, and adversarial.
- Note that the official POPE dataset contains approximately 8910 cases. POPE includes three tracks, and there are some overlapping samples among the three tracks. To reduce the data file size, we have kept only a single copy of the overlapping samples (about 5127 examples). However, the final accuracy will be calculated on the ~9k samples.
- Some API models, due to safety policies, refuse to answer certain questions, so their actual capabilities may be higher than the reported scores.
- We report the average F1 score across the three types of data as the overall score. Accuracy, precision, and recall are also shown in the table. F1 score = 2 * (precision * recall) / (precision + recall). 
"""

LEADERBOARD_MD['SEEDBench2_Plus'] = """
## SEEDBench2 Plus Evaluation Results

- SEEDBench2 Plus comprises 2.3K multiple-choice questions with precise human annotations, spanning three broad categories: Charts, Maps, and Webs, each of which covers a wide spectrum of textrich scenarios in the real world.
"""

LEADERBOARD_MD['MMT-Bench_VAL'] = """
## MMT-Bench Validation Evaluation Results

- MMT-Bench comprises 31,325 meticulously curated multi-choice visual questions from various multimodal scenarios such as vehicle driving and embodied navigation, covering 32 core meta-tasks and 162 subtasks in multimodal understanding.
- MMT-Bench_VAL is the validation set of MMT-Bench. MMT-Bench_ALL includes both validation and test sets. The suffix `MI`, such as `MMT-Bench_VAL_MI`, represents the multi-image version of the dataset with several images input. 
The defualt version is the single-image version, which concats the multiple images into a single image as input.
"""

LEADERBOARD_MD['SEEDBench2'] = """
## SEEDBench2 Evaluation Results

- SEEDBench2 comprises 24K multiple-choice questions with accurate human annotations, which spans 27 dimensions, including the evaluation of both text and image generation.
- Note that we only evaluate and report the part of model's results on the SEEDBench2.
"""

LEADERBOARD_MD['BLINK'] == """
## BLINK Test Evaluation Results

- BLINK is a benchmark containing 14 visual perception tasks that can be solved by humans “within a blink”, but pose significant challenges for current multimodal large language models (LLMs).
- We evaluate BLINK on the test set of the benchmark, which contains 1901 visual questions in multi-choice format. 
"""
