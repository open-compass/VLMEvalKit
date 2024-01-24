import json
import pandas as pd
from collections import defaultdict
import gradio as gr
import copy as cp
import numpy as np
from .misc import listinstr

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
### Welcome to the OpenVLM Leaderboard! On this leaderboard we share the evaluation results of VLMs obtained by the OpenSource Framework [**VLMEvalKit**](https://github.com/open-compass/VLMEvalKit) 🏆 
### Currently, OpenVLM Leaderboard covers {} different VLMs (including GPT-4v, Gemini, QwenVLPlus, LLaVA, etc.) and {} different multi-modal benchmarks. 

This leaderboard was last updated: {}. 
"""
# CONSTANTS-FIELDS
META_FIELDS = ['Method', 'Parameters (B)', 'Language Model', 'Vision Model', 'OpenSource', 'Verified']
MAIN_FIELDS = ['MMBench_TEST_EN', 'MMBench_TEST_CN', 'CCBench', 'MME', 'SEEDBench_IMG', 'MMVet', 'MMMU_VAL', 'MathVista', 'HallusionBench', 'LLaVABench']
MMBENCH_FIELDS = ['MMBench_TEST_EN', 'MMBench_DEV_EN', 'MMBench_TEST_CN', 'MMBench_DEV_CN', 'CCBench']
MODEL_SIZE = ['<10B', '10B-20B', '20B-40B', '>40B', 'Unknown']
MODEL_TYPE = ['API', 'OpenSource', 'Proprietary']

LEADERBOARD_MD = {
    'MAIN': """
## Main Evaluation Results

- Avg Score: The average score on all VLM Benchmarks (normalized to 0 - 100, the higher the better). 
- Avg Rank: The average rank on all VLM Benchmarks (the lower the better). 
- The overall evaluation results on 10 VLM benchmarks, sorted by the ascending order of Avg Rank. 
""",
    'SEEDBench_IMG': """
## SEEDBench_IMG Scores (Prefetch / ChatGPT Answer Extraction / Official Leaderboard)

- **Overall**: The overall accuracy across all questions with **ChatGPT answer matching**.
- **Overall (prefetch)**: The accuracy when using exact matching for evaluation. 
- **Overall (official)**: SEEDBench_IMG acc on the official leaderboard (if applicable). 
""",
    'MMVet': """
## MMVet Evaluation Results

- In MMVet Evaluation, we use GPT-4-Turbo (gpt-4-1106-preview) as the judge LLM to assign scores to the VLM outputs. We only perform the evaluation once due to the limited variance among results of multiple evaluation pass originally reported. 
- No specific prompt template adopted for **ALL VLMs**.
- We also provide performance on the [**Official Leaderboard**](https://paperswithcode.com/sota/visual-question-answering-on-mm-vet) for models that are applicable. Those results are obtained with GPT-4-0314 evaluator (which has been deperacted for new users).  
""",
    'MMMU_VAL': """
## MMMU Validation Evaluation Results

- For MMMU, we support the evaluation of the `dev` (150 samples) and `validation` (900 samples) set. Here we only report the results on the `validation` set. 
- **Answer Inference:**
  - For models with `interleave_generate` interface (accept interleaved images & texts as inputs), all testing samples can be inferred. **`interleave_generate` is adopted for inference.**
  - For models without `interleave_generate` interface, samples with more than one images are skipped (42 out of 1050, directly count as wrong). **`generate` is adopted for inference.**
- **Evaluation**:
  - MMMU include two types of questions: **multi-choice questions** & **open-ended QA**. 
  - For **open-ended QA (62/1050)**, we re-formulate it as multi-choice questions: `{'question': 'QQQ', 'answer': 'AAA'} -> {'question': 'QQQ', 'A': 'AAA', 'B': 'Other Answers', 'answer': 'A'}`, and then adopt the same evaluation paradigm for **multi-choice questions**. 
  - For **multi-choice questions (988/1050)**, we use **GPT-3.5-Turbo-0613** for matching prediction with options if heuristic matching does not work. 
""",
    'MathVista': """
## MMMU TestMini Evaluation Results

- We report the evaluation results on MathVista **TestMini**, which include 1000 test samples. 
- We adopt `GPT-4-Turbo (1106)` as the answer extractor when we failed to extract the answer with heuristic matching. 
- The performance of **Human  (High school)** and **Random Choice** are copied from the official leaderboard. 
**Category Definitions:** **FQA:** figure QA, **GPS:** geometry problem solving, **MWP:** math word problem, **TQA:** textbook QA, **VQA:** visual QA, **ALG:** algebraic, **ARI:** arithmetic, **GEO:** geometry, **LOG:** logical , **NUM:** numeric, **SCI:** scientific, **STA:** statistical.
""",
    'HallusionBench': """
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
""",
    'LLaVABench': """
## LLaVABench Evaluation Results

- In LLaVABench Evaluation, we use GPT-4-Turbo (gpt-4-1106-preview) as the judge LLM to assign scores to the VLM outputs. We only perform the evaluation once due to the limited variance among results of multiple evaluation pass originally reported. 
- No specific prompt template adopted for **ALL VLMs**.
- We also include the official results (obtained by gpt-4-0314) for applicable models. 
"""
}

from urllib.request import urlopen

def load_results():
    data = json.loads(urlopen(URL).read())
    return data

def nth_large(val, vals):
    return sum([1 for v in vals if v > val]) + 1

def format_timestamp(timestamp):
    return timestamp[:2] + '.' + timestamp[2:4] + '.' + timestamp[4:6] + ' ' + timestamp[6:8] + ':' + timestamp[8:10] + ':' + timestamp[10:12]

def model_size_flag(sz, FIELDS):
    if pd.isna(sz) and 'Unknown' in FIELDS:
        return True
    if pd.isna(sz):
        return False
    if '<10B' in FIELDS and sz < 10:
        return True
    if '10B-20B' in FIELDS and sz >= 10 and sz < 20:
        return True
    if '20B-40B' in FIELDS and sz >= 20 and sz < 40:
        return True
    if '>40B' in FIELDS and sz >= 40:
        return True
    return False

def model_type_flag(line, FIELDS):
    if 'OpenSource' in FIELDS and line['OpenSource'] == 'Yes':
        return True
    if 'API' in FIELDS and line['OpenSource'] == 'No' and line['Verified'] == 'Yes':
        return True
    if 'Proprietary' in FIELDS and line['OpenSource'] == 'No' and line['Verified'] == 'No':
        return True
    return False

def BUILD_L1_DF(results, fields):
    res = defaultdict(list)
    for i, m in enumerate(results):
        item = results[m]
        meta = item['META']
        for k in META_FIELDS:
            if k == 'Parameters (B)':
                param = meta['Parameters']
                res[k].append(float(param.replace('B', '')) if param != '' else None)
            elif k == 'Method':
                name, url = meta['Method']
                res[k].append(f'<a href="{url}">{name}</a>')
            else:
                res[k].append(meta[k])
        scores, ranks = [], []
        for d in fields:
            res[d].append(item[d]['Overall'])
            scores.append(item[d]['Overall'])
            ranks.append(nth_large(item[d]['Overall'], [x[d]['Overall'] for x in results.values()]))
        res['Avg Score'].append(round(np.mean(scores), 1))
        res['Avg Rank'].append(round(np.mean(ranks), 2))

    df = pd.DataFrame(res)
    df = df.sort_values('Avg Rank')
    
    check_box = {}
    check_box['essential'] = ['Method', 'Parameters (B)', 'Language Model', 'Vision Model']
    check_box['required'] = ['Avg Score', 'Avg Rank']
    check_box['all'] = check_box['required'] + ['OpenSource', 'Verified'] + fields
    type_map = defaultdict(lambda: 'number')
    type_map['Method'] = 'html'
    type_map['Language Model'] = type_map['Vision Model'] = type_map['OpenSource'] = type_map['Verified'] = 'str'
    check_box['type_map'] = type_map
    return df, check_box
        
def BUILD_L2_DF(results, dataset):
    res = defaultdict(list)
    fields = list(list(results.values())[0][dataset].keys())
    non_overall_fields = [x for x in fields if 'Overall' not in x]
    overall_fields = [x for x in fields if 'Overall' in x]
    if dataset == 'MME':
        non_overall_fields = [x for x in non_overall_fields if not listinstr(['Perception', 'Cognition', 'Total'], x)]
        overall_fields = ['Perception', 'Cognition', 'Total']
    
    for m in results:
        item = results[m]
        meta = item['META']
        for k in META_FIELDS:
            if k == 'Parameters (B)':
                param = meta['Parameters']
                res[k].append(float(param.replace('B', '')) if param != '' else None)
            elif k == 'Method':
                name, url = meta['Method']
                res[k].append(f'<a href="{url}">{name}</a>')
            else:
                res[k].append(meta[k])
        fields = [x for x in fields]
    
        for d in non_overall_fields:
            res[d].append(item[dataset][d])
        for d in overall_fields:
            res[d].append(item[dataset][d])

    df = pd.DataFrame(res)
    df = df.sort_values('Overall' if dataset != 'MME' else 'Total')
    df = df.iloc[::-1]
    
    check_box = {}
    check_box['essential'] = ['Method', 'Parameters (B)', 'Language Model', 'Vision Model']
    check_box['required'] = overall_fields
    check_box['all'] = non_overall_fields + overall_fields
    type_map = defaultdict(lambda: 'number')
    type_map['Method'] = 'html'
    type_map['Language Model'] = type_map['Vision Model'] = type_map['OpenSource'] = type_map['Verified'] = 'str'
    check_box['type_map'] = type_map
    return df, check_box