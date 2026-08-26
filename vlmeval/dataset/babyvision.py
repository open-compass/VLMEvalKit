import json
import os.path as osp
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from vlmeval.smp import (LMUDataRoot, d2df, download_file, dump, get_intermediate_file_path, load,
                         md5)
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge

try:
    import regex
except ImportError:  # pragma: no cover - official BabyVision depends on regex.
    import re as regex


BABYVISION_DATASET = 'BabyVision'
BABYVISION_FINAL_ANSWER_PROMPT = 'Think about the question and give your final answer in \\boxed{Answer} format.'
BABYVISION_ZIP_URL = (
    'https://github.com/UniPat-AI/BabyVision/raw/'
    '7f92fd4b1dc1c68b7b936a9bc09c68b4a944a55a/data/babyvision_data.zip'
)
BABYVISION_ZIP_MD5 = '498d4a3dbdc33fa443b1cb2eee41ece1'

LLM_JUDGE_PROMPT = """You are a careful and strict evaluator. You will be given:

1. **Question**
2. **Ground Truth Answer** (correct answer)
3. **Model Output** (answer from another model)

**Your goal:** Determine if the Model Output **accurately matches** the Ground Truth Answer in meaning.

* Matching means: the facts, entities, and key details are equivalent, even if phrasing differs.
* Not matching means: the Model Output is wrong, incomplete, contains extra incorrect facts, or changes the meaning.

**Process (internal reasoning):**

1. Read and understand the Question, Ground Truth Answer, and Model Output.
2. Ignore small wording differences, formatting, or synonyms.
3. If all factual content matches, conclude `1`. Otherwise, conclude `0`.

**Important:**

* Think through your decision step-by-step **internally** before responding.
* In your final output, return **only** True or False, with no extra text or explanation.

**Output format:**

True

or

False

**Input:**

Question: {question},
Ground Truth Answer: {groundtruth},
Model Output: {modeloutput}
"""

TYPE_MATCH = {
    'Fine-grained Discrimination': '1',
    'Visual Tracking': '2',
    'Spatial Perception': '3',
    'Visual Pattern Recognition': '4',
}


def format_choices(choices):
    """Format multiple choice options as (A), (B), (C), etc."""
    if len(choices) == 0:
        return ''
    formatted = ''
    for idx, choice in enumerate(choices):
        formatted += f'({chr(65 + idx)}) {choice}\n'
    return formatted.strip()


def build_babyvision_question(item):
    if item['ansType'] == 'blank':
        question = item['question']
    else:
        question = item['question'] + '\nChoices:\n' + format_choices(item['options'])
    return question + '\n' + BABYVISION_FINAL_ANSWER_PROMPT


def build_babyvision_answer(item):
    if item['ansType'] == 'blank':
        return clean_babyvision_optional(item.get('blankAns', ''))
    return chr(65 + int(item['choiceAns']))


def clean_babyvision_optional(value):
    if value is None:
        return ''
    if isinstance(value, str) and value.lower() == 'null':
        return ''
    return value


def build_babyvision_judge_prompt(question, groundtruth, modeloutput):
    return LLM_JUDGE_PROMPT.format(
        question=question,
        groundtruth=groundtruth,
        modeloutput=modeloutput,
    )


def extract_boxed_answer(text):
    """
    Extract the content from the last \\boxed{} pattern.

    Also supports alternative format: <|begin_of_box|>...<|end_of_box|>

    Returns None if no pattern found.
    """
    if text is None:
        return None

    pattern = r'\\boxed\{((?:[^{}]|{(?:[^{}]|{.*})*})*)\}'
    matches = regex.findall(pattern, text)

    if matches:
        return matches[-1]

    pattern_alt = r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>'
    matches_alt = regex.findall(pattern_alt, text)
    if matches_alt:
        return matches_alt[-1].strip()

    return None


def parse_judge_response(response):
    judge_response_clean = str(response).strip().lower()
    if 'true' in judge_response_clean:
        return True
    if 'false' in judge_response_clean:
        return False
    return False


def _safe_model_name(model_name):
    return str(model_name).replace('/', '_').replace(':', '_')


class BabyVision(ImageBaseDataset):
    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    DEFAULT_JUDGE_MODEL = 'openai/gpt-5.2'
    DATASET_URL = {BABYVISION_DATASET: BABYVISION_ZIP_URL}
    DATASET_MD5 = {BABYVISION_DATASET: BABYVISION_ZIP_MD5}
    force_use_dataset_prompt = True

    def __init__(self, dataset=BABYVISION_DATASET, data_root=None, meta_jsonl=None, **kwargs):
        super().__init__(dataset=dataset, skip_noimg=False)

    @classmethod
    def supported_datasets(cls):
        return [BABYVISION_DATASET]

    @staticmethod
    def _data_root():
        return Path(LMUDataRoot()) / BABYVISION_DATASET

    def _meta_path(self):
        return self._data_root() / 'babyvision_data' / 'meta_data.jsonl'

    def _candidate_meta_paths(self):
        return [self._meta_path()]

    def _find_meta_path(self):
        seen = set()
        for path in self._candidate_meta_paths():
            path = path.expanduser()
            if path in seen:
                continue
            seen.add(path)
            if path.exists():
                return path
        return None

    def _record_from_item(self, item, image_path):
        options = item.get('options') or []
        return {
            'index': item['taskId'],
            'taskId': item['taskId'],
            'question': build_babyvision_question(item),
            'raw_question': item['question'],
            'image_path': str(image_path),
            'answer': build_babyvision_answer(item),
            'ansType': item['ansType'],
            'options': json.dumps(options, ensure_ascii=False),
            'choiceAns': clean_babyvision_optional(item.get('choiceAns', '')),
            'blankAns': clean_babyvision_optional(item.get('blankAns', '')),
            'Type': item['type'],
            'Subtype': item['subtype'],
            'status': clean_babyvision_optional(item.get('status', '')),
            'coT': clean_babyvision_optional(item.get('coT', '')),
        }

    @staticmethod
    def _safe_extract_zip(zip_path, extract_root):
        extract_root = Path(extract_root).resolve()
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            for member in zip_file.infolist():
                target = (extract_root / member.filename).resolve()
                if target != extract_root and extract_root not in target.parents:
                    raise RuntimeError(f'Unsafe path in BabyVision zip: {member.filename}')
            zip_file.extractall(extract_root)

    def _download_official_data(self):
        data_root = self._data_root()
        data_root.mkdir(parents=True, exist_ok=True)

        zip_path = data_root / 'babyvision_data.zip'
        zip_url = self.DATASET_URL[BABYVISION_DATASET]
        zip_md5 = self.DATASET_MD5[BABYVISION_DATASET]
        if not zip_path.exists() or md5(str(zip_path)) != zip_md5:
            download_file(zip_url, str(zip_path))

        if md5(str(zip_path)) != zip_md5:
            raise RuntimeError(f'BabyVision official zip md5 mismatch: {zip_path}')

        self._safe_extract_zip(zip_path, data_root)
        meta_path = data_root / 'babyvision_data' / 'meta_data.jsonl'
        if not meta_path.exists():
            raise FileNotFoundError(f'BabyVision official zip did not contain {meta_path}')
        return meta_path

    def load_data(self, dataset):
        assert dataset == BABYVISION_DATASET, dataset
        meta_path = self._find_meta_path()
        if meta_path is None:
            meta_path = self._download_official_data()

        base_dir = meta_path.parent

        records = []
        with open(meta_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                image_path = base_dir / item['image']
                records.append(self._record_from_item(item, image_path))

        return pd.DataFrame(records)

    def build_prompt(self, line):
        return super().build_prompt(line)

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) if pd.notna(x) else '' for x in data['prediction']]
        data['answer'] = [str(x) if pd.notna(x) else '' for x in data['answer']]

        judge_model = judge_kwargs.pop('model', self.DEFAULT_JUDGE_MODEL)
        safe_model = _safe_model_name(judge_model)
        storage = get_intermediate_file_path(eval_file, f'_{safe_model}_judge')
        tmp_file = get_intermediate_file_path(eval_file, f'_{safe_model}_judge_tmp', 'pkl')
        score_file = get_intermediate_file_path(eval_file, f'_{safe_model}_score', 'json')
        nproc = judge_kwargs.pop('nproc', 4)

        if osp.exists(storage):
            judged = load(storage)
        else:
            model = build_judge(model=judge_model, max_tokens=16, **judge_kwargs)
            if not model.working():
                raise RuntimeError('Judge model is not working properly. ' + DEBUG_MESSAGE)

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            lines = [data.iloc[i] for i in range(len(data))]
            indices = [line['index'] for line in lines]
            tasks = [(model, line) for line in lines if line['index'] not in ans]
            keys = [line['index'] for line in lines if line['index'] not in ans]
            if tasks:
                track_progress_rich(
                    self._evaluate_single,
                    tasks,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=keys,
                    save=tmp_file,
                )
                ans = load(tmp_file)

            judged = data.copy()
            judged['ModelResult'] = judged['prediction']
            judged['GroundTruth'] = judged['answer']
            judged['ExtractedAnswer'] = [ans[idx]['extracted_answer'] for idx in indices]
            judged['LLMJudgeResult'] = [ans[idx]['llm_judge_result'] for idx in indices]
            judged['JudgeResponse'] = [ans[idx]['judge_response'] for idx in indices]
            judged['JudgePrompt'] = [ans[idx]['judge_prompt'] for idx in indices]
            dump(judged, storage)

        metrics = self.compute_metrics(judged)
        dump(metrics, score_file)
        return metrics

    @staticmethod
    def _evaluate_single(model, line):
        question = line['question']
        answer = str(line['answer'])
        prediction = str(line['prediction'])
        extracted_answer = extract_boxed_answer(prediction)
        judge_prompt = build_babyvision_judge_prompt(
            question=question,
            groundtruth=answer,
            modeloutput=extracted_answer,
        )
        judge_response = model.generate(judge_prompt)
        llm_judge_result = parse_judge_response(judge_response)
        return {
            'extracted_answer': extracted_answer,
            'judge_prompt': judge_prompt,
            'judge_response': judge_response,
            'llm_judge_result': llm_judge_result,
            'score': int(llm_judge_result),
        }

    @staticmethod
    def compute_metrics(results):
        total = len(results)
        correct = sum(1 for _, row in results.iterrows() if bool(row['LLMJudgeResult']))
        metrics = {'Overall': correct / total * 100 if total > 0 else 0.0}

        for type_name in sorted(set(results['Type'])):
            subset = results[results['Type'] == type_name]
            metrics[f'Type/{type_name}'] = np.mean([bool(x) for x in subset['LLMJudgeResult']]) * 100

        subtype_scores = {}
        for _, row in results.iterrows():
            type_id = TYPE_MATCH.get(row['Type'], '0')
            key = f"{type_id}{row['Type']}/{row['Subtype']}"
            subtype_scores.setdefault(key, []).append(bool(row['LLMJudgeResult']))
        for subtype_name in sorted(subtype_scores):
            metrics[f'Subtype/{subtype_name}'] = np.mean(subtype_scores[subtype_name]) * 100

        return {k: round(float(v), 4) for k, v in metrics.items()}

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict) and 'Overall' in metrics:
            return {'Overall': metrics['Overall']}
        return super().report_primary_metric(metrics)

    @staticmethod
    def metrics_to_frame(metrics):
        return d2df(metrics)
