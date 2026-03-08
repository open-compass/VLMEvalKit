import pandas as pd
from typing import Literal
from pydantic import BaseModel
from vlmeval.smp import *
from .utils.judge_util import build_judge
from .image_base import ImageBaseDataset


JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or \
not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. \
Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], \
focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. \
Do not comment on any background to the problem, do not attempt to solve the problem, \
do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within \
a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, \
ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0 and 100 from [response]. \
Put -1 if there is no confidence score available.

Provide your evaluation in a strict JSON format:
{{
  "extracted_final_answer": "string"
  "reasoning": "str",
  "correct": "str",
  "confidence": "int"
}}
"""


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int


class HLE(ImageBaseDataset):

    DATASET_URL = {
        'HLE': 'https://opencompass.openxlab.space/utils/VLMEval/HLE.tsv'
    }

    DATASET_MD5 = {
        'HLE': 'd82e2cef290db5050937853781562ce7'
    }
    DEFAULT_JUDGE = 'o3-mini'
    JUDGE_FORMAT = '{model_name}_{dataset_name}_judge.tsv'
    TYPE = 'VQA'

    def __init__(self, dataset='HLE', skip_noimg=False):
        super().__init__(dataset, skip_noimg)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']
        msgs = []
        if line['modality'] == 'image':
            tgt_path = self.dump_image(line)
            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image', value=p) for p in tgt_path])
            else:
                msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    @classmethod
    def judge_single_sample(cls, judge, line):
        prompt = JUDGE_PROMPT.format(
            question=line['question'],
            response=line['prediction'],
            correct_answer=line['answer']
        )
        retry = 3

        for i in range(retry):
            try:
                resp = judge.generate(prompt, response_format=ExtractedAnswer)
                if isinstance(resp, dict):
                    return resp
            except:
                pass
        return {
            'extracted_final_answer': None,
            'reasoning': None,
            'correct': 'no',
            'confidence': 0
        }

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        nproc = judge_kwargs.pop('nproc', 16)
        if 'timeout' not in judge_kwargs:
            judge_kwargs['timeout'] = 300
        judge = build_judge(**judge_kwargs)
        tmp_file = get_intermediate_file_path(eval_file, "_tmp", 'pkl')
        judge_file = get_intermediate_file_path(eval_file, "_judge", "tsv")
        if not osp.exists(judge_file):
            res = {}
            if osp.exists(tmp_file):
                try:
                    res = load(tmp_file)
                    res = {k: v for k, v in res.items() if isinstance(v, dict)}
                except:
                    os.remove(tmp_file)

            lines = [row for _, row in data.iterrows()]
            tups = [dict(judge=judge, line=row) for row in lines if row['index'] not in res]
            if len(tups):
                indices = [x['line']['index'] for x in tups]
                _ = track_progress_rich(
                    self.judge_single_sample,
                    tups,
                    nproc=nproc,
                    save=tmp_file,
                    keys=indices,
                    desc="LM-based matching on HLE"
                )
                res = load(tmp_file)
            keys = ['extracted_final_answer', 'reasoning', 'correct', 'confidence']
            for k in keys:
                data[k] = [res[idx][k] for idx in data['index']]
            dump(data, judge_file)

        data = load(judge_file)
        data['hit'] = [x == 'yes' for x in data['correct']]
        accuracy = np.mean(data['hit'])
        res = {'Overall': accuracy}
        for cate in ['image', 'text']:
            sub = data[data['modality'] == cate]
            res[cate] = np.mean(sub['hit'])
        rating_file = get_intermediate_file_path(eval_file, "_acc", "csv")
        dump(d2df(res), rating_file)
        return res
