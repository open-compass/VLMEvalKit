import re
from ..smp import *
from .image_base import ImageBaseDataset
from .utils.judge_util import DEBUG_MESSAGE, build_judge
from pydantic import BaseModel
from typing import List


class JudgeOutput(BaseModel):
    question_solution_analysis: str
    is_fully_correct: bool
    check_scoring_point: str
    awarded_points: List[str]
    final_score: int


EVAL_INST_PART_1 = """\
You are an expert math teacher and grader. \
Your task is to evaluate a student's solution to a mathematical question and provide a score.
You will be provided with the mathematical question (which may include multiple sub-questions), \
the student's solution, the correct answer, and the maximum possible score for the question below:
"""

EVAL_INST_PART_2 = """
{{'student_solution':{prediction}, 'number_of_answers':{number_of_answers}, \
    'correct_answer':{answer}, 'max_score':{total_score}}}.

Please follow these steps precisely:

1. Initial Check for Correctness:
 - Thoroughly review the question and the student_solution to identify the student’s final answer.
 - Compare this final answer directly with the provided correct_answer.
 - If the answers match exactly, award the full max_score.

2. Partial Credit Evaluation:
 - If the student's answer is not fully correct, evaluate its work for partial credit using the grading rubric: \
    {{'scoring_points':{scoring_points}, 'point_values':{scores}}}.
 - Go through each scoring_point, indicate if the student successfully completed that step.
 - Write down all the point_ids that the student earned and calculate the total score \
    by summing the values of those points.

3. Provide your evaluation in a strict JSON format:
{{
  "question_solution_analysis": "string"
  "is_fully_correct": "boolean",
  "check_scoring_point":"string",
  "awarded_points": ["all" OR a list of earned point_ids like "p1", "p2"],
  "final_score": "number"
}}

Field Explanations:
 - "question_solution_analysis": Analyze the question requirements and \
    compare the student's answer against the correct_answer."
 - "is_fully_correct": True if the student's solution is fully correct, otherwise False.
 - "check_scoring_point": If fully correct, provide an empty string "". If not fully correct, \
    explain where in the student_solution each scoring point is fully met or not met.
 - "awarded_points": If fully correct, this should be ["all"]. If partially correct, \
    provide a list containing the fully met point_ids (e.g., ["p1", "p3"]). \
        If no points met, provide an empty list [].
 - "final_score": the max_score if fully correct, or the sum of partial scores otherwise.

Provide your answer in ENGLISH!
"""

FAIL_MSG = 'Failed to obtain answer via API.'


def MathVR_auxeval(model, line, q_prompt):
    judge_prompt = []
    judge_prompt.append(dict(type='text', value=EVAL_INST_PART_1))
    judge_prompt.extend(q_prompt)
    score_inst = EVAL_INST_PART_2.format(
        prediction=line['prediction'],
        number_of_answers=line['number_of_answers'],
        answer=line['answer'],
        total_score=line['total_score'],
        scoring_points=line['scoring_points'],
        scores=line['scores'],
    )
    judge_prompt.append(dict(type='text', value=score_inst))
    resp = None
    retry = 3
    temperature = 0

    for i in range(retry):
        try:
            resp = model.generate(judge_prompt, temperature=temperature, response_format=JudgeOutput)
            assert isinstance(resp, dict)
            for k in JudgeOutput.__fields__:
                assert k in resp, f"Key {k} not found in judge output {resp}."
            return resp
        except:
            temperature += 0.5
            resp = None

    if resp is None:
        return {
            'question_solution_analysis': '',
            'is_fully_correct': False,
            'check_scoring_point': FAIL_MSG,
            'awarded_points': [],
            'final_score': 0,
        }


def MathVR_acc(eval_file):
    data = load(eval_file)
    stats = {
        'tot_score': 0, 'got_score': 0, 'missing_score': 0,
        'tot_avg_score': 0, 'got_avg_score': 0, 'missing_avg_score': 0
    }
    categories = ['text', 'multimodal', 'overall']
    res = {
        k: cp.deepcopy(stats) for k in categories
    }
    for i, row in data.iterrows():
        modal = row['category']
        if row['check_scoring_point'] == FAIL_MSG:
            for k in [modal, 'overall']:
                res[k]['missing_score'] += row['total_score']
                res[k]['missing_avg_score'] += 1
        else:
            for k in [modal, 'overall']:
                res[k]['tot_score'] += row['total_score']
                res[k]['got_score'] += row['final_score']
                res[k]['tot_avg_score'] += 1
                res[k]['got_avg_score'] += row['final_score'] / row['total_score']
    ret = {}
    ret['stats'] = res
    ret['score'] = {
        k: {
            'micro_avg': res[k]['got_score'] / (res[k]['tot_score'] + res[k]['missing_score']),
            'macro_avg': res[k]['got_avg_score'] / (res[k]['tot_avg_score'] + res[k]['missing_avg_score']),
        } for k in categories
    }
    ret['score_wo_missing'] = {
        k: {
            'micro_avg': res[k]['got_score'] / res[k]['tot_score'],
            'macro_avg': res[k]['got_avg_score'] / res[k]['tot_avg_score'],
        } for k in categories
    }
    ret['missing_rate'] = {
        k: {
            'micro': res[k]['missing_score'] / (res[k]['tot_score'] + res[k]['missing_score']),
            'macro': res[k]['missing_avg_score'] / (res[k]['tot_avg_score'] + res[k]['missing_avg_score']),
        } for k in categories
    }
    return ret


class MathVR(ImageBaseDataset):

    TYPE = 'VQA'
    DEFAULT_JUDGE = 'gpt-4.1'

    DATASET_URL = {
        "MathVR": "https://opencompass.openxlab.space/utils/VLMEval/MathVR.tsv",
    }
    DATASET_MD5 = {
        "MathVR": "c55a93dac55aa2faa4d882e5f98667b5",
    }

    INST = (
        "You should begin by detailing the internal reasoning process, "
        "and then present the answer to the user. \n"
        "Engage in an internal thinking process with thorough reasoning and reflections. "
        "You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem. \n"  # noqa: E501
        "The reasoning process should be enclosed within <think> </think> tags, as follows:\n"
        "<think> reasoning process here </think> answer here. "
    )

    INST_GEN = (
        "You should begin by detailing the internal reasoning process, "
        "and then present the answer to the user. \n"
        "Engage in an internal thinking process with thorough reasoning and reflections. "
        "You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem. \n"  # noqa: E501
        "You can insert generated images in the answer or thinking process. \n"
        "The reasoning process should be enclosed within <think> </think> tags, as follows:\n"
        "<think> reasoning process here </think> answer here. "
    )

    def build_prompt(self, line, with_system=True):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)
        question_text = line['question']
        msgs = [dict(role='system', type='text', value=self.INST)] if with_system else []

        pattern = r'(<image\d>)'
        segs = re.split(pattern, question_text)
        for i, seg in enumerate(segs):
            if seg.startswith('<image'):
                image_idx = int(seg.split('<image')[1][0]) - 1
                msgs.append({'type': 'image', 'value': tgt_path[image_idx]})
            elif len(seg):
                msgs.append({'type': 'text', 'value': seg})
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        model = judge_kwargs['model']
        storage = get_intermediate_file_path(eval_file, f'_{model}')
        tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        judge_keys = [
            'question_solution_analysis', 'is_fully_correct', 'check_scoring_point',
            'awarded_points', 'final_score'
        ]

        if not osp.exists(storage):
            self.data['index'] = self.data['index'].astype(str)
            raw_lines = [row for _, row in self.data.iterrows()]
            raw_line_map = {x['index']: x for x in raw_lines}

            data = load(eval_file)
            data['index'] = data['index'].astype(str)
            model = build_judge(max_tokens=2048, temperature=0.0, **judge_kwargs)
            assert model.working(), 'MathVista evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [
                (model, line, self.build_prompt(raw_line_map[line['index']], with_system=False))
                for line in lines
            ]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                _ = track_progress_rich(
                    MathVR_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)

            for k in judge_keys:
                data[k] = [ans[idx][k] for idx in data['index']]
            dump(data, storage)

        score = MathVR_acc(storage)
        score_pth = get_intermediate_file_path(storage, '_score', 'json')
        dump(score, score_pth)
        return score
