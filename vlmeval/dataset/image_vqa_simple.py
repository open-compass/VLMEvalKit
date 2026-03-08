from .image_base import ImageBaseDataset
from .utils.judge_util import build_judge
from .utils.extractor import LLM_VERIFIER
from .utils.multiple_choice import report_acc_json
import os.path as osp
import re
from vlmeval.smp import *


VERIFY_PROMPT = """\
You are an AI assistant to help me judge if a model prediction is correct given the original question and the groundtruth answer. \
Your output should include both the correctness of the prediction (yes or no) and the reason of your judgement. \
Your should output a single json string with "correctness" and "reason" as keys. \
The value of "correctness" can only be two integers: 1 (yes) or 0 (no).

Example 1:
Question: Write the set of numbers represented on the number line in interval notation.
Answer: (-2,1]
Prediction: Extracted Answer: \\((-2, 1)\\)
Your output: {{"correctness": 0, "reason": "(-2, 1] is not the same as (-2, 1), 1 is included in the first interval but excluded in the second interval. "}}

Example 2:
Question: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
Answer: C
Prediction: B:2\u221a{{3}}
Your output: {{"correctness": 0, "reason": "The correct answer is C, but the prediction is B. They are not the same. "}}

Your Task:
Question: {question}?\nAnswer: {answer}.\nPrediction: {prediction}\nYour output:
"""  # noqa: E501


def format_check(s):
    s = parse_json(s)
    if "correctness" not in s or "reason" not in s:
        return None
    if s['correctness'] in ['0', '1']:
        s['correctness'] = int(s['correctness'])
    assert s['correctness'] in [0, 1], s['correctness']
    return s


class ImageVQASimple(ImageBaseDataset):

    TYPE = "VQA"
    DATASET_URL = {
        'ERQA': 'https://opencompass.openxlab.space/utils/VLMEval/ERQA.tsv',
        'VQARAD': 'https://opencompass.openxlab.space/utils/VLMEval/VQARAD.tsv'
    }
    DATASET_MD5 = {
        'ERQA': '32d447580ab5ec0d0c028ae9468a78ba',
        'VQARAD': 'be573933948143e3d676afa49c0c05bd',
    }

    DEFAULT_JUDGE = 'deepseek'
    JUDGE_FORMAT = '{model_name}_{dataset_name}_{judge_name}.tsv'
    RATING_FORMAT = '{model_name}_{dataset_name}_{judge_name}_rating.json'

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file).sort_values(by='index')
        nproc = judge_kwargs.pop('nproc', 16)
        judge = build_judge(**judge_kwargs)
        model_name = judge_kwargs['model']
        judge_file = get_intermediate_file_path(eval_file, f'_{model_name}', 'tsv')
        rating_file = get_intermediate_file_path(eval_file, f'_{model_name}_rating', 'json')
        lines = [row for _, row in data.iterrows()]

        if not osp.exists(judge_file):
            verifier = LLM_VERIFIER(judge, VERIFY_PROMPT, format_check)
            results = track_progress_rich(verifier.verify, lines, nproc=nproc, desc='Judging')
            data['hit'] = [x['correctness'] if x is not None else None for x in results]
            data['log'] = [x['reason'] if x is not None else None for x in results]
            dump(data, judge_file)
        else:
            data = load(judge_file)

        result = report_acc_json(data)
        dump(result, rating_file)
        return result


class BabyVision(ImageVQASimple):

    DATASET_URL = {
        'BabyVision': 'https://opencompass.openxlab.space/utils/VLMEval/BabyVision.tsv'
    }

    DATASET_MD5 = {
        'BabyVision': 'ec0fcceb522c1615a7f62a64ae775caa',
    }

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
Ground Truth Answer: {answer},
Model Output: {prediction}
"""
    DEFAULT_JUDGE = 'gpt-5.2'
    JUDGE_FORMAT = '{model_name}_{dataset_name}_{judge_name}.tsv'
    RATING_FORMAT = '{model_name}_{dataset_name}_{judge_name}_score.json'

    @staticmethod
    def extract_boxed_answer(text):
        """
        Extract the content from the last \\boxed{} pattern.
        Also supports alternative format: <|begin_of_box|>...<|end_of_box|>
        Returns None if no pattern found.
        """
        import regex
        if text is None:
            return None
        # Match \boxed{...} with support for nested braces
        pattern = r'\\boxed\{((?:[^{}]|{(?:[^{}]|{.*})*})*)\}'
        matches = regex.findall(pattern, text)
        if matches:
            return matches[-1]  # Return content from last \boxed{}
        # Alternative pattern
        pattern_alt = r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>'
        matches_alt = regex.findall(pattern_alt, text)
        if matches_alt:
            return matches_alt[-1].strip()
        return None

    @staticmethod
    def evaluate_single(judge, line):
        sample = dict(line)
        gpt_prompt = BabyVision.LLM_JUDGE_PROMPT.format(**sample)
        retry = 3
        temperature = 1

        while retry:
            gpt_response = judge.generate(gpt_prompt, temperature=temperature)
            if 'true' in gpt_response.lower():
                return dict(hit=1, log=gpt_response)
            elif 'false' in gpt_response.lower():
                return dict(hit=0, log=gpt_response)
            retry -= 1

        return dict(hit=0, log=FAIL_MSG + ', Last GPT Response: ' + gpt_response)

    def evaluate(self, eval_file, **kwargs):
        judge_name = kwargs.get('model')
        nproc = kwargs.pop('nproc', 32)
        kwargs['temperature'] = 1.0

        judge_file = self.get_judge_file_path(eval_file, judge_name=judge_name)
        rating_file = self.get_rating_file_path(eval_file, judge_name=judge_name)

        if osp.exists(judge_file):
            data = load(judge_file)
        else:
            data = load(eval_file)
            judge = build_judge(**kwargs)
            jobs = [dict(judge=judge, line=line) for _, line in data.iterrows()]
            results = track_progress_rich(
                self.evaluate_single,
                jobs,
                nproc=nproc,
                desc='Judging BabyVision Results')
            for k in ['hit', 'log']:
                data[k] = [item[k] for item in results]
            dump(data, judge_file)

        acc = report_acc_json(data, inds=['category', 'l2-category'], key='hit')
        dump(acc, rating_file)
        return acc


class VibeEval(ImageVQASimple):

    DATASET_URL = {
        'VibeEval': 'https://opencompass.openxlab.space/utils/VLMEval/VibeEval.tsv'
    }

    DATASET_MD5 = {
        'VibeEval': '278cb1d095bf268cb14235a9ecc825ad',
    }

    LLM_JUDGE_PROMPT = """\
[Question]
{question}

[Assistant Response]
{prediction}

[Ground Truth Response]
{answer}

[System]
Rate whether the assistant response correctly matches the ground truth, in regards to the image above.
The rating should be 1-5, where 1 is incorrect and 5 is correct.
Your response should be in the format:
Explanation: (your explanation)
Rating: (int)"""

    DEFAULT_JUDGE = 'gpt-4o-1120'
    JUDGE_FORMAT = '{model_name}_{dataset_name}_{judge_name}.tsv'
    RATING_FORMAT = '{model_name}_{dataset_name}_{judge_name}_score.json'

    @staticmethod
    def evaluate_single(judge, line):
        sample = dict(line)
        gpt_prompt = VibeEval.LLM_JUDGE_PROMPT.format(**sample)
        retry = 3
        temperature = 0

        while retry:
            gpt_response = judge.generate(gpt_prompt, temperature=temperature)
            re_match = re.search(r"Rating:\s*([1-5])", gpt_response)
            if re_match is not None:
                score = int(re_match.group(1))
                assert 1 <= score <= 5, gpt_response
                return dict(score=score, log=gpt_response)

            retry -= 1
            temperature += 0.5

        return dict(hit=0, log=FAIL_MSG + ', Last GPT Response: ' + gpt_response)

    def evaluate(self, eval_file, **kwargs):
        judge_name = kwargs.get('model')
        nproc = kwargs.pop('nproc', 32)

        judge_file = self.get_judge_file_path(eval_file, judge_name=judge_name)
        rating_file = self.get_rating_file_path(eval_file, judge_name=judge_name)

        if osp.exists(judge_file):
            data = load(judge_file)
        else:
            data = load(eval_file)
            judge = build_judge(**kwargs)
            jobs = [dict(judge=judge, line=line) for _, line in data.iterrows()]
            results = track_progress_rich(
                self.evaluate_single,
                jobs,
                nproc=nproc,
                desc='Judging VibeEval Results')
            for k in ['score', 'log']:
                data[k] = [item[k] for item in results]
            dump(data, judge_file)

        rating = report_acc_json(data, inds=['category'], key='score')
        for k in rating:
            if 'err_rate' in k:
                continue
            rating[k] = (rating[k] - 1) * 25
        dump(rating, rating_file)
        return rating
