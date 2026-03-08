import os
import re
import os.path as osp
import pandas as pd
from PIL import Image
from collections import defaultdict
from .gen_base import GenBaseTextDataset
from ...smp import load, dump, track_progress_rich, get_intermediate_file_path
from ..utils.judge_util import build_judge  # your existing judge builder


# Category buckets and overall aggregation
WISE_DIMENSIONS = {
    'CULTURE': ['CULTURE'],
    'TIME': ['TIME'],
    'SPACE': ['SPACE'],
    'BIOLOGY': ['BIOLOGY'],
    'PHYSICS': ['PHYSICS'],
    'CHEMISTRY': ['CHEMISTRY'],
    'overall': [],
}


# Prompts for generation and scoring
SYSTEM_CAL_SCORE_PROMPT = (
    "You are a professional image quality auditor. "
    "Evaluate the image quality strictly according to the protocol."
)

SYSTEM_GENER_PRED_PROMPT = (
    "You are an intelligent chatbot designed for generating images based on a detailed description.\n"
    "------\n"
    "INSTRUCTIONS:\n"
    "- Read the detailed description carefully.\n"
    "- Generate an image based on the detailed description.\n"
    "- The image should be a faithful representation of the detailed description."
)

USER_GENER_PRED_PROMPT = (
    "Please generate an image based on the following description:\n"
    "{prompt}\n"
    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the image."
)


def prepare_response_prompt(item: pd.Series) -> str:
    """Build the text prompt for the image generator."""
    if isinstance(item, pd.Series):
        item = item.to_dict()
    return USER_GENER_PRED_PROMPT.format(prompt=item['prompt'])


def prepare_score_prompt(item):
    """Build the scoring prompt as a joined string with short physical lines (flake8 E501-safe)."""
    # Convert pandas Series to dict if needed
    if isinstance(item, pd.Series):
        item = item.to_dict()

    message = []
    eval_prompt = [
        "You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE "
        "RUTHLESSNESS.",
        "Only images meeting the HIGHEST standards should receive top scores.",
        "",
        "**Input Parameters**",
        "- PROMPT: [User's original prompt]",
        "- EXPLANATION: [Further explanation of the original prompt]",
        "---",
        "",
        "## Scoring Criteria",
        "",
        "**Consistency (0-2):** How accurately and completely the image reflects the PROMPT.",
        "* **0 (Rejected):** Fails to capture key elements of the prompt, or contradicts the prompt.",
        "* **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, "
        "or not accurately. Noticeable deviations from the prompt's intent.",
        "* **2 (Exemplary):** Perfectly and completely aligns with the PROMPT. Every single element and "
        "nuance of the prompt is flawlessly represented in the image. The image is an ideal, "
        "unambiguous visual realization of the given prompt.",
        "",
        "**Realism (0-2):** How realistically the image is rendered.",
        "* **0 (Rejected):** Physically implausible and clearly artificial. Breaks fundamental laws of "
        "physics or visual realism.",
        "* **1 (Conditional):** Contains minor inconsistencies or unrealistic elements. While somewhat "
        "believable, noticeable flaws detract from realism.",
        "* **2 (Exemplary):** Achieves photorealistic quality, indistinguishable from a real photograph. "
        "Flawless adherence to physical laws, accurate material representation, and coherent spatial "
        "relationships. No visual cues betraying AI generation.",
        "",
        "**Aesthetic Quality (0-2):** The overall artistic appeal and visual quality of the image.",
        "* **0 (Rejected):** Poor aesthetic composition, visually unappealing, and lacks artistic merit.",
        "* **1 (Conditional):** Demonstrates basic visual appeal, acceptable composition, and color "
        "harmony, but lacks distinction or artistic flair.",
        "* **2 (Exemplary):** Possesses exceptional aesthetic quality, comparable to a masterpiece. "
        "Strikingly beautiful, with perfect composition, a harmonious color palette, and a captivating "
        "artistic style. Demonstrates a high degree of artistic vision and execution.",
        "",
        "---",
        "",
        "## Output Format",
        "",
        "**Do not include any other text, explanations, or labels.** Return exactly three lines:",
        "Consistency: <0|1|2>",
        "Realism: <0|1|2>",
        "Aesthetic Quality: <0|1|2>",
        "",
        "---",
        "",
        "**IMPORTANT Enforcement:**",
        "Be EXTREMELY strict. A score of '2' is rare and only for the very best images.",
        "If in doubt, downgrade.",
        "",
        "For Consistency, '2' means complete, flawless adherence to every aspect of the prompt.",
        "For Realism, '2' means virtually indistinguishable from a real photograph.",
        "For Aesthetic Quality, '2' requires exceptional artistic merit.",
        "",
        "---",
        "Here are the inputs for this evaluation:",
        f'PROMPT: "{item["prompt"]}"',
        f'EXPLANATION: "{item["explanation"]}"',
        "IMAGE:",
    ]
    eval_prompt = '\n'.join(eval_prompt)
    message.append(dict(type='text', value=eval_prompt))
    output_img = GenBaseTextDataset.extract_single_image_from_response(item['prediction'])
    assert output_img is not None, item

    message.append(dict(type='image', value=output_img))
    message.append(dict(type='text', value="Please strictly follow the criteria and output template."))
    return message


def parse_score_dict(raw: str) -> dict:
    s = "" if raw is None else str(raw).strip()
    mC = re.search(r'Consistency\s*:\s*([0-2])', s, re.I)
    mR = re.search(r'Realism\s*:\s*([0-2])', s, re.I)
    mA = re.search(r'(Aesthetic\s*Quality|Aesthetic)\s*:\s*([0-2])', s, re.I)
    if mC and mR and mA:
        C, R, A = int(mC.group(1)), int(mR.group(1)), int(mA.group(2))
        return dict(c=C, r=R, a=A, log=s)

    nums = re.findall(r'(?<!\d)[0-2](?!\d)', s)
    if len(nums) == 3:
        C, R, A = int(nums[0]), int(nums[1]), int(nums[2])
        return dict(c=C, r=R, a=A, log=s)

    return dict(c=0, r=0, a=0, log='SCORE PARSING FAILED')


class WISE(GenBaseTextDataset):
    """WISE dataset wrapper for image generation and scoring."""

    TYPE = 'T2I'
    NUM_GENERATIONS = 1
    DEFAULT_JUDGE = 'gpt-4o-1120'

    DATASET_URL = {
        'WISE': 'https://opencompass.openxlab.space/utils/GenEval/WISE.tsv',
    }

    DATASET_MD5 = {
        'WISE': 'f4fb0fd05e83bd1c5ec48a37abe91735',
    }

    def build_prompt(self, line):
        """Build a text-only prompt list for JanusGeneration.generate_inner."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        prompt_text = prepare_response_prompt(line)
        messages = []
        messages.append(dict(type='text', value=prompt_text))
        return messages

    @classmethod
    def evaluate_sample(self, judge_model, sample):
        if isinstance(sample, pd.Series):
            sample = sample.to_dict()
        output_img = self.extract_single_image_from_response(sample['prediction'])
        if output_img is None:
            return dict(c=0, r=0, a=0, log=f'GENERATION FAILED, {self.FAIL_MSG}')
        score_prompt = prepare_score_prompt(sample)
        temperature = 0
        retry = 3
        resp = judge_model.generate(message=score_prompt, temperature=temperature)
        score = parse_score_dict(resp)
        while score['log'] == 'SCORE PARSING FAILED' and retry > 0:
            retry -= 1
            temperature += 0.5
            resp = judge_model.generate(message=score_prompt, temperature=temperature)
            score = parse_score_dict(resp)
        if 'FAILED' in score['log']:
            score['log'] = score['log'] + f',{self.FAIL_MSG}'
        return score

    def evaluate(self, eval_file, **judge_kwargs):
        """Score generated images with an LLM and aggregate WISE metrics.

        eval_file: .pkl with columns at least ['index', 'prediction'] where
                   'prediction' is a PIL.Image.Image.
        """
        judge = judge_kwargs['model']
        nproc = judge_kwargs.pop('nproc', 16)
        _ = judge_kwargs.pop('verbose', None)
        _ = judge_kwargs.pop('retry', None)

        tmp_file = get_intermediate_file_path(eval_file, f'_{judge}_tmp', 'pkl')
        score_file = get_intermediate_file_path(eval_file, f'_{judge}', 'tsv')
        rating_file = get_intermediate_file_path(eval_file, f'_{judge}_rating', 'json')

        judge_kwargs['temperature'] = 0.0
        model = build_judge(**judge_kwargs)

        eval_df = load(eval_file)
        eval_df['index'] = eval_df['index'].astype(str)

        if not osp.exists(score_file):
            # resume support: load tmp results and drop failures
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if 'FAILED' not in v['log']}

            todo_mask = ~eval_df['index'].isin(res.keys())
            data_un = eval_df[todo_mask].reset_index(drop=True)

            lt = len(data_un)
            # Build plain string prompts to avoid BaseAPI 'value' key issues
            if lt > 0:
                samples = [data_un.iloc[i] for i in range(len(data_un))]
                indices = [x['index'] for x in samples]
                jobs = [dict(judge_model=model, sample=sample) for sample in samples]
                _ = track_progress_rich(
                    self.evaluate_sample,      # callable(judge_model: BaseAPI, sample: dict) -> dict
                    jobs,      # iterable of dicts: {'judge_model': BaseAPI, 'sample': dict}
                    keys=indices,        # map results by 'index'
                    save=tmp_file,    # resume file
                    nproc=nproc
                )
                score_map = load(tmp_file) if osp.exists(tmp_file) else {}
                score_map.update(res)
            else:
                score_map = res

            for k in ['c', 'r', 'a', 'log']:
                eval_df[k] = [score_map[idx][k] for idx in eval_df['index']]
            dump(eval_df, score_file)

        if osp.exists(tmp_file):
            os.remove(tmp_file)

        results = load(score_file)
        rating = self.report_metric(results)
        dump(rating, rating_file)
        return rating

    @staticmethod
    def calculate_wiscore(consistency: int, realism: int, aesthetic_quality: int) -> float:
        """Weighted score in [0, 1]."""
        return (0.7 * consistency + 0.2 * realism + 0.1 * aesthetic_quality) / 2.0

    def report_metric(self, results):
        assert isinstance(results, pd.DataFrame)
        df = results
        disps = list(set(df['discipline']))
        counter = defaultdict(dict)
        for d in disps + ['overall']:
            counter[d]['score'] = []
            counter[d]['missing'] = 0
        for i in range(len(df)):
            row = df.iloc[i]
            discipline = row.get('discipline', '')
            if 'FAILED' not in row['log']:
                c, r, a = row['c'], row['r'], row['a']
                wiscore = self.calculate_wiscore(c, r, a)
                counter[discipline]['score'].append(wiscore)
                counter['overall']['score'].append(wiscore)
            else:
                counter[discipline]['missing'] += 1
                counter['overall']['missing'] += 1

        def get_mean(counter_dict, mode='normal'):
            assert mode in ['normal', 'wo_missing']
            if mode == 'normal':
                return sum(counter_dict['score']) / (len(counter_dict['score']) + counter_dict['missing'])
            elif mode == 'wo_missing':
                return sum(counter_dict['score']) / len(counter_dict['score'])

        results = {}
        # first we calculate normal score
        score = {}
        for k, v in counter.items():
            score[k] = get_mean(counter[k], mode='normal')
        score['weighted_score'] = (
            0.4 * score['CULTURE']
            + 0.167 * score['TIME']
            + 0.133 * score['SPACE']
            + 0.1 * score['BIOLOGY']
            + 0.1 * score['PHYSICS']
            + 0.1 * score['CHEMISTRY']
        )
        results['score'] = score

        score_wo_missing = {}
        for k, v in counter.items():
            score_wo_missing[k] = get_mean(counter[k], mode='wo_missing')
        score_wo_missing['weighted_score'] = (
            0.4 * score_wo_missing['CULTURE']
            + 0.167 * score_wo_missing['TIME']
            + 0.133 * score_wo_missing['SPACE']
            + 0.1 * score_wo_missing['BIOLOGY']
            + 0.1 * score_wo_missing['PHYSICS']
            + 0.1 * score_wo_missing['CHEMISTRY']
        )
        results['score_wo_missing'] = score_wo_missing

        missing_rate = {}
        for k, v in counter.items():
            missing_rate[k] = counter[k]['missing'] / (len(counter[k]['score']) + counter[k]['missing'])
        results['missing_rate'] = missing_rate
        overall_score = results['score']['weighted_score']
        overall_score = float(f'{overall_score * 100: .2f}')
        results['overall'] = overall_score
        return results
