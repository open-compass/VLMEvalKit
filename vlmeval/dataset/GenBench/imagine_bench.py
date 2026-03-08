import pandas as pd
from .gen_base import GenBaseTextDataset
from ...smp import load, dump, track_progress_rich, get_intermediate_file_path
import validators
import os
import os.path as osp
import re


def prepare_score_prompt(item):
    """Build the scoring prompt for GPT evaluation."""
    if isinstance(item, pd.Series):
        item = item.to_dict()

    system_prompt = """You are an AI quality auditor for text-to-image generation. \
Apply these rules with ABSOLUTE RUTHLESSNESS. Only images meeting the HIGHEST standards should receive top scores. \
Your job is to evaluate how well the image fulfills the fantasy design task, \
considering all instructions with maximum precision."""  # noqa: E501

    user_prompt = f"""Please evaluate strictly and return ONLY the three scores as requested.

# Fantasy Object Image Evaluation Protocol (Human-based tasks)

## Input Parameters
- PROMPT: [Original instruction provided to the model]
- EXPLANATION: [Detailed explanation of what the prompt is trying to achieve]

---

## Scoring Criteria (0–10 for each)

**Fantasy Fulfillment (0–10):**
How well does the image realize the intended fantasy transformation described in the prompt?
- 0: No sign of the transformation; the fantasy idea is entirely ignored or contradicted.
- 1–3: The transformation is misunderstood or poorly executed, with key fantasy features missing, wrong, or distorted.
- 4–6: Some aspects of the transformation appear, but with clear flaws — such as vague, generic, or misaligned features.
- 7–9: Most fantasy elements are present and understandable, but minor details may be off in material, form, or integration.
- 10: The transformation is fully and precisely implemented. Every visual element aligns with the prompt's intent — including texture, shape, integration, and plausibility.

To score 10, the image must exactly reflect the imagined transformation with no major deviations.

**Identity Preservation (0–10):**
How clearly does the image preserve the recognizable identity of the original object/person despite the fantasy alteration?
- 0: The object/person is completely unrecognizable or heavily distorted.
- 1–3: The identity is barely preserved; key visual traits are missing or incorrect.
- 4–6: Some identity traits remain, but many are altered, stylized, or inconsistent.
- 7–9: The core features are retained well, with minor issues.
- 10: The base object/person is clearly and faithfully represented in all major aspects.
Stylized or cartoon-like rendering should be rated **lower**, even if the shape is roughly preserved. Identity must be preserved in **realistic detail**, not just symbolic outline.

**Aesthetic Quality (0–10):**
How visually appealing, clear, and realistic is the image overall?
- 0: Poor quality, low resolution, or visually broken.
- 1–3: Basic rendering flaws or artifacts significantly hurt visual quality.
- 4–6: Adequate quality with moderate imperfections.
- 7–9: High-quality rendering with good composition and polish.
- 10: Excellent visual clarity, realism, and artistic balance.

---

## Output Format
Return scores (0–10) and **brief justification** for each item.

Output Format Example:

Fantasy Fulfillment: <score>
Reason: <one-sentence explanation>

Identity Preservation: <score>
Reason: <one-sentence explanation>

Aesthetic Quality: <score>
Reason: <one-sentence explanation>

---

Only provide the scores and reasons. Do not include any extra formatting or comments.

---

## Enforcement Notes
- Be extremely strict and objective.
- A score of **10** must indicate complete success and flawless execution.
- If the fantasy transformation is weak or confusing → downgrade **Fantasy Fulfillment**.
- If the base object/person is unrecognizable or overly stylized → downgrade **Identity Preservation**.
- If realism or visual appeal is compromised → downgrade **Aesthetic Quality**.
- Reject images that exhibit cartoonish rendering or inconsistent fantasy logic.

---

Here are the inputs for evaluation:
PROMPT: "{item['question']}"
EXPLANATION: "{item['note']}"

Please evaluate this image:
"""  # noqa: E501
    img = GenBaseTextDataset.extract_single_image_from_response(item['prediction'])
    assert img is not None, item
    messages = [
        {"role": "system", "value": system_prompt},
        {"role": "user", "type": "text","value": user_prompt},
        {"role": "user", "type": "image","value": img}
    ]
    return messages


def extract_scores(evaluation_text):
    """Extract scores from GPT evaluation response."""
    pattern = r"(Fantasy Fulfillment|Identity Preservation|Aesthetic Quality)\s*[:：]?\s*(\d{1,2})"
    scores = {
        "Fantasy Fulfillment": -1,
        "Identity Preservation": -1,
        "Aesthetic Quality": -1
    }
    for match in re.findall(pattern, evaluation_text):
        field, value = match
        field = field.strip()
        try:
            val = float(value)
            val = max(0, min(val, 10))
            if field in scores:
                scores[field] = val
        except ValueError:
            continue
    # The new keys will be ['fantasy', 'identity', 'aesthetic']
    scores = {k.split()[0].lower(): v for k, v in scores.items()}
    return scores


class ImagineBench(GenBaseTextDataset):
    """ImagineBench dataset wrapper for text generation and scoring."""

    TYPE = 'T2I'
    NUM_GENERATIONS = 1
    DEFAULT_JUDGE = 'gpt-4o-1120'

    DATASET_URL = {
        'ImagineBench': 'https://opencompass.openxlab.space/utils/GenEval/ImagineBench.tsv'
    }

    DATASET_MD5 = {
        'ImagineBench': 'dd65cfbb5672946a439b16b7696af2a0',
    }

    def evaluate_sample(self, judge_model, sample):
        img = self.extract_single_image_from_response(sample['prediction'])
        if img is None:
            return dict(fantasy=-1, identity=-1, aesthetic=-1, log=f"PREDICTION MISSING, {self.FAIL_MSG}")
        message = prepare_score_prompt(sample)
        retry = 3
        temperature = 0.0
        response = judge_model.generate(message, temperature=temperature)
        score = extract_scores(response)

        def score_ok(d):
            return d['fantasy'] >= 0 and d['identity'] >= 0 and d['aesthetic'] >= 0

        log = f'GPT Response Iter 0: {response}\n'
        while not score_ok(score) and retry > 0:
            temperature += 0.5
            retry -= 1
            response = judge_model.generate(message, temperature=temperature)
            new_score = extract_scores(response)
            log += f'GPT Response Iter {3-retry}: {response}\n'
            for k in score:
                if score[k] == -1 and new_score[k] >= 0:
                    score[k] = new_score[k]
        if not score_ok(score):
            log += f'FAILED to extract scores from GPT response.\n{self.FAIL_MSG}'
        score['log'] = log
        return score

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate generated images using GPT-4o based on the ImagineBench protocol."""
        from ..utils.judge_util import build_judge
        judge = judge_kwargs.get('model', None)
        if judge is None:
            raise ValueError("Missing 'model' key in judge_kwargs. Please specify a judge model.")

        nproc = judge_kwargs.pop('nproc', 4)
        _ = judge_kwargs.pop('verbose', None)
        _ = judge_kwargs.pop('retry', None)

        tmp_file = get_intermediate_file_path(eval_file, f'_{judge}_tmp', 'pkl')
        tgt_file = get_intermediate_file_path(eval_file, f'_{judge}', 'tsv')
        rating_file = get_intermediate_file_path(eval_file, f'_{judge}_rating', 'json')

        judge_kwargs['temperature'] = 0.0
        model = build_judge(**judge_kwargs)

        eval_df = load(eval_file)
        eval_df['index'] = eval_df['index'].astype(str)

        # run scoring if not already present
        if not osp.exists(tgt_file):
            # resume support: load tmp results and drop failures
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if 'FAILED' not in v['log']}

            todo_mask = ~eval_df['index'].isin(res.keys())
            data_un = eval_df[todo_mask].reset_index(drop=True)

            lt = len(data_un)
            # Build plain string prompts to avoid BaseAPI 'value' key issues
            if lt > 0:
                samples = [data_un.iloc[i] for i in range(lt)]
                indices = [x['index'] for x in samples]
                jobs = [dict(judge_model=model, sample=sample) for sample in samples]

                _ = track_progress_rich(
                    self.evaluate_sample,      # callable(message: str) -> str
                    jobs,         # iterable of dicts: {'message': str}
                    keys=indices,        # map results by 'index'
                    save=tmp_file,       # resume file
                    nproc=nproc,
                )
                new_map = load(tmp_file)
                score_map = {**res, **new_map}
            else:
                score_map = res
            for k in ['aesthetic', 'fantasy', 'identity', 'log']:
                eval_df[k] = [score_map[idx][k] for idx in eval_df['index']]
            dump(eval_df, tgt_file)

        if osp.exists(tmp_file):
            os.remove(tmp_file)
        results = load(tgt_file)
        score = self.report_metric(results)
        dump(score, rating_file)
        return score

    def report_metric(self, results):
        types = set(results['type'])
        types = list(types) + ['overall']
        keys = ['fantasy', 'identity', 'aesthetic', 'weighted']
        from collections import defaultdict
        counter = defaultdict(dict)
        for tp in types:
            for k in keys:
                counter[tp][k + '_score'] = []
                counter[tp][k + '_missing'] = 0

        for _, row in results.iterrows():
            valid_cnt = 0
            tp = row['type']
            if row['fantasy'] >= 0:
                valid_cnt += 1
                counter[tp]['fantasy_score'].append(row['fantasy'])
                counter['overall']['fantasy_score'].append(row['fantasy'])
            else:
                counter[tp]['fantasy_missing'] += 1
                counter['overall']['fantasy_missing'] += 1

            if row['identity'] >= 0:
                valid_cnt += 1
                counter[tp]['identity_score'].append(row['identity'])
                counter['overall']['identity_score'].append(row['identity'])
            else:
                counter[tp]['identity_missing'] += 1
                counter['overall']['identity_missing'] += 1

            if row['aesthetic'] >= 0:
                valid_cnt += 1
                counter[tp]['aesthetic_score'].append(row['aesthetic'])
                counter['overall']['aesthetic_score'].append(row['aesthetic'])
            else:
                counter[tp]['aesthetic_missing'] += 1
                counter['overall']['aesthetic_missing'] += 1

            if valid_cnt == 3:
                weighted = 0.6 * row['fantasy'] + 0.3 * row['identity'] + 0.1 * row['aesthetic']
                counter[tp]['weighted_score'].append(weighted)
                counter['overall']['weighted_score'].append(weighted)
            else:
                counter[tp]['weighted_missing'] += 1
                counter['overall']['weighted_missing'] += 1
        ret = {}
        score = defaultdict(dict)
        for tp in types:
            for k in keys:
                score[tp][k] = sum(counter[tp][k + '_score']) / (len(counter[tp][k + '_score']) + counter[tp][k + '_missing'])  # noqa: E501
        ret['score'] = score

        score_wo_missing = defaultdict(dict)
        for tp in types:
            for k in keys:
                score_wo_missing[tp][k] = sum(counter[tp][k + '_score']) / len(counter[tp][k + '_score'])
        ret['score_wo_missing'] = score_wo_missing

        missing_rate = defaultdict(dict)
        for tp in types:
            for k in keys:
                missing_rate[tp][k] = counter[tp][k + '_missing'] / (len(counter[tp][k + '_score']) + counter[tp][k + '_missing'])  # noqa: E501
        ret['missing_rate'] = missing_rate
        overall_score = ret['score']['overall']['weighted']
        overall_score = float(f'{overall_score:.2f}')
        ret['overall'] = overall_score
        return ret
