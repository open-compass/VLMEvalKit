from .gen_base import GenBaseTextDataset
from vlmeval.smp import load, dump, track_progress_rich, get_intermediate_file_path
import pandas as pd
import os
import os.path as osp
import json
import ast

SYS_PROMPT = """You are an expert image evaluator.

Your task is to determine whether the given image faithfully satisfies the \
visual instruction and the expectation checklist.

Follow these rules strictly:
1. The image must match **all** expectations, including:
- Object classes
- Counts of each object
- Colors of each object
- Spatial position within the image (e.g., "above", "below", based on real pixel position)
- Size and relative scale of objects
2. The image must appear as a **natural, coherent, photo-like single image**.
- Do NOT allow stylized images (e.g., cartoons, sketches, anime).
- Do NOT allow collage-style or multi-panel images. Only one consistent, realistic scene is acceptable.
3. Be very strict and conservative in your judgment.

Return your result as a JSON object using this format:
{
"correct": 1 if the image fully satisfies all expectations, else 0,
"reason": "You may explain in detail what is missing or incorrect"
}"""  # noqa: E501


def prepare_score_prompt(item):
    """Build the scoring prompt for GPT evaluation."""
    if isinstance(item, pd.Series):
        item = item.to_dict()
    # Convert string representation of list back to list
    try:
        include_objects = ast.literal_eval(item['include'])
    except:
        include_objects = []

    explanation = metadata_to_explanation({
        'include': include_objects
    })
    img = GenBaseTextDataset.extract_single_image_from_response(item['prediction'])
    assert img is not None, img

    messages = [
        {
            "role": "system",
            "value": SYS_PROMPT
        },
        {
            "role": "user",
            "type": "text",
            "value": f"Instruction:\n{item['question']}\n\nExpectation checklist:\n{explanation}"
        },
        {
            "role": "user",
            "type": "image",
            "value": img
        }
    ]
    return messages


def metadata_to_explanation(metadata):
    """Convert metadata to natural language explanation."""
    parts = []

    def format_item(item):
        obj = item["class"]
        count = item.get("count", 1)
        color = item.get("color", None)
        noun = f"{count} {obj + 's' if count > 1 else obj}"
        desc_parts = []
        if color:
            desc_parts.append(f"{color}-colored")
        if desc_parts:
            noun = f"{' '.join(desc_parts)} {noun}"
        return f"{noun} present in the image"

    for item in metadata.get("include", []):
        parts.append(f"- {format_item(item)}.")

    return "This image should contain:\n" + "\n".join(parts)


def extract_json_from_response(text):
    """Extract JSON object from GPT response."""
    import re
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))
    else:
        return dict(correct=-1, reason=f"JSON PARSING FAILED, {GenEvalPP.FAIL_MSG}")


class GenEvalPP(GenBaseTextDataset):
    """Geneval++ dataset wrapper for text generation and scoring."""

    TYPE = 'T2I'
    NUM_GENERATIONS = 1
    DEFAULT_JUDGE = 'gpt-4o-1120'

    DATASET_URL = {
        'GenEvalPP': 'https://opencompass.openxlab.space/utils/GenEval/GenEvalPP.tsv',
    }

    DATASET_MD5 = {
        'GenEvalPP': '52907d369795a4ce06fe359174f72acf',
    }

    def evaluate_sample(self, judge_model, sample):
        img = self.extract_single_image_from_response(sample['prediction'])
        if img is None:
            return dict(correct=-1, reason=f"PREDICTION MISSING, {GenEvalPP.FAIL_MSG}", raw_response='')

        message = prepare_score_prompt(sample)
        retry = 3
        temperature = 0.0
        response = judge_model.generate(message=message, temperature=temperature)
        extracted = extract_json_from_response(response)
        while 'FAILED' in extracted['reason'] and retry > 0:
            retry -= 1
            temperature += 0.5
            response = judge_model.generate(message=message, temperature=temperature)
            extracted = extract_json_from_response(response)
        extracted['raw_response'] = response
        if extracted['correct'] != -1:
            extracted['correct'] = 0 if extracted['correct'] < 0.5 else 1
        return extracted

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate generated images using GPT-4o based on the Geneval++ protocol."""
        from ..utils.judge_util import build_judge

        judge = judge_kwargs['model']
        nproc = judge_kwargs.pop('nproc', 16)
        _ = judge_kwargs.pop('verbose', None)
        _ = judge_kwargs.pop('retry', None)

        tmp_file = get_intermediate_file_path(eval_file, f'_{judge}_tmp', 'pkl')
        judge_file = get_intermediate_file_path(eval_file, f'_{judge}', 'tsv')
        rating_file = get_intermediate_file_path(eval_file, f'_{judge}_rating', 'json')

        judge_kwargs['temperature'] = 0.0
        model = build_judge(**judge_kwargs)
        eval_df = load(eval_file)
        eval_df['index'] = eval_df['index'].astype(str)

        # run scoring if not already present
        if not osp.exists(judge_file):
            # resume support: load tmp results and drop failures
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if self.FAIL_MSG not in v['reason']}

            todo_mask = ~eval_df['index'].isin(res.keys())
            data_un = eval_df[todo_mask].reset_index(drop=True)

            lt = len(data_un)
            # Build plain string prompts to avoid BaseAPI 'value' key issues
            if lt > 0:
                samples = [data_un.iloc[i] for i in range(lt)]
                indices = [x['index'] for x in samples]
                jobs = [dict(judge_model=model, sample=sample) for sample in samples]
                _ = track_progress_rich(
                    self.evaluate_sample,       # callable(message: str) -> str
                    jobs,                       # iterable of dicts: {'message': str}
                    keys=indices,               # map results by 'index'
                    save=tmp_file,              # resume file
                    nproc=nproc,
                )
                new_map = load(tmp_file)
                score_map = {**res, **new_map}
            else:
                score_map = res
            key_map = {
                'correct': 'score',
                'reason': 'log',
                'raw_response': 'gpt_response',
            }
            for k in ['correct', 'reason', 'raw_response']:
                eval_df[key_map[k]] = [score_map[idx][k] for idx in eval_df['index']]
            dump(eval_df, judge_file)
            final_result = eval_df
        else:
            final_result = load(judge_file)
        acc = self.report_metric(final_result)
        dump(acc, rating_file)

        if osp.exists(tmp_file):
            os.remove(tmp_file)

        return acc

    def report_metric(self, final_result):
        tags = set(final_result['tag'])
        tags = list(tags) + ['overall']
        counter = {k: dict(score=[], missing=0) for k in tags}

        for _, row in final_result.iterrows():
            tag = row['tag']
            correct = row['score']
            if correct != -1:
                counter[tag]['score'].append(correct)
                counter['overall']['score'].append(correct)
            else:
                counter[tag]['missing'] += 1
                counter['overall']['missing'] += 1
        res = {}
        score = {}
        for k in tags:
            score[k] = sum(counter[k]['score']) / (len(counter[k]['score']) + counter[k]['missing'])
        res['score'] = score

        score_wo_missing = {}
        for k in tags:
            score_wo_missing[k] = sum(counter[k]['score']) / len(counter[k]['score'])
        res['score_wo_missing'] = score_wo_missing

        missing_rate = {}
        for k in tags:
            missing_rate[k] = counter[k]['missing'] / (len(counter[k]['score']) + counter[k]['missing'])
        res['missing_rate'] = missing_rate
        for k in res:
            for kk in res[k]:
                res[k][kk] = float(f'{res[k][kk] * 100:.2f}')

        res['overall'] = res['score']['overall']
        res['format'] = 'percentage'
        return res
