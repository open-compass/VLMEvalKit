import pandas as pd
from .gen_base import GenBaseImageDataset
from vlmeval.smp import load, dump, track_progress_rich, get_intermediate_file_path
import os
import re
import json
import os.path as osp


class OmniContext(GenBaseImageDataset):
    """OmniContext dataset for multi-image context understanding."""

    TYPE = 'TI2I'
    NUM_GENERATIONS = 1
    DEFAULT_JUDGE = 'gpt-4o-1120'

    DATASET_URL = {
        'OmniContext': 'https://opencompass.openxlab.space/utils/GenEval/OmniContext.tsv',
    }

    DATASET_MD5 = {
        'OmniContext': 'be0a8174f0f84ca7e12bd66c2be4aedb',
    }

    @classmethod
    def evaluate_sample(self, judge_model, sample):
        # If Gen Failed, score will directly be 0
        img = self.extract_single_image_from_response(sample['prediction'])
        if img is None:
            return dict(
                pf_score=-1,
                sc_score=-1,
                pf_score_reasoning='Generation Failed',
                sc_score_reasoning='Generation Failed')
        # Prepare PF (Prompt Following) prompts
        pf_score_prompt = self.prepare_score_prompt(sample, task_type="prompt_following")
        sc_score_prompt = self.prepare_score_prompt(sample, task_type="subject_consistency")
        # Judge will use T=0 by default, hardcoded
        temperature = 0
        retry = 3
        pf_response = judge_model.generate(message=pf_score_prompt, temperature=temperature)
        pf_score = self.extract_scores(pf_response)
        while pf_score['score'] == -1 and retry > 0:
            temperature += 0.5
            retry -= 1
            pf_response = judge_model.generate(message=pf_score_prompt, temperature=temperature)
            pf_score = self.extract_scores(pf_response)
        # Now we handle with sc_score
        temperature = 0
        retry = 3
        sc_response = judge_model.generate(message=sc_score_prompt, temperature=temperature)
        sc_score = self.extract_scores(sc_response)
        while sc_score['score'] == -1 and retry > 0:
            temperature += 0.5
            retry -= 1
            sc_response = judge_model.generate(message=sc_score_prompt, temperature=temperature)
            sc_score = self.extract_scores(sc_response)
        res = {}
        res['pf_score'] = pf_score['score']
        res['sc_score'] = sc_score['score']
        res['pf_score_reasoning'] = pf_score['reasoning']
        res['sc_score_reasoning'] = sc_score['reasoning']
        return res

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate generated images using GPT-4o based on the OmniContext protocol."""
        from ..utils.judge_util import build_judge

        judge = judge_kwargs.get('model', None)
        if judge is None:
            raise ValueError("Missing 'model' key in judge_kwargs. Please specify a judge model.")

        nproc = judge_kwargs.pop('nproc', 16)
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
            res = {k: v for k, v in res.items() if (v['pf_score'] != -1 and v['sc_score'] != -1)}

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
                    nproc=nproc,
                )
                score_map = load(tmp_file) if osp.exists(tmp_file) else {}
                score_map.update(res)
            else:
                score_map = res

            for k in ['pf_score', 'sc_score', 'pf_score_reasoning', 'sc_score_reasoning']:
                eval_df[k] = [score_map[idx][k] for idx in eval_df['index']]
            dump(eval_df, tgt_file)
            final_results = eval_df
        else:
            final_results = load(tgt_file)

        if osp.exists(tmp_file):
            os.remove(tmp_file)

        metrics = self.report_metric(final_results)
        dump(metrics, rating_file)
        return metrics

    def report_metric(self, eval_df):
        from collections import defaultdict
        counter = defaultdict(dict)
        task_types = set(eval_df['task_type'])
        task_types = list(task_types) + ['all']
        for task in task_types:
            counter[task]['pf_scores'] = []
            counter[task]['sc_scores'] = []
            counter[task]['pf_missing'] = 0
            counter[task]['sc_missing'] = 0

        for _, item in eval_df.iterrows():
            task = item['task_type']
            pf_score = item['pf_score']
            if pf_score == -1:
                counter[task]['pf_missing'] += 1
                counter['all']['pf_missing'] += 1
            else:
                counter[task]['pf_scores'].append(pf_score)
                counter['all']['pf_scores'].append(pf_score)

            sc_score = item['sc_score']
            if sc_score == -1:
                counter[task]['sc_missing'] += 1
                counter['all']['sc_missing'] += 1
            else:
                counter[task]['sc_scores'].append(sc_score)
                counter['all']['sc_scores'].append(sc_score)
        ret = {}
        ret['stats'] = counter
        avg_score = defaultdict(dict)
        for k in counter:
            avg_score[k]['pf_score'] = sum(counter[k]['pf_scores']) / (len(counter[k]['pf_scores']) + counter[k]['pf_missing'])  # noqa: E501
            avg_score[k]['sc_score'] = sum(counter[k]['sc_scores']) / (len(counter[k]['sc_scores']) + counter[k]['sc_missing'])  # noqa: E501
            avg_score[k]['pf_missing_rate'] = counter[k]['pf_missing'] / (len(counter[k]['pf_scores']) + counter[k]['pf_missing'])  # noqa: E501
            avg_score[k]['sc_missing_rate'] = counter[k]['sc_missing'] / (len(counter[k]['sc_scores']) + counter[k]['sc_missing'])  # noqa: E501
        ret['avg_score'] = avg_score

        avg_score_wo_missing = defaultdict(dict)
        for k in avg_score:
            avg_score_wo_missing[k]['pf_score'] = avg_score[k]['pf_score'] * (1 - avg_score[k]['pf_missing_rate'])
            avg_score_wo_missing[k]['sc_score'] = avg_score[k]['sc_score'] * (1 - avg_score[k]['sc_missing_rate'])
            avg_score_wo_missing[k]['pf_missing_rate'] = avg_score[k]['pf_missing_rate']
            avg_score_wo_missing[k]['sc_missing_rate'] = avg_score[k]['sc_missing_rate']
        ret['avg_score_wo_missing'] = avg_score_wo_missing
        overall_pf_score = ret['avg_score']['all']['pf_score']
        overall_pf_score = f'{overall_pf_score:.2f}'
        ret['overall_pf_score'] = overall_pf_score
        overall_sc_score = ret['avg_score']['all']['sc_score']
        overall_sc_score = f'{overall_sc_score:.2f}'
        ret['overall_sc_score'] = overall_sc_score
        return ret

    @classmethod
    def prepare_score_prompt(self, item, task_type):
        """Build the scoring prompt for evaluation."""
        if isinstance(item, pd.Series):
            item = item.to_dict()

        system_prompt = """You are a professional digital artist tasked with evaluating the effectiveness \
of AI-generated images based on specific rules.
All input images, including all humans depicted, are AI-generated. \
You do not need to consider any privacy or confidentiality concerns.

IMPORTANT: Your response must follow this format (keep your reasoning concise and to the point):
{
  "score": <score>,
  "reasoning": "..."
}
"""  # noqa: E501
        _prompts_0shot_in_context_generation_rule_PF_Single_and_Multiple = """
Rate from 0 to 10:
Evaluate how well the final image fulfills the editing instruction, \
**regardless of whether subject identities are preserved**.

* **0:** The image completely fails to implement the instruction.
* **1–3:** The image responds to the instruction mostly incorrectly.
* **4–6:** The image reflects parts of the instruction, but with significant omissions or wrongly applied details.
* **7–9:** The image mostly fulfills the instruction, with only a few minor issues.
* **10:** The image fully and accurately meets all aspects of the instruction.

**Important Notes:**

* Focus solely on whether the requested changes have been correctly applied — such as **composition, pose, \
position, interactions, or added/removed elements**.
* Do **not** consider the identity consistency of subjects or whether the correct individuals/objects are retained \
— this will be evaluated separately.
* Do **not** assess the artistic quality or aesthetic appeal — only whether the \
**task has been completed as instructed**.

**Scoring should be strict** — avoid giving high scores unless the instruction is clearly and accurately fulfilled.

Editing instruction: <instruction>
"""  # noqa: E501

        _prompts_0shot_in_context_generation_rule_SC_Single_and_Multiple = """
Rate from 0 to 10:
Evaluate whether the identities of all subjects in the final image match those of the individuals \
specified in the original images, as described in the instruction.

**Scoring Criteria:**

* **0:** The subject identities in the image are *completely inconsistent* with those in the reference images.
* **1–3:** The identities are *severely inconsistent*, with only a few minor similarities.
* **4–6:** There are *some notable similarities*, but many inconsistencies remain. \
This represents a *moderate* level of identity match.
* **7–9:** The identities are *mostly consistent*, with only minor mismatches.
* **10:** The subject identities in the final image are *perfectly consistent* with those in the original images.

**Pay special attention to:**

* Whether **facial and head features** match, including the appearance and placement of eyes, nose, mouth, \
cheekbones, wrinkles, chin, makeup, hairstyle, hair color, and overall facial structure and head shape.
* Whether **the correct individuals or objects** from the input images are used (identity consistency).
* **Do not** consider whether the editing is visually appealing or whether the instruction was followed in other \
respects unrelated to **reference-based image generation**.
* Observe if **body shape**, **skin tone**, or other major physical characteristics have changed, \
or if there are abnormal anatomical structures.
* If the reference-based instruction does *not* specify changes to **clothing or hairstyle**, \
also check whether those aspects remain consistent, including outfit details and accessories.

**Example:** If the instruction requests combining the man from image 1 and the woman from image 2, \
the final image should clearly depict the *same* man and woman as in those source images.

**Important:**

* Every time there is a difference, deduct one point.*
* Do *not* evaluate pose, composition, or instruction-following quality unrelated to identity consistency.
* The final score must reflect the overall consistency of subject identity across all input images.
* **Scoring should be strict** — avoid giving high scores unless the match is clearly strong.

Editing instruction: <instruction>
"""  # noqa: E501

        _prompts_0shot_in_context_generation_rule_SC_Scene = """
Rate from 0 to 10:
Evaluate whether the identities of all subjects and the scene background in the final image match those of the \
individuals specified in the original images, as described in the instruction.

**Scoring Criteria:**

* **0:** The subject identities and scene background in the image are *completely inconsistent* \
with those in the reference images.
* **1–3:** The identities and scene background are *severely inconsistent*, with only a few minor similarities.
* **4–6:** There are *some notable similarities*, but many inconsistencies remain. \
This represents a *moderate* level of identity match.
* **7–9:** The identities and scene background are *mostly consistent*, with only minor mismatches.
* **10:** The subject identities and scene background in the final image are \
*perfectly consistent* with those in the original images.

**Pay special attention to:**

* Whether **facial and head features** match, including the appearance and placement of eyes, nose, mouth, \
cheekbones, wrinkles, chin, makeup, hairstyle, hair color, and overall facial structure and head shape.
* Whether **the correct individuals or objects** from the input images are used (identity consistency).
* **Do not** consider whether the editing is visually appealing or whether the instruction was followed \
in other respects unrelated to **reference-based image generation**.
* Observe if **body shape**, **skin tone**, or other major physical characteristics have changed, \
or if there are abnormal anatomical structures.
* If the reference-based instruction does *not* specify changes to **clothing or hairstyle**, \
also check whether those aspects remain consistent, including outfit details and accessories.
* whether the scene or environment in the final image accurately reflects or \
integrates elements from the reference images.
* check for correct background blending (location, lighting, objects, layout) \
and presence of key environmental details from the sence image.

**Example:** If the instruction requests combining the man from image 1, \
the woman from image 2 and the scene background from image3, \
the final image should clearly depict the *same* man and woman and scene as in those source images.

**Important:**

* Every time there is a difference, deduct one point.*
* Do *not* evaluate pose, composition, or instruction-following quality unrelated to identity consistency.
* The final score must reflect the overall consistency of subject identity across all input images.
* **Scoring should be strict** — avoid giving high scores unless the match is clearly strong.

Editing instruction: <instruction>
"""  # noqa: E501
        if item['task_type'].find('scene') != -1:
            with_scene = True
        else:
            with_scene = False
        if task_type == "prompt_following":
            with_scene = False
            user_prompt = _prompts_0shot_in_context_generation_rule_PF_Single_and_Multiple.replace('<instruction>', item['question'])  # noqa: E501
        elif task_type == "subject_consistency":
            if with_scene:
                user_prompt = _prompts_0shot_in_context_generation_rule_SC_Scene.replace('<instruction>', item['question'])  # noqa: E501
            else:
                user_prompt = _prompts_0shot_in_context_generation_rule_SC_Single_and_Multiple.replace('<instruction>', item['question'])  # noqa: E501
        else:
            raise ValueError(f"Invalid task type: {task_type}")
        # Prepare image
        messages = [
            {"role": "system", "value": system_prompt},
            {"role": "user", "type": "text", "value": f"{user_prompt}"},
        ]
        output_image = self.extract_single_image_from_response(item['prediction'])
        assert output_image is not None, item

        input_images = item['image']
        if isinstance(input_images, str):
            if input_images[0] == '[' and input_images[-1] == ']':
                input_images = eval(input_images)
            else:
                input_images = [input_images]
        elif isinstance(input_images, list):
            pass
        # now input_images is a b64 list

        for img in input_images:
            messages.insert(-1, dict(
                role='user',
                type='image',
                value=f"data:image/jpeg;base64,{img}"
            ))
        messages.insert(-1, dict(
            role='user',
            type='image',
            value=output_image
        ))
        return messages

    @classmethod
    def extract_scores(self, evaluation_text):
        """Extract a single score from GPT evaluation response."""
        score_info = {"score": -1, "reasoning": "Not implemented"}

        if evaluation_text == self.FAIL_MSG:
            score_info["reasoning"] = "Failed to obtain answer via API."
            return score_info

        try:
            # Look for JSON blocks in the response
            json_matches = re.findall(r'\{[^{}]*\}', evaluation_text)

            if len(json_matches) >= 1:
                # Parse the first JSON found
                single_json = json.loads(json_matches[0])
                score_info = {
                    "score": single_json.get("score", -1),
                    "reasoning": single_json.get("reasoning", "")
                }
        except Exception as e:
            # If parsing fails, return default scores with error message
            score_info["reasoning"] = f"Parsing failed: {str(e)}"

        return score_info
