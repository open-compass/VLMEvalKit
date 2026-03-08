import pandas as pd
from .gen_base import GenBaseImageDataset
from vlmeval.smp import load, dump, toliststr, track_progress_rich, get_intermediate_file_path
import os
import os.path as osp
import numpy as np
from .utils.rise import *
import re

SKIP_MSG = 'Preliminary judge failed, skip this sample.'

subtask_dic = {
    "Temp": [
        "Life Progression",
        "Material Progression",
        "Environmental Cycles",
        "Societal Transformation",
    ],
    "Causal": [
        "Structural Deformation",
        "State Transition",
        "Chemical and Biological Transformation",
        "Physics Manifestation",
    ],
    "Spa": [
        "Component Assembly",
        "Object Arrangement",
        "Viewpoint Generation",
        "Structural Inference",
        "Layout Reasoning",
    ],
    "Logic": ["Pattern Prediction", "Mathematical Derivation", "Puzzle Solving"],
}

class RISEBench(GenBaseImageDataset):
    """RISEBench dataset for image editing task."""

    TYPE = 'TI2I'
    NUM_GENERATIONS = 1
    DEFAULT_JUDGE = 'gpt-4.1'

    DATASET_URL = {
        'RISEBench': 'https://opencompass.openxlab.space/utils/GenEval/risebench.tsv',
    }

    DATASET_MD5 = {
        'RISEBench': '890a0c3fe8bf3591115c89bdb59f77e2',
    }

    def build_prompt(self, line):
        """Build prompt for OmniContext task with input images and question."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['instruction']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    @classmethod
    def eval_vanilla(self, judge_model, item, **kwargs):
        instruct = item['instruction']
        index = item['index']
        category = item['category']
        
        ret = {'judge_cons': self.FAIL_MSG, 'judge_rea': self.FAIL_MSG, 'judge_plau': self.FAIL_MSG}
        if category == 'logical_reasoning':
            ret.pop('judge_plau')

        output_img = self.extract_single_image_from_response(item['prediction'])
        if output_img is None:
            print(f'Failed to generate image for index {index}, skip evaluation.')
            return ret 
        
        judge_rea_require_img = False

        if category in ['temporal_reasoning', 'causal_reasoning']:
            img1 = item['image']
            reference = item['reference']
            if "reference_img" in item and not pd.isna(item['reasoning_img']):
                judge_rea_require_img = True
                prompt_rea = prompt_reasoning_w_input.format(instruct=instruct, reference=reference)
            else:
                prompt_rea = prompt_reasoning.format(instruct=instruct, reference=reference)

            prompt_cons = prompt_consist.format(instruct=instruct)
            prompt_qua = prompt_generation

        elif category == 'spatial_reasoning':
            img1 = item['image']
            if "reference_img" in item and not pd.isna(item['reference_img']):
                judge_rea_require_img = True
                img1 = item['reference_img']
                prompt_rea = prompt_spatial_ref_img.format(instruct=instruct)
            elif not pd.isna(item['reasoning_img']):
                judge_rea_require_img = True
                reference = item['reference']
                prompt_rea = prompt_spatial_ref_w_input.format(instruct=instruct, reference=reference)
            else:
                reference = item['reference']
                prompt_rea = prompt_spatial_ref.format(instruct=instruct, reference=reference)

            prompt_cons = prompt_spatial_cons.format(instruct=instruct)
            prompt_qua = prompt_spatial_qual

        elif category == 'logical_reasoning':
            if "reference_txt" in item and not pd.isna(item['reference_txt']):
                img1 = item['image']
                reference = item['reference_txt']
                prompt_cons = prompt_logical_cons_ans.format(instruct=instruct, reference=reference)
                prompt_rea = prompt_logical_txt.format(instruct=instruct, reference=reference)
            elif "reference_img" in item and not pd.isna(item['reference_img']):
                judge_rea_require_img=True
                img1 = item['reference_img']
                prompt_cons = prompt_logical_cons.format(instruct=instruct)
                if 'reasoning_wo_ins' in item:
                    prompt_rea = prompt_logical_img_wo_q
                else:
                    prompt_rea = prompt_logical_img.format(instruct=instruct)

        if isinstance(img1, str):
            if img1[0] == '[' and img1[-1] == ']':
                img1 = eval(img1)
            else:
                img1 = [img1]
        elif isinstance(img1, list):
            pass

        retry = 3
        # Judge consitency
        if 'consistency_free' in item and not pd.isna(item['consistency_free']):
            consist_judge = None
            print('Consistency Judgement not required. Ignore.')
        else:
            message = []
            text = {'type': 'text', 'value': prompt_cons}
            image1 = {
                'type': 'image',
                'value': f"data:image/jpeg;base64,{img1[0]}",
            }
            image2 = {
                'type': 'image',
                'value': output_img,
            }
            message.append(text)
            message.append(image1)
            message.append(image2)

            consist_judge = judge_model.generate(message, **kwargs)

            # retry, if fail, pass and return
            while consist_judge == self.FAIL_MSG and retry > 0:
                retry -= 1
                consist_judge = judge_model.generate(message, **kwargs)
            if consist_judge != self.FAIL_MSG:
                ret['judge_cons'] = consist_judge

        retry = 3
        # Judge reasoning
        if judge_rea_require_img:
            message2 = [
                {'type': 'text', 'value': prompt_rea},
                {'type': 'image','value': f"data:image/jpeg;base64,{img1[0]}"},
                {'type': 'image','value': output_img}
            ]
        else:
            message2 = [
                {'type': 'text', 'value': prompt_rea}, 
                {'type': 'image', 'value': output_img},
            ]

        rea_judge = judge_model.generate(message2)

        # retry, if fail, pass and return
        while rea_judge == self.FAIL_MSG and retry > 0:
            retry -= 1
            rea_judge = judge_model.generate(message2, **kwargs)
        if rea_judge != self.FAIL_MSG:
            ret['judge_rea'] = rea_judge
        
        retry = 3
        # Judge Plausibility
        if category in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning']:
            message3 = [{'type': 'text', 'value': prompt_qua}, {
                'type': 'image',
                'value': output_img,
            }]
            plau_judge = judge_model.generate(message3)

            while plau_judge == self.FAIL_MSG and retry > 0:
                retry -= 1
                plau_judge = judge_model.generate(message3, **kwargs)

            if plau_judge != self.FAIL_MSG:
                ret['judge_plau'] = plau_judge
        return ret

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
        tgt_file = get_intermediate_file_path(eval_file, f'_{judge}_raw', 'tsv')
        judge_file = get_intermediate_file_path(eval_file, f'_{judge}', 'tsv')
        rating_file = get_intermediate_file_path(eval_file, f'_{judge}_rating', 'json')

        judge_kwargs['temperature'] = 0.0
        judge_kwargs['img_size'] = 512
        judge_kwargs['max_tokens'] = 4096
        judge_kwargs['timeout'] = 240
        model = build_judge(**judge_kwargs)

        eval_df = load(eval_file)
        eval_df['index'] = eval_df['index'].astype(str)

        # run scoring if not already present
        if not osp.exists(tgt_file):
            # resume support: load tmp results and drop failures
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            metrics = ['judge_cons', 'judge_rea', 'judge_plau']
            res = {k: v for k, v in res.items() if np.all([v[metric] != self.FAIL_MSG for metric in metrics])}

            todo_mask = ~eval_df['index'].isin(res.keys())
            data_un = eval_df[todo_mask].reset_index(drop=True)

            lt = len(data_un)
            # Build plain string prompts to avoid BaseAPI 'value' key issues
            if lt > 0:
                samples = [data_un.iloc[i] for i in range(len(data_un))]
                indices = [x['index'] for x in samples]
                jobs = [dict(judge_model=model, item=sample) for sample in samples]
                _ = track_progress_rich(
                    self.eval_vanilla,      # callable(judge_model: BaseAPI, sample: dict) -> dict
                    jobs,      # iterable of dicts: {'judge_model': BaseAPI, 'sample': dict}
                    keys=indices,        # map results by 'index'
                    save=tmp_file,    # resume file
                    nproc=nproc,
                    dec='Evaluating RISEBench'
                )
                score_map = load(tmp_file) if osp.exists(tmp_file) else {}
                score_map.update(res)
            else:
                score_map = res

            for k in ['judge_cons', 'judge_rea', 'judge_plau']:
                eval_df[k] = [score_map[idx].get(k, None) for idx in eval_df['index']]
            dump(eval_df, tgt_file)
            results = eval_df
        else:
            results = load(tgt_file)

        if osp.exists(tmp_file):
            os.remove(tmp_file)
        final_score = self.cal_metric(results, judge_file)
        dump(final_score, rating_file)
        return final_score

    def cal_metric(self, data, judge_file):
        # scores, judge_combine, judge_cons, judge_reas, judge_qua = [], [], [], [], []
        reasoning, consistency, plausibility, succeed = [], [], [], []
        def legal_score(s):
            return s in [1, 2, 3, 4, 5]
        
        for i, row in data.iterrows():
            r = extract(row['judge_rea'])
            c = extract(row['judge_cons'])
            p = extract(row['judge_plau'])
            cate = row['category']
            if cate == 'logical_reasoning':
                r = 4 * r + 1 if r in [0, 1] else None
                c = 4 * c + 1 if c in [0, 1] else None
                if legal_score(r) and legal_score(c):
                    s = True
                else:
                    s = False
            else:
                if legal_score(r) and legal_score(c) and legal_score(p):
                    s = True
                else:
                    s = False
            reasoning.append(r)
            consistency.append(c)
            plausibility.append(p)
            succeed.append(s)

        data['reasoning'] = reasoning
        data['consistency'] = consistency
        data['plausibility'] = plausibility
        data['succeed'] = succeed   

        data['score'] = data.apply(calculate_score, axis=1)
        data['complete'] = data.apply(calculate_completion, axis=1)

        dump(data, judge_file)

        df_causal = data[data['category'] == 'causal_reasoning']
        df_temporal = data[data['category'] == 'temporal_reasoning']
        df_spatial = data[data['category'] == 'spatial_reasoning']
        df_logical = data[data['category'] == 'logical_reasoning']

        score_final = data['score'].mean()
        completion_rate = data['complete'].mean()

        # calculate score and accuracy per main task
        temporal_final, temporal_comp_rate = df_temporal['score'].mean(), df_temporal['complete'].mean()
        causal_final, causal_comp_rate = df_causal['score'].mean(), df_causal['complete'].mean()
        spatial_final, spatial_comp_rate = df_spatial['score'].mean(), df_spatial['complete'].mean()
        logical_final, logical_comp_rate = df_logical['score'].mean(), df_logical['complete'].mean()

        reasoning_average = data['reasoning'].mean()
        img_consist_average = data['consistency'].mean()
        data_wo_logical = data[data['category'] != 'logical_reasoning']
        generation_quality = data_wo_logical['plausibility'].mean()

        temp_rea_avg, temp_cons_avg, temp_qua_avg = df_temporal['reasoning'].mean(), df_temporal['consistency'].mean(), df_temporal['plausibility'].mean()
        cau_rea_avg, cau_cons_avg, cau_qua_avg = df_causal['reasoning'].mean(), df_causal['consistency'].mean(), df_causal['plausibility'].mean()
        spa_rea_avg, spa_cons_avg, spa_qua_avg = df_spatial['reasoning'].mean(), df_spatial['consistency'].mean(), df_spatial['plausibility'].mean()
        logic_rea_avg, logic_cons_avg = df_logical['reasoning'].mean(), df_logical['consistency'].mean()

        def trans_to_percent(s):
            return 25 * (s - 1)

        final_score = dict(
            Overall={'avg_score': trans_to_percent(score_final), 'accuracy': completion_rate * 100},
            Temporal={'avg_score': trans_to_percent(temporal_final), 'accuracy': temporal_comp_rate * 100},
            Causal={'avg_score': trans_to_percent(causal_final), 'accuracy': causal_comp_rate * 100},
            Spatial={'avg_score': trans_to_percent(spatial_final), 'accuracy': spatial_comp_rate * 100},
            Logical={'avg_score': trans_to_percent(logical_final), 'accuracy': logical_comp_rate * 100},
            Overall_Reasoning={'avg_score': trans_to_percent(reasoning_average)},
            Overall_ApprConsistency={'avg_score': trans_to_percent(img_consist_average)},
            Overall_VisualPlausibility_total={'avg_score': trans_to_percent(generation_quality)},
            Temporal_Reasoning = {'avg_score': trans_to_percent(temp_rea_avg)},
            Temporal_Consistency = {'avg_score': trans_to_percent(temp_cons_avg)},
            Temporal_Quality = {'avg_score': trans_to_percent(temp_qua_avg)},
            Causal_Reasoning = {'avg_score': trans_to_percent(cau_rea_avg)},
            Causal_Consistency = {'avg_score': trans_to_percent(cau_cons_avg)},
            Causal_Quality = {'avg_score': trans_to_percent(cau_qua_avg)},
            Spatial_Reasoning = {'avg_score': trans_to_percent(spa_rea_avg)},
            Spatial_Consistency = {'avg_score': trans_to_percent(spa_cons_avg)},
            Spatial_Quality = {'avg_score': trans_to_percent(spa_qua_avg)},
            Logical_Reasoning = {'avg_score': trans_to_percent(logic_rea_avg)},
            Logical_Consistency = {'avg_score': trans_to_percent(logic_cons_avg)},
        )
        final_score['overall_score'] = final_score['Overall']['avg_score']
        final_score['overall_acc'] = final_score['Overall']['accuracy']
        return final_score


def extract(answer):
    if answer == RISEBench.FAIL_MSG or answer is None or pd.isna(answer):
        return None
    matches = re.findall(r'\*?\*?Final Score\*?\*?:?\s*([\d*\s,\n]*)', answer, re.IGNORECASE)
    numbers = []
    if matches:
        for match in matches:
            extracted_numbers = re.findall(r'\d+', match.replace('\n', ' '))
            if extracted_numbers:
                numbers.extend(map(int, extracted_numbers))
                break
        if numbers != []:
            return int(numbers[0])

    matches = re.findall(r'\*?\*?Final Scores\*?\*?:?\s*([\d*\s,\n]*)', answer, re.IGNORECASE)
    numbers = []
    if matches:
        for match in matches:
            extracted_numbers = re.findall(r'\d+', match.replace('\n', ' '))
            if extracted_numbers:
                numbers.extend(map(int, extracted_numbers))
                break
        return int(numbers[0])
    else:
        return None

def calculate_score(row):
    # calculate weighted score. weighted score is not used, please refer to Accuracy(Completion Rate)
    try:
        if row['category'] in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning']:
            if 'consistency_free' in row and row['consistency_free']:
                score = 0.2 * row['plausibility'] + 0.8 * row['reasoning']
            else:
                score = 0.3 * row['consistency'] + 0.5 * row['reasoning'] + 0.2 * row['plausibility']

        elif row['category'] == 'logical_reasoning':
            score = 0.3 * row['consistency'] + 0.7 * row['reasoning']
        if row['reasoning'] == 1:
            score = score * 0.5
            score = 1 if score < 1 else score
    except Exception as e:
        print(
            f"Weighted score calculation failed for row {row['index']}, reasoning score: {row['reasoning']}, "
            f"appr consistency score: {row['consistency']}, visual plausibility score: {row['plausibility']}")
        print(e)
        return 0
    return score

def calculate_completion(row):
    if row['category'] in ['temporal_reasoning', 'causal_reasoning', 'spatial_reasoning']:
        return (
            1
            if row['consistency'] == 5 and row['reasoning'] == 5 and row['plausibility'] == 5
            else 0
        )
    elif row['category']=='logical_reasoning':
        return (
            1 if row['consistency'] == 5 and row['reasoning'] == 5 else 0
        )
