from .image_base import ImageBaseDataset
import numpy as np
import pandas as pd
from ..smp import *
from .utils import build_judge, DEBUG_MESSAGE
from ..utils import track_progress_rich

prompt_dict = {
    # Subjective Judge [GPT-4o reference]
    'subjective':"""
Please act as an impartial judge and evaluate the quality of two responses provided by AI assistants to the user prompt.

Your task is to carefully assess two responses based on provided instructions and evaluation criteria. After evaluating both responses, determine which response features better quality and better meets the criteria. If both responses are similar or nearly identical in quality, you should indicate a tie. Avoid position bias toward the first or second response.

Suggested Steps for Evaluation:
1. Review both responses independently and then carefully compare their strengths and weaknesses. A good response should feature good language quality, follow the user instruction and meet as many criteria as possible.
2. After completing the first evaluation, swap the positions of response A and B and repeat Step 1 and get the 2nd evaluation outcome. This helps to mitigate the potential position bias.
3. After completing both evaluations (in the original and reversed order), combine your analysis and provide a final conclusion based on the overall assessment. If both responses are relatively similar, or the differences are minimal and hard to distinguish, your conclusion should indicate a tie ([[A=B]]). 

Your **conclusion** should be one of the following options (A, B are of the original order):
1. [[A>>B]]: Response A is clearly better than Response B.
2. [[A>B]]: Response A is slightly better than Response B.
3. [[A=B]]: Response A is nearly identical to Response B.
4. [[B>A]]: Response B is slightly better than Response A.
5. [[B>>A]]: Response B is clearly better than Response A.

User Instruction:\n[INSTRUCTIONS]\n{instructions}\n[END INSTRUCTIONS]\n\n
Repsonse A:\n[RESPONSE A]\n{reference_answer_by_gpt4o}\n[END RESPONSE A]\n\n
Response B:\n[RESPONSE B]\n{prediction}\n[END RESPONSE B]\n\n
Evaluation Criteria:\n[CRITERIA]\n{criteria}\n[END CRITERIA]\n\n

Your output should include:
1. Conclusion: Your final conclusion based on the overall assessment.
2. Reasoning: Your reasoning process and analysis of the two responses.

Your output should follow the following format (CONCLUSION should be one of the five options: A>>B, A>B, A=B, B>A, B>>A):

Final Conclusion: [[CONCLUSION]]
Reasoning Process: [REASONING]\n
""",

    # Criteria Alignment w/o GT
    'objective_without_gt':"""
Please act as an impartial judge and evaluate the **Criteria Alignment** of the two responses provided by AI assistants to the user prompt. The responses were generated based on the provided instructions and visual input from images. 

Suggested Steps for Evaluation:
1. Evaluate **Criteria Alignment** of both responses based on the criteria.   
    • If a criterion consist of **X aspects**, each aspect is worth **10 / X points**.
    • For each aspect, there may be multiple sub-criteria. If there are **Y sub-criteria for the aspect**, each sub-criterion worths **10 / (X * Y) points**.
2. Assign a total score out of 10 for each response.

User Instruction:\n[INSTRUCTIONS]\n{instructions}\n[END INSTRUCTIONS]\n\n
Repsonse A:\n[RESPONSE A]\n{reference_answer_by_gpt4o}\n[END RESPONSE A]\n\n
Response B:\n[RESPONSE B]\n{prediction}\n[END RESPONSE B]\n\n
Criteria:\n[CRITERIA]\n{criteria}\n[END CRITERIA]\n\n

Your output should evaluate alignment scores of each response and end with a conclusion in the following format (The full score is 10. X, Y are alignment scores for Response A and B):

Response A Alignment Score: X/10
Response B Alignment Score: Y/10\n
""",

    # Criteria Alignment w. GT
    'objective_with_gt':"""
Please act as an impartial judge and evaluate the **Criteria Alignment** of the two responses provided by AI assistants to the user prompt. The responses were generated based on the provided instructions and visual input from images. There is also a ground truth corresponding to the instructions provided for reference. 
Take this context into account when making your judgment.

Steps for Evaluation:
1. Evaluate **Criteria Alignment** of both responses based on the criteria and the ground truth.   
    • If a criterion consist of **X aspects**, each aspect is worth **10 / X points**.
    • For each aspect, there may be multiple sub-criteria. If there are **Y sub-criteria for the aspect**, each sub-criterion worths **10 / (X * Y) points**.
2. Assign a total score out of 10 for each response.

User Instruction:\n[INSTRUCTIONS]\n{instructions}\n[END INSTRUCTIONS]\n\n
Ground Truth:\n[GROUND TRUTH]\n{groundtruth}\n[END GROUND TRUTH]\n\n
Repsonse A:\n[RESPONSE A]\n{reference_answer_by_gpt4o}\n[END RESPONSE A]\n\n
Response B:\n[RESPONSE B]\n{prediction}\n[END RESPONSE B]\n\n
Criteria:\n[CRITERIA]\n{criteria}\n[END CRITERIA]\n\n

Your output should evaluate alignment scores of each response and end with a conclusion in the following format (The full score is 10. X, Y are alignment scores for Response A and B):

Response A Alignment Score: X/10
Response B Alignment Score: Y/10\n
""",
}

def is_criteria_valid(criteria):     
    import re
    for value in criteria.values():
        if value == '\\' or value == '' or not re.search('[a-zA-Z]', value):
            return False
    return True

def build_prompt(line):
    try:
        criteria = eval(line['criteria'])
    except:
        criteria = line['criteria']

    if isinstance(criteria, dict):
        new_criteria = {}
        for k in criteria:
            if 'subjective' in k.lower():
                new_criteria['subjective'] = criteria[k]
            else:
                new_criteria['objective'] = criteria[k] 
    else:
        assert isinstance(criteria, str)
        new_criteria = {'subjective': criteria}
    criteria = new_criteria
    assert 'subjective' in criteria, 'No subjective criteria found in the criteria dict'
    
    prompts = {}
    prompts['subjective'] = prompt_dict['subjective'].format(
        instructions=line['question'],
        criteria=criteria['subjective'],
        reference_answer_by_gpt4o=line['reference_answer_by_gpt4o'],
        prediction=line['prediction']
    )
    if 'objective' in criteria:
        if 'ground_truth' in line and (not pd.isna(line['ground_truth'])) and line['ground_truth'] != '':
            prompts['objective'] = prompt_dict['objective_with_gt'].format(
                instructions=line['question'],
                criteria=criteria['objective'],
                groundtruth=line['ground_truth'],
                reference_answer_by_gpt4o=line['reference_answer_by_gpt4o'],
                prediction=line['prediction'])
        else:
            prompts['objective'] = prompt_dict['objective_without_gt'].format(
                instructions=line['question'],
                criteria=criteria['objective'],
                reference_answer_by_gpt4o=line['reference_answer_by_gpt4o'],
                prediction=line['prediction'])
    return prompts

  
def Generate_Creation_MMBench_judge(model, prompt):
    assert isinstance(prompt, dict)
    response = {}
    for key in prompt.keys():
        response[key] = model.generate(prompt[key])
    return response


def extract_subjective(inp):
    lines = inp.split('\n')
    for line in lines:
        line = line.upper()
        if line.startswith('FINAL CONCLUSION:'):
            rem = line.split('FINAL CONCLUSION:')[1].strip()
            rem = rem.split('[[')[1].split(']]')[0].strip()
            cands = [
                'A>>B', 'A>B', 'A=B', 'B>A', 'B>>A', 
                'B<<A', 'B<A', 'B=A', 'A<B', 'A<<B'
            ]
            if rem in cands:
                return rem
    return None


def extract_objective(inp):
    # Response A Alignment Score: X/10
    if pd.isna(inp) or inp is None or inp == '':
        return 'NO_OBJECTIVE'
    lines = inp.split('\n')
    a_score, b_score = None, None
    for line in lines:
        line = line.upper()
        if line.startswith('RESPONSE A ALIGNMENT SCORE:'):
            rem = line.split('RESPONSE A ALIGNMENT SCORE:')[1].strip()
            rem = rem.split('/')[0].strip()
            try:
                a_score = float(rem)
            except:
                continue
        elif line.startswith('RESPONSE B ALIGNMENT SCORE:'):
            rem = line.split('RESPONSE B ALIGNMENT SCORE:')[1].strip()
            rem = rem.split('/')[0].strip()
            try:
                b_score = float(rem)
            except:
                continue
    if a_score is not None and b_score is not None and (0 <= a_score <= 10) and (0 <= b_score <= 10):
        return f'{a_score}|{b_score}' 
    else:
        return None
    

def Creation_MMBench_extract(judge_response_pkl, org_data):
    import copy as cp
    data = cp.deepcopy(org_data)
    data['subjective_judge'] = [judge_response_pkl[idx]['subjective'] for idx in data['index']]
    data['objective_judge'] = [judge_response_pkl[idx].get('objective', None) for idx in data['index']]
    data['subjective_score'] = [extract_subjective(x) for x in data['subjective_judge']]
    data['objective_score'] = [extract_objective(x) for x in data['objective_judge']]
    return data


def get_dimension_rating(score_file_name, rev=False):
    def get_pw_score(text):
        if 'A<<B' in text or 'B>>A' in text:
            return 2
        elif 'A<B' in text or 'B>A' in text:
            return 1
        elif 'A=B' in text or 'B=A' in text:
            return 0
        elif 'A>B' in text or 'B<A' in text:
            return -1
        elif 'A>>B' in text or 'B<<A' in text:
            return -2
        else:
            return None
        
    score_file = load(score_file_name)
    base_dict = {'sub_valid': 0, 'sub_missing': 0, 'sub_score': [], 'obj_valid': 0, 'obj_missing': 0, 'obj_ref_score': [], 'obj_score': [],  'obj_rel_score': []}
    return_dict = {'overall': cp.deepcopy(base_dict)}

    for idx, item in score_file.iterrows():
        task_name = item['task_name']
        if task_name not in return_dict.keys():
            return_dict[task_name] = cp.deepcopy(base_dict)
        
        if not pd.isna(item['subjective_score']):
            for k in ['overall', task_name]:
                return_dict[k]['sub_valid'] += 1
                return_dict[k]['sub_score'].append(get_pw_score(item['subjective_score']))
        else:
            return_dict['overall']['sub_missing'] += 1
            return_dict[task_name]['sub_missing'] += 1

        if item['objective_score'] == 'NO_OBJECTIVE':
            continue
        elif not pd.isna(item['objective_score']):
            score = item['objective_score']
            assert '|' in score
            ref_score, score = [float(x) for x in score.split('|')]
            for k in ['overall', task_name]:
                return_dict[k]['obj_valid'] += 1
                return_dict[k]['obj_score'].append(score)
                return_dict[k]['obj_ref_score'].append(ref_score)
                # return_dict[k]['obj_rel_score'].append(score / ref_score * 10)
        else:
            return_dict['overall']['obj_missing'] += 1
            return_dict[task_name]['obj_missing'] += 1
            
    final_res = {}
    
    for k, v in return_dict.items():
        res = {}
        res['sub_parse_ok'] = v['sub_valid'] / (v['sub_valid'] + v['sub_missing'])
        dist = defaultdict(lambda: 0)
        for x in v['sub_score']:
            dist[x] += 1
        assert len(dist) <= 5 and sum(list(dist.values())) == v['sub_valid']
        if v['sub_valid']:
            res['sub_dist'] = {k: dist[k] / v['sub_valid'] for k in [-2, -1, 0, 1, 2]}
            res['sub_reward'] = (-100 * dist[-2] - 50 * dist[-1] + 50 * dist[1] + 100 * dist[2]) / v['sub_valid']
    
        if v['obj_valid'] + v['obj_missing']:
            res['obj_parse_ok'] = v['obj_valid'] / (v['obj_valid'] + v['obj_missing'])
            if v['obj_valid']:
                res['obj_score'] = sum(v['obj_score']) / v['obj_valid']
                # res['obj_rel_score'] = sum(v['obj_rel_score']) / v['obj_valid']
                res['obj_ref_score'] = sum(v['obj_ref_score']) / v['obj_valid']
        final_res[k] = res

    final_res['raw'] = return_dict
    return final_res


def merge_dual(raw, raw_dual):
    final_res = {}
    for k, v in raw.items():
        # merge dual: {'sub_valid': 0, 'sub_missing': 0, 'sub_score': [], 'obj_valid': 0, 'obj_missing': 0, 'obj_ref_score': [], 'obj_score': [],  'obj_rel_score': []}
        dual_v = raw_dual[k]
        v['sub_valid'] += dual_v['sub_valid'] 
        v['sub_missing'] += dual_v['sub_missing']
        v['sub_score'].extend([-x for x in dual_v['sub_score']])
        v['obj_valid'] += dual_v['obj_valid']
        v['obj_missing'] += dual_v['obj_missing']
        v['obj_score'].extend(dual_v['obj_ref_score'])
        v['obj_ref_score'].extend(dual_v['obj_score'])
        raw[k] = v

        res = {}
        res['sub_parse_ok'] = v['sub_valid'] / (v['sub_valid'] + v['sub_missing'])
        dist = defaultdict(lambda: 0)
        for x in v['sub_score']:
            dist[x] += 1
        assert len(dist) <= 5 and sum(list(dist.values())) == v['sub_valid']
        res['sub_dist'] = {k: dist[k] / v['sub_valid'] for k in [-2, -1, 0, 1, 2]}
        res['sub_reward'] = (-100 * dist[-2] - 50 * dist[-1] + 50 * dist[1] + 100 * dist[2]) / v['sub_valid']

        if v['obj_valid'] + v['obj_missing']:
            res['obj_parse_ok'] = v['obj_valid'] / (v['obj_valid'] + v['obj_missing'])
            if v['obj_valid']:
                res['obj_score'] = sum(v['obj_score']) / v['obj_valid']
                # res['obj_rel_score'] = sum(v['obj_rel_score']) / v['obj_valid']
                res['obj_ref_score'] = sum(v['obj_ref_score']) / v['obj_valid']
        final_res[k] = res

    final_res['raw'] = raw
    return final_res


class CreationMMBenchDataset(ImageBaseDataset):

    TYPE = 'CreationVQA'
    DATASET_URL = {
        'LiveMMBench_Creation': ''
    }
    DATASET_MD5 = {}

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        # build_prompt, Generate_Creation_MMBench_judge,
        # Creation_MMBench_extract, get_dimension_rating
        rating_rev = None
        dual_eval = judge_kwargs.pop('dual_eval', True)
        if dual_eval:
            src = load(eval_file)
            tgt = load(eval_file)
            tgt['reference_answer_by_gpt4o'] = src['prediction']
            tgt['prediction'] = src['reference_answer_by_gpt4o']
            tgt_file_name = eval_file.replace('.xlsx', '_rev.xlsx')
            dump(tgt, tgt_file_name)
            judge_kwargs['dual_eval'] = False
            rating_rev = self.evaluate(tgt_file_name, **judge_kwargs)
        judge_kwargs.pop('dual_eval', None)

        suffix = '.' + eval_file.split('.')[-1]

        score_file = eval_file.replace(suffix, '_score.csv')
        tgt_file = eval_file.replace(suffix, '_rating.json')

        model = judge_kwargs.pop('model', 'gpt-4o-0806')
        model_name = model.split('/')[-1] if '/' in model else model
        tmp_file = eval_file.replace(suffix, f'_{model_name}.pkl')

        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(score_file):
            data = load(eval_file)
            lt = len(data)
            lines = [data.iloc[i] for i in range(len(data))]
            judge_kwargs['max_tokens'] = 4096
            model = build_judge(model=model, **judge_kwargs)
            assert model.working(), ('CreationMMBench evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

            prompts = [build_prompt(line) for line in lines]
            tups = [(model, prompt) for prompt in prompts]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                _ = track_progress_rich(
                    Generate_Creation_MMBench_judge,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
            ans = load(tmp_file)
            data = Creation_MMBench_extract(ans, data)
            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)

        if dual_eval:
            raw = rating['raw']
            rev_tgt_file = tgt_file.replace('rating.json', 'rev_rating.json')
            rev_raw = load(rev_tgt_file)['raw']
            merged_rating = merge_dual(raw, rev_raw)
            dump(merged_rating, tgt_file.replace('rating.json', 'merged_rating.json'))
            print(f"Rating:\n{rating['overall']}\n\nDual Rating:\n{merged_rating['overall']}")
            return merged_rating['overall']
        else:
            return rating['overall']