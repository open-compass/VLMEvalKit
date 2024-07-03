import os.path as osp
import pandas as pd
from vlmeval.evaluate.misc import build_judge
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
import numpy as np

FAIL_MSG = 'Failed to obtain answer via API.'

system_prompt = """
As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct 'groundtruth' answer, and the candidate answer will be provided to you.
Your assessment should range from 0 to 3, \
based solely on the semantic similarity between the groundtruth and the candidate answer, \
disregarding any grammatical differences.
A rating of 0 suggests no similarity, implying the candidate answer is entirely incorrect.
A rating of 1 suggests low similarity, meaning the candidate answer is largely incorrect.
A rating of 2 suggests high similarity, meaning the candidate answer is largely correct.
Lastly, a rating of 3 indicates complete similarity, which means the candidate answer is entirely correct.
Your response should be a single integer from 0, 1, 2, or 3.
"""

MMV_DIMENSIONS = {
    'CP': ['Video Topic', 'Video Emotion', 'Video Scene', 'Video Style'],
    'FP-S': ['OCR', 'Object Recognition', 'Attribute Recognition', 'Event Recognition', 'Human Motion', 'Counting'],
    'FP-C': ['Spatial Relationship', 'Human-object Interaction', 'Human Interaction'],
    'HL': ['Hallucination'],
    'LR': ['Structuralized Image-Text Understanding', 'Mathematical Calculation'],
    'AR': ['Physical Property', 'Function Reasoning', 'Identity Reasoning'],
    'RR': ['Natural Relation', 'Physical Relation', 'Social Relation'],
    'CSR': ['Common Sense Reasoning'],
    'TR': ['Counterfactual Reasoning', 'Causal Reasoning', 'Future Prediction'],
}
L3_DIMS = []
for k, v in MMV_DIMENSIONS.items():
    L3_DIMS.extend(v)

MMV_DIMENSIONS['Perception'] = []
MMV_DIMENSIONS['Reasoning'] = []
MMV_DIMENSIONS['Overall'] = []
for k in ['CP', 'FP-C', 'FP-S', 'HL']:
    MMV_DIMENSIONS['Perception'].extend(MMV_DIMENSIONS[k])
    MMV_DIMENSIONS['Overall'].extend(MMV_DIMENSIONS[k])
for k in ['LR', 'AR', 'RR', 'CSR', 'TR']:
    MMV_DIMENSIONS['Reasoning'].extend(MMV_DIMENSIONS[k])
    MMV_DIMENSIONS['Overall'].extend(MMV_DIMENSIONS[k])


def get_dimension_rating(data_path):
    data = load(data_path)
    coarse_rating = {k: [] for k in MMV_DIMENSIONS}
    fine_rating = {k: [] for k in L3_DIMS}

    for i in range(len(data)):
        cate = data.iloc[i]['dimensions']
        cates = eval(cate)

        for c in cates:
            fine_rating[c].append(data.iloc[i]['score'])

        for d in MMV_DIMENSIONS:
            if np.any([x in MMV_DIMENSIONS[d] for x in cates]):
                coarse_rating[d].append(data.iloc[i]['score'])

    coarse_all = {k: f'{np.mean([max(x, 0) for x in v]):.2f}' for k, v in coarse_rating.items()}
    coarse_valid = {k: f'{np.mean([x for x in v if x >= 0]):.2f}' for k, v in coarse_rating.items()}
    fine_all = {k: f'{np.mean([max(x, 0) for x in v]):.2f}' for k, v in fine_rating.items()}
    fine_valid = {k: f'{np.mean([x for x in v if x >= 0]):.2f}' for k, v in fine_rating.items()}
    return dict(coarse_all=coarse_all, coarse_valid=coarse_valid, fine_all=fine_all, fine_valid=fine_valid)


def build_prompt(item):
    tmpl = 'Question: {}\nGroundtruth answer: {}\nCandidate answer: {}\nYour response: '
    return tmpl.format(item['question'], item['answer'], item['prediction'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--judge', type=str, default='gpt-4-1106')
    parser.add_argument('--nproc', type=int, default=6)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


def MMBenchVideo_eval(data_file, **judge_kwargs):
    assert data_file.endswith('.xlsx'), 'data file should be an xlsx file'
    judge = judge_kwargs['model']
    nproc = judge_kwargs.pop('nproc', 4)

    tmp_file = data_file.replace('.xlsx', f'_{judge}_tmp.pkl')
    tgt_file = data_file.replace('.xlsx', f'_{judge}_rating.json')
    score_file = data_file.replace('.xlsx', f'_{judge}_score.xlsx')

    model = build_judge(system_prompt=system_prompt, **judge_kwargs)

    if not osp.exists(score_file):
        res = {} if not osp.exists(tmp_file) else load(tmp_file)
        res = {k: v for k, v in res.items() if model.fail_msg not in v}

        data = load(data_file)
        data_un = data[~data['index'].isin(res)]
        data_un = data_un[~pd.isna(data_un['prediction'])]
        lt = len(data_un)
        prompts = [build_prompt(data_un.iloc[i]) for i in range(lt)]
        indices = [data_un.iloc[i]['index'] for i in range(lt)]

        if len(prompts):
            _ = track_progress_rich(
                model.generate,
                prompts,
                keys=indices,
                save=tmp_file,
                nproc=nproc,
                chunksize=nproc
            )
        score_map = load(tmp_file)
        data['score'] = [score_map[idx] if idx in score_map else -1 for idx in data['index']]
        rejected = [x for x in score_map.values() if FAIL_MSG in x]
        data['score'] = [int(x) if istype(x, int) else -1 for x in data['score']]
        print(
            f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(score_map)} questions, '
            f'failed to obtain the score for another {len(rejected)} questions. '
            f'Those questions will be counted as 0 score in ALL rating, and will not be counted in VALID rating.'
        )

        dump(data, score_file)

    rating = get_dimension_rating(score_file)
    for k, v in rating.items():
        print(k + ': ')
        print(v)
        print('')
    dump(rating, tgt_file)


def main():
    args = parse_args()
    judge_kwargs = dict(model=args.judge, nproc=args.nproc)
    if 'OPENAI_API_KEY_JUDGE' in os.environ and os.environ['OPENAI_API_KEY_JUDGE']:
        judge_kwargs['key'] = os.environ['OPENAI_API_KEY_JUDGE']
    if 'OPENAI_API_BASE_JUDGE' in os.environ and os.environ['OPENAI_API_BASE_JUDGE']:
        judge_kwargs['api_base'] = os.environ['OPENAI_API_BASE_JUDGE']

    _ = MMBenchVideo_eval(args.data, **judge_kwargs)


if __name__ == '__main__':
    main()
