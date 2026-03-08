from .image_base import ImageBaseDataset
from ..smp import *
from .utils import build_judge, DEBUG_MESSAGE

import re
import json
from collections import defaultdict


JUDGE_TMPL = """You are an AI assistant tasked with extracting an answer tuple of points from a model response.
You **MUST** completely ignore internal thinking process wrapped in special tokens such as <think>...</think>.
Your output must be strictly formatted as a list of coordinate tuples: [(x1, y1), (x2, y2), ...].

Model response:
{}
"""


def text2pts(model, text: str, width=640, height=480, is_absolute=False) -> np.ndarray:
    block_pat = re.compile(
        r"\[(?:\s*\(\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*\)\s*"
        r"(?:,\s*\(\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*\)\s*)*)\]"
        r"\s*(?![\s\S]*\[\s*\()"
    )
    m = block_pat.search(text)
    points = []
    if m:
        # exact_matching
        ans = m.group(0)
        judge_output = ""
    elif model is not None:
        # model judge
        prompt = JUDGE_TMPL.format(text)
        retry = 3
        for i in range(retry):
            judge_output = model.generate(prompt, temperature=0.2 * i)
            m = block_pat.search(judge_output)
            if m:
                ans = m.group(0)
                break

    matches = re.findall(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)", ans)
    for x, y in matches:
        x, y = float(x), float(y)
        # Looks like the coordinates are normalized to [0, 1000]
        if x > 1 or y > 1:
            x, y = x / 1000, y / 1000
        if not is_absolute:
            x = int(x * width)
            y = int(y * height)
        points.append((x, y))

    return {
        "points": np.array(points),
        "judge_output": judge_output,
    }


def refspatial_prompt(object_name, prompt, suffix):
    return f"{prompt} {suffix}"


def refspatial_compute_accuracy(model, line):
    mask = np.array(decode_base64_to_image(line['mask'])) / 255.
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0).astype(np.uint8)

    try:
        res = text2pts(model, line['prediction'], mask.shape[1], mask.shape[0])
        points = res["points"]
    except Exception as e:
        print(f"Failed to parse prediction {line['prediction']}: {e}")
        return {"acc": 0.0, "answer_points": json.dumps([]), "judge_output": ""}

    acc = 0.0
    if len(points) > 0:
        in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) & \
            (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
        acc = np.concatenate([
            mask[points[in_range, 1], points[in_range, 0]],
            np.zeros(points.shape[0] - in_range.sum())
        ]).mean()

    return {
        "acc": acc,
        "answer_points": json.dumps(res["points"].tolist()),
        "judge_output": res["judge_output"],
    }


def refspatial_stat(eval_file):
    data = load(eval_file)

    ret = defaultdict(lambda: defaultdict(list))
    for i, row in data.iterrows():
        category = row['category']
        step = row['step']
        ret[category][step].append(row['acc'])

    stat = {}
    overall_acc_list = []
    for category, x in ret.items():
        stat[category] = {}
        category_acc_list = []
        for step, accs in x.items():
            stat[category][f"step{step}"] = np.mean(accs)
            category_acc_list += accs
        stat[category]["overall"] = np.mean(category_acc_list)
        overall_acc_list += category_acc_list
    stat['overall'] = np.mean(overall_acc_list)
    return stat


class RefSpatial(ImageBaseDataset):

    TYPE = 'VQA'
    DEFAULT_JUDGE = 'gpt-4o-mini'

    DATASET_URL = {
        'RefSpatial_Bench': 'https://opencompass.openxlab.space/utils/VLMEval/RefSpatial_Bench.tsv',
        'RefSpatial_Expand_Bench': 'https://opencompass.openxlab.space/utils/VLMEval/RefSpatial_Expand_Bench.tsv',
    }

    DATASET_MD5 = {
        'RefSpatial_Bench': 'b8094c161dd7e3073dd00ca6662c6c9a',
        'RefSpatial_Expand_Bench': '90ffa654509020f00cfd215db1486dca',
    }

    DEFAULT_JUDGE = 'gpt-4o-mini'
    JUDGE_FORMAT = '{model_name}_{dataset_name}_{judge_name}.tsv'
    RATING_FORMAT = '{model_name}_{dataset_name}_{judge_name}_score.json'

    def build_prompt(self, line):
        images = self.dump_image(line)
        suffix = line['suffix'].split('The coordinates should be between 0 and 1')[0].strip()
        question = refspatial_prompt(line['object'], line['prompt'], suffix)

        prompt = []
        for im in images:
            prompt.append(dict(type='image', value=im))
        prompt.append(dict(type='text', value=question))
        return prompt

    @classmethod
    def report_score(cls, model_name, dataset_name, root, verbose=False, **kwargs):
        judge_name = kwargs.pop('judge_name', cls.DEFAULT_JUDGE)
        rating_file = cls.RATING_FORMAT.format(
            model_name=model_name, dataset_name=dataset_name, judge_name=judge_name)
        rating_file = osp.join(root, rating_file)
        rating = load(rating_file)
        rating['overall'] = rating['overall'] * 100
        for k in ['location', 'placement', 'unseen']:
            for kk in rating[k]:
                rating[k][kk] = rating[k][kk] * 100
        res = {'overall': rating['overall']}
        if verbose:
            res['rating'] = rating
        return res

    def evaluate(self, eval_file, **judge_kwargs):
        model = judge_kwargs['model']
        storage = get_intermediate_file_path(eval_file, f"_{model}")
        tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        judge_keys = [
            'acc', 'answer_points', 'judge_output',
        ]

        if not osp.exists(storage):
            self.data['index'] = self.data['index'].astype(str)

            data = load(eval_file)
            data['index'] = data['index'].astype(str)

            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(max_tokens=2048, temperature=0.0, **judge_kwargs)
                assert model.working(), 'RefSpatial evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                _ = track_progress_rich(
                    refspatial_compute_accuracy,
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

        score = refspatial_stat(storage)
        score_pth = get_intermediate_file_path(storage, '_score', 'json')
        dump(score, score_pth)
        return score
