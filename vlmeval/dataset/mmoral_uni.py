import os.path as osp
import warnings
from collections import defaultdict

import pandas as pd

from ..smp import dump, get_intermediate_file_path, load, toliststr
from ..utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge


class MMOral_Uni(ImageBaseDataset):
    TYPE = 'VQA'
    MODALITY = 'IMAGE'

    DATASET_URL = {
        'MMOral_Uni':
        'https://huggingface.co/datasets/OralGPT/MMOral-Omni-Bench/resolve/main/MMOral-Omni-Bench.tsv'  # noqa: E501
    }
    DATASET_MD5 = {
        'MMOral_Uni': '139e90f132f02e2a87d60eff1c24254a',
    }

    def __init__(self, dataset='MMOral_Uni', skip_noimg=False, categories=None):
        if dataset != 'MMOral_Uni':
            warnings.warn(
                'To evaluate on MMOral_Uni, we suggest using `MMOral_Uni` as the dataset name.'
            )
        self._categories_filter = categories if categories else None
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

    def post_build(self, dataset):
        if self._categories_filter:
            keywords = [str(kw) for kw in self._categories_filter]
            mask = self.data['category'].apply(lambda c: any(kw in str(c) for kw in keywords))
            self.data = self.data[mask].reset_index(drop=True)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']
        if 'image' in line and str(line['image']) == 'nan':
            return [dict(type='text', value=question)]

        tgt_path = toliststr(line['image_path']) if self.meta_only else self.dump_image(line)

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    @staticmethod
    def _build_judge_prompt(line):
        question = line['question']
        gt = str(line['answer'])
        prediction = str(line['prediction'])
        prompt = ("You are an expert judge for an oral and dental multimodal imaging "
                  "benchmark. Strictly evaluate how correct the prediction is compared "
                  "with the ground truth.\n\n"
                  "Only output one numeric score: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, "
                  "0.6, 0.7, 0.8, 0.9, or 1.0.\n\n"
                  "Evaluation rules:\n"
                  "- Judge clinical semantic equivalence, not wording.\n"
                  "- Treat wrong disease, tooth, side, stage, grade, or severity as "
                  "major errors.\n"
                  "- Accept clinically equivalent synonyms and equivalent tooth "
                  "numbering systems.\n"
                  "- Minor omissions of coordinates or tooth numbers may receive "
                  "partial credit when the clinical meaning remains correct.\n"
                  "- Penalize extra statements that contradict the ground truth or "
                  "introduce clinically false information.\n"
                  "- Refusals, generic non-answers, and fundamentally contradictory "
                  "answers should receive 0.0.\n\n"
                  "Scoring guide:\n"
                  "- 1.0: Fully correct, with no clinically material omissions or "
                  "extra wrong claims.\n"
                  "- 0.7-0.9: Mostly correct, with only minor omissions or "
                  "clinically insignificant inaccuracies.\n"
                  "- 0.3-0.6: Partially correct, but missing a core aspect or "
                  "including a major error.\n"
                  "- 0.1-0.2: Largely incorrect, with minimal overlap.\n"
                  "- 0.0: Totally incorrect.\n\n"
                  "Question | Ground truth | Prediction | Correctness\n"
                  "--- | --- | --- | ---")
        return prompt + '\n' + ' | '.join(
            [question, gt.replace('<AND>', ' <AND> ').replace('<OR>', ' <OR> '), prediction, ''])

    @staticmethod
    def _auxeval(model, line):

        def float_cvt(s):
            try:
                return float(s)
            except ValueError:
                return None

        prompt = MMOral_Uni._build_judge_prompt(line)
        log = ''
        for i in range(5):
            output = model.generate(prompt, temperature=i * 0.5)
            score = float_cvt(output)
            if score is None:
                log += f'Try {i}: output is {output}, failed to parse.\n'
            elif score < 0 or score > 1:
                log += f'Try {i}: output is {output}, invalid score: {score}.\n'
            else:
                log += 'Succeed'
                return dict(log=log, score=score)
        log += 'All 5 retries failed.\n'
        return dict(log=log, score=0.0)

    @staticmethod
    def _acc(result_file):
        data = load(result_file)
        tot = defaultdict(lambda: 0)
        score = defaultdict(lambda: 0)
        cate2_list = []
        coarse_categories = [
            'TP',
            'Endodontics',
            'Implant Dentistry',
            'Periodontics',
            'II_Loc',
            'II_Dx-I',
            'Orthodontics',
            'Cancer',
            'Gingivitis',
            'Defective Dentition',
            'Normality',
            'Tooth Discoloration',
            'Ulcer',
            'Caries',
            'Calculus',
            'II_Dx-R',
            'Fenestration and Dehiscence',
            'Malocclusion Issues Assessment',
            'PA',
            'Impacted Tooth',
            'Pulpitis',
            'Periodontitis',
            'Apical Periodontitis',
            'Mixed Dentition',
            'Bone Loss',
            'Root Canal Treatment',
            'Crown',
            'Restoration',
            'CE',
            'PI',
            'Leukoplakia with Dysplasia',
            'Leukoplakia without Dysplasia',
            'Oral Squamous Cell Carcinoma',
            'Oral Submucous Fibrosis',
            'IV',
        ]

        for i in range(len(data)):
            item = data.iloc[i]
            cate = str(item['category'])
            cate2 = cate.replace(',', '_')
            if cate2 not in cate2_list:
                cate2_list.append(cate2)
            grade = float(item['score'])

            for capa in coarse_categories:
                if capa in cate:
                    tot[capa] += 1
                    score[capa] += grade
            tot['Overall'] += 1
            tot[cate2] += 1
            score['Overall'] += grade
            score[cate2] += grade

        res = defaultdict(list)
        res2 = defaultdict(list)
        for k in coarse_categories + ['Overall']:
            res['Category'].append(k)
            res['tot'].append(tot[k])
            res['acc'].append(score[k] / tot[k] * 100 if tot[k] else 0)
        for v in cate2_list + ['Overall']:
            res2['Category'].append(v)
            res2['tot'].append(tot[v])
            res2['acc'].append(score[v] / tot[v] * 100 if tot[v] else 0)
        return pd.DataFrame(res), pd.DataFrame(res2)

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        judge_model_name = judge_kwargs.pop('model', 'gpt-5.4-mini')
        storage = get_intermediate_file_path(eval_file, f'_{judge_model_name}')
        tmp_file = get_intermediate_file_path(eval_file, f'_{judge_model_name}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(model=judge_model_name, max_tokens=16384, **judge_kwargs)
            assert model.working(), (
                'MMOral_Uni evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            )

            lines = [data.iloc[i] for i in range(len(data))]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    cls._auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['score'] == v['score']

            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score, score_fine = cls._acc(storage)
        score_pth = get_intermediate_file_path(storage, '_score', 'csv')
        score_fine_pth = get_intermediate_file_path(storage, '_score_fine', 'csv')
        dump(score, score_pth)
        dump(score_fine, score_fine_pth)
        return score
