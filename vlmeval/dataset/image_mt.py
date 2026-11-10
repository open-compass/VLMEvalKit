import os.path as osp
from collections import defaultdict

import numpy as np
import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, load, toliststr
from vlmeval.utils import track_progress_rich
from ..interactive_mt import StaticTurnPolicy
from .image_base import ImageBaseDataset
from .interactive_mt import InteractiveMTMixin
from .utils.judge_util import build_judge


class MMDUDataset(InteractiveMTMixin, ImageBaseDataset):

    TYPE = 'VQA'
    DEFAULT_JUDGE_MODEL = 'gpt-4o'
    DEFAULT_POLICY_MODEL = None
    DATASET_URL = {'MMDU': 'https://opencompass.openxlab.space/utils/VLMEval/MMDU.tsv'}
    DATASET_MD5 = {'MMDU': '848b635a88a078f49aebcc6e39792061'}
    DIMS = [
        'Creativity', 'Richness', 'Visual Perception', 'Logical Coherence',
        'Answer Accuracy', 'Image Relationship Understanding', 'Overall Score'
    ]

    # ---- InteractiveMTMixin -------------------------------------------------

    def _build_messages(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            media_paths = toliststr(line['image_path'])
        else:
            media_paths = self.dump_image(line)

        questions = toliststr(line['question'])
        if 'answer' in line:
            answers = toliststr(line['answer'])
        else:
            answers = [''] * len(questions)
        assert len(questions) == len(answers)
        tag = '<ImageHere>'
        messages, media_used = [], 0
        for question, answer in zip(questions, answers):
            if tag in question:
                content = []
                tag_number = question.count(tag)
                media = media_paths[media_used: media_used + tag_number]
                media_used += tag_number
                parts = question.split(tag)
                for i in range(tag_number):
                    if parts[i] != '':
                        content.append(dict(type='text', value=parts[i]))
                    content.append(dict(type='image', value=media[i]))
                if parts[-1] != '':
                    content.append(dict(type='text', value=parts[-1]))
            else:
                content = [dict(type='text', value=question)]
            messages.append(dict(role='user', content=content))
            assert tag not in answer, 'We currently do not support images in the answer. '
            messages.append(dict(role='assistant', content=[dict(type='text', value=answer)]))
        return messages

    def build_prompt(self, case):
        """Return the complete context for the first model call only."""
        messages = self._build_messages(case)
        if not messages:
            raise ValueError('An MT sample must contain at least one question.')
        return messages[:1]

    def build_policy(self, case, messages, **policy_kwargs):
        """Static MT: replay the remaining user turns in order."""
        all_messages = self._build_messages(case)
        return StaticTurnPolicy(messages, all_messages[2::2])

    # ---- evaluation ---------------------------------------------------------

    def calculat_metric(self, ans):
        all = defaultdict(lambda: 0)
        tot = defaultdict(lambda: 0)
        valid = defaultdict(lambda: 0)
        for k in ans:
            res = ans[k]['res']
            assert isinstance(res, pd.DataFrame)
            lt = len(res)
            for i in range(lt):
                line = res.iloc[i]
                for k in self.DIMS:
                    tot[k] += 1
                    if k in line and line[k] is not None:
                        try:
                            score = int(line[k])
                            score = np.clip(score, 0, 10)
                            all[k] += score
                            valid[k] += 1
                        except Exception as e:
                            print(f'Failed to parse the score: {str(e)}')
        sp1 = {'set': 'all'}
        sp1.update({k: all[k] / tot[k] * 10 for k in self.DIMS})
        sp2 = {'set': 'valid'}
        sp2.update({k: all[k] / valid[k] * 10 for k in self.DIMS})

        return pd.DataFrame([sp1, sp2])

    def evaluate(self, eval_file, **judge_kwargs):
        model = judge_kwargs['model']

        tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
        score_file = get_intermediate_file_path(eval_file, f'_{model}_score', 'csv')
        nproc = judge_kwargs.pop('nproc', 4)

        data = load(eval_file)
        model = judge_kwargs.pop('model', self.DEFAULT_JUDGE_MODEL)
        judge_model = build_judge(model=model, **judge_kwargs)

        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        tups = [(judge_model, line) for line in lines]
        indices = [line['index'] for line in lines]

        ans = {}
        if osp.exists(tmp_file):
            ans = load(tmp_file)

        tups = [x for x, i in zip(tups, indices) if i not in ans]
        indices = [i for i in indices if i not in ans]

        from .utils.mmdu import mmdu_score

        if len(indices):
            new_results = track_progress_rich(
                mmdu_score,
                tups,
                nproc=nproc,
                chunksize=nproc,
                keys=indices,
                save=tmp_file,)
            ans = load(tmp_file)
            for k, v in zip(indices, new_results):
                assert k in ans

        metric = self.calculat_metric(ans)
        dump(metric, score_file)
        return metric
