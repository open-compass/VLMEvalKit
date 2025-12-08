import os.path as osp
import string
import pandas as pd

from .image_mcq import ImageMCQDataset
from ..smp.file import load
from ..smp.misc import toliststr


class MMSIBench(ImageMCQDataset):
    TYPE = 'MCQ'

    # VLMEvalKit officially provides an MMSI *circular* TSV.
    # In this repo we only run the *non-circular* evaluation, which aligns with the
    # evaluation protocol described in the MMSI paper.
    # To avoid modifying upstream VLMEvalKit, we do NOT integrate the circular set here.
    # (Use the official pipeline if you need the circular split.)
    DATASET_URL = {
        'MMSIBench_wo_circular': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MMSIBench_wo_circular.tsv'  # noqa: E501
    }

    DATASET_MD5 = {
        'MMSIBench_wo_circular': '548c5f33f1a12948d5355d5f600749e4'
    }

    def _task_category(self):
        return [
            "Pos-Cam-Cam",
            "Pos-Obj-Obj",
            "Pos-Reg-Reg",
            "Pos-Cam-Obj",
            "Pos-Obj-Reg",
            "Pos-Cam-Reg",
            "Attr-Meas",
            "Attr-Appr",
            "Motion-Cam",
            "Motion-Obj",
            "MSR"
        ]

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }

        # Prompt format aligned with mmsi code base
        options_prompt = 'Options: '
        for key, item in options.items():
            options_prompt += f'{key}: {item}, '
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None

        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'{question}\n'
        if len(options):
            prompt += options_prompt

        # MMSI Direct
        post_prompt = (
            "Answer with the option's letter from the given choices directly. "
            "Enclose the option's letter within ``."
        )

        prompt = f'{prompt}\n{post_prompt}'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import eval_mcq_score, build_mcq_score_fn

        # Select MCQ scoring function (rule-based or LLM-based) according to judge_kwargs['model'].
        score_fn = build_mcq_score_fn(**judge_kwargs)

        return eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col='category',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'MMSIBench')
        )
