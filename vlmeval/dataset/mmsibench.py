import os.path as osp
import string
import pandas as pd

from .image_mcq import ImageMCQDataset
from ..smp.file import LMUDataRoot, load
from ..smp.misc import toliststr


class MMSIBench(ImageMCQDataset):
    TYPE = 'MCQ'

    # VLMEvalKit officially provides an MMSI *circular* TSV.
    # In this repo we only run the *non-circular* evaluation.
    # To avoid modifying upstream VLMEvalKit, we do NOT integrate the circular set here.
    # (Use the official pipeline if you need the circular split.)
    MMSI_TASKS = [
        'wo_circular',
    ]

    LMUData_root = LMUDataRoot()

    # TODO: change this into hugging face path
    DATASET_URL = {}

    for task in MMSI_TASKS:
        name = f"MMSIBench_{task}"
        path = osp.join(LMUData_root, name + ".tsv")
        DATASET_URL[name] = path

    DATASET_MD5 = {key: None for key in DATASET_URL}

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
        from .utils.spatial_rel_bench.cal_scores import compute_mcq_score, eval_mcq_core

        return eval_mcq_core(
            load_fn=load,
            eval_file=eval_file,
            score_fn=compute_mcq_score,
            group_col='category',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'MMSIBench')
        )
