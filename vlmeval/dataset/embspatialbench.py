import os.path as osp
import ast
import json
import string

from .image_mcq import ImageMCQDataset
from ..smp.file import LMUDataRoot, load
from ..smp.misc import toliststr


class EmbSpatialBench(ImageMCQDataset):
    TYPE = 'MCQ'

    LMUData_root = LMUDataRoot()
    DATASET_URL = {}

    # TODO: change this to hugging face url after upload
    DATASET_URL['EmbSpatialBench'] = osp.join(LMUData_root, "EmbSpatialBench.tsv")

    DATASET_MD5 = {key: None for key in DATASET_URL}

    def _task_category(self):
        return [
            'mp3d',
            'ai2thor',
            'scannet'
        ]

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = line['options']

        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                options = ast.literal_eval(options)

        UpperLetters = list(string.ascii_uppercase)
        option_text = "\n".join(
            f"{UpperLetters[i]}: {options[i]}"
            for i in range(len(options))
        )

        # EmbSpatial has not yet released official inference code,
        # We use the SiteBench prompt format here.
        prompt = ''
        prompt += "Question: " + question + "\n"
        prompt += "Options:\n" + option_text + "\n"
        post_prompt = "Give me the answer letter directly. The best answer is:"
        prompt += post_prompt

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
            group_col='data_source',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'EmbSpatialBench')
        )
