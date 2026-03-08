import ast

from vlmeval.smp import *
from .image_base import ImageBaseDataset


class SPBench(ImageBaseDataset):
    """
    SPBench.

    Reference:
      SpatialLadder: Progressive Training for Spatial Reasoning in Vision-Language Models
      https://arxiv.org/abs/2510.08531
    """

    TYPE = 'MCQ'

    # Prompt template directly from SPBench codebase:
    # https://github.com/ZJU-REAL/SpatialLadder/blob/main/eval_spld/data_utils/vsi_utils.py
    THINKING_TEMPLATE = (
        "Question: {question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "  # noqa: E501
        "It's encouraged to include self-reflection or verification in the reasoning process. \n"
    )

    PROMPT_TEMPLATES = {
        'default': {
            'pre_prompt': 'Question: {question}\n',
            'mca_post_prompt': "Please answer with the option's letter from the given choices (e.g., A, B, etc.) directly.",  # noqa: E501
            'na_post_prompt': 'Please answer the question using a numerical value (e.g., 42 or 3.1) directly.',
        },
        'thinking': {
            'pre_prompt': THINKING_TEMPLATE,
            'mca_post_prompt': (
                'Please provide your detailed reasoning between the <think> </think> tags, '
                "and then answer the question with the option's letter from the given choices (e.g., A, B, etc.) within the <answer> </answer> tags."  # noqa: E501
            ),
            'na_post_prompt': (
                'Please provide your detailed reasoning between the <think> </think> tags, '
                'and then answer the question with a numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags.'  # noqa: E501
            ),
        },
    }

    DATASET_URL = {
        'SPBench-MV': 'https://opencompass.openxlab.space/utils/Spatial/SPBench-MV.tsv',
        'SPBench-MV_CoT': 'https://opencompass.openxlab.space/utils/Spatial/SPBench-MV.tsv',
        'SPBench-SI': 'https://opencompass.openxlab.space/utils/Spatial/SPBench-SI.tsv',
        'SPBench-SI_CoT': 'https://opencompass.openxlab.space/utils/Spatial/SPBench-SI.tsv',
    }

    DATASET_MD5 = {
        'SPBench-MV': '70b525250806c64d08f59b0e68dc5a95',
        'SPBench-MV_CoT': '70b525250806c64d08f59b0e68dc5a95',
        'SPBench-SI': '200691adc3bdf7227238c7b16e23a529',
        'SPBench-SI_CoT': '200691adc3bdf7227238c7b16e23a529',
    }

    IMG_ZIP_URL = 'https://opencompass.openxlab.space/utils/Spatial/SPBench.zip'
    IMG_ZIP_MD5 = 'f41fe315acec0329d0a73cdd3a8324bf'

    def __init__(self, dataset, skip_noimg=True):
        self._CATEGORY_TASK_ORDER = None
        self.use_cot = self.parse_dataset_name(dataset)
        print(f'Evaluate {dataset} with CoT = {self.use_cot}')
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

    def prepare_tsv(self, url, file_md5=None, img_zip=IMG_ZIP_URL, img_zip_md5=IMG_ZIP_MD5):
        return super().prepare_tsv(url, file_md5, img_zip, img_zip_md5)

    @staticmethod
    def parse_dataset_name(name: str) -> bool:
        if not isinstance(name, str):
            return False

        lower = name.lower()
        return lower.endswith('_cot')

    def get_task_type(self, question_type):
        mcq_items = [
            'object_rel_direction',
            'object_rel_distance'
        ]

        na_items = [
            'object_counting',
            'object_abs_distance',
            'object_size_estimation'
        ]

        if question_type in mcq_items:
            return 'MCQ'
        elif question_type in na_items:
            return 'NA'
        else:
            raise ValueError(f'Unknown question type: {question_type}')

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = line['candidates']

        if options is None:
            options = []
        elif isinstance(options, str):
            try:
                options = ast.literal_eval(options)
            except Exception:
                options = [options] if options.strip() else []
        elif not isinstance(options, (list, tuple)):
            options = [options]

        if options:
            question += '\nOptions:\n' + '\n'.join(options)

        question_type = line['question_type']
        task_type = self.get_task_type(question_type)

        # Prompt format in SPBench codebase
        prompt_type = 'thinking' if self.use_cot else 'default'
        prompt_template = self.PROMPT_TEMPLATES.get(prompt_type)

        prompt_text = prompt_template['pre_prompt'].format(question=question)
        if task_type == 'MCQ':
            prompt_text += '\n' + prompt_template['mca_post_prompt']
        elif task_type == 'NA':
            prompt_text += '\n' + prompt_template['na_post_prompt']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt_text))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .VideoBench.vsibench_easi import VsiBench_EASI

        # Reuse VsiBench.evaluate because SPBench and VsiBench share the same metric computation logic.
        if not hasattr(type(self), '_aggregate'):
            type(self)._aggregate = VsiBench_EASI._aggregate
        return VsiBench_EASI.evaluate(self, eval_file, **judge_kwargs)
