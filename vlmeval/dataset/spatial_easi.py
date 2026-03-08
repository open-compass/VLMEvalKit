from .image_mcq import ImageMCQDataset
from .utils import report_acc
from vlmeval.smp import *
import string
import pandas as pd


class ERIQ(ImageMCQDataset):
    """
    ERIQ.

    Reference:
      Unified Embodied VLM Reasoning with Robotic Action via Autoregressive Discretized Pre-training
      https://arxiv.org/abs/2512.24125
    """

    TYPE = 'MCQ'
    DATASET_URL = {
        'ERIQ': 'https://opencompass.openxlab.space/utils/Spatial/ERIQ.tsv',  # noqa: E501
    }
    DATASET_MD5 = {
        'ERIQ': 'ddd53eaf6c40dddb741597cb73f32574',
    }
    IMG_ZIP = 'https://opencompass.openxlab.space/utils/Spatial/ERIQ.zip'
    IMG_ZIP_MD5 = '0642afa5ef9cc295f393c42868de8ae2'
    DEFAULT_JUDGE = 'gpt-4o-mini'

    def prepare_tsv(self, url, file_md5=None, img_zip=IMG_ZIP, img_zip_md5=IMG_ZIP_MD5):
        return super().prepare_tsv(url, file_md5, img_zip, img_zip_md5)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        # ERIQ QA prompt format use in paper
        prompt = line['question_raw']

        msgs = self.build_msgs(tgt_path, prompt)
        return msgs

    @staticmethod
    def build_msgs(tgt_path, prompt):
        """
        Interlaced text and pictures
        """
        images = tgt_path if isinstance(tgt_path, list) else [tgt_path]

        parts = prompt.split('<image>')
        segs = []

        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                segs.append(dict(type='text', value=part))
            if (i != len(parts) - 1) and (i < len(images)):
                segs.append(dict(type='image', value=images[i]))

        return [s for s in segs if s['value']]


class MindCube(ImageMCQDataset):
    """
    MindCube.

    Reference:
      Spatial Mental Modeling from Limited Views
      https://arxiv.org/abs/2506.21458
    """

    TYPE = 'MCQ'

    DATASET_URL = {
        'MindCube_tiny_raw': 'https://opencompass.openxlab.space/utils/Spatial/MindCube_tiny_raw.tsv',
        'MindCube_raw': 'https://opencompass.openxlab.space/utils/Spatial/MindCube_raw.tsv'
    }
    DATASET_MD5 = {
        'MindCube_tiny_raw': 'a11b9b40ce11e70807063831b5476c96',
        'MindCube_raw': '94cbcd61a835147ec51c40c7fec61964'
    }
    DEFAULT_JUDGE = 'gpt-4o-mini'
    IMG_ZIP = 'https://opencompass.openxlab.space/utils/Spatial/MindCube.zip'
    IMG_ZIP_MD5 = '8b6c43491eadb07f7fcc21addbdcc048'

    def prepare_tsv(self, url, file_md5=None, img_zip=IMG_ZIP, img_zip_md5=IMG_ZIP_MD5):
        return super().prepare_tsv(url, file_md5, img_zip=img_zip, img_zip_md5=img_zip_md5)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        # # Raw QA prompt format use in paper
        prompt = line['input_prompt']

        msgs = self.build_msgs(tgt_path, prompt)
        return msgs

    @staticmethod
    def build_msgs(tgt_path, prompt):
        """
        Interlaced text and pictures
        """
        images = tgt_path if isinstance(tgt_path, list) else [tgt_path]

        parts = prompt.split('<image>')
        segs = []

        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                segs.append(dict(type='text', value=part))
            if (i != len(parts) - 1) and (i < len(images)):
                segs.append(dict(type='image', value=images[i]))

        return [s for s in segs if s['value']]


class EmbSpatialBench(ImageMCQDataset):
    """
    EmbSpatial-Bench.

    Reference:
      EmbSpatial-Bench: Benchmarking Spatial Understanding for Embodied Tasks with Large Vision-Language Models
      https://arxiv.org/abs/2406.05756
    """

    TYPE = 'MCQ'
    DATASET_URL = {
        'EmbSpatialBench': 'https://opencompass.openxlab.space/utils/Spatial/EmbSpatialBench.tsv'  # noqa: E501
    }
    DATASET_MD5 = {
        'EmbSpatialBench': 'f7f2620abae9dfa4b9ce245cbd66aad9'
    }
    DEFAULT_JUDGE = 'gpt-4o-mini'

    def report_acc(self, data):
        return report_acc(data, inds=['data_source'])


class OmniSpatialBench(ImageMCQDataset):
    """
    OmniSpatial.

    Reference:
      OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models
      https://arxiv.org/abs/2506.03135
    """

    TYPE = 'MCQ'

    OMNI_TSV_URL = 'https://opencompass.openxlab.space/utils/Spatial/OmniSpatialBench.tsv'
    OMNI_TSV_MD5 = 'b16fc540ee83fd3f61fc32f0749acc79'
    IMG_ZIP = 'https://opencompass.openxlab.space/utils/Spatial/OmniSpatial.zip'
    IMG_ZIP_MD5 = '3851cd3cf7592d6ae51364df80ec8f5b'

    VARIANTS = [
        'OmniSpatialBench',
        'OmniSpatialBench_default',
        'OmniSpatialBench_zeroshot_cot',
        'OmniSpatialBench_manual_cot',
    ]

    # Prompt template adapted from the official OmniSpatial codebase:
    # https://github.com/qizekun/OmniSpatial/tree/main
    RE_FORMAT = """
End your answer with a separate line formatted exactly as:

Answer: X
where X ∈ {A, B, C, D}.
"""

    DEFAULT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive
1. **Image** - a single RGB frame depicting a scene.
2. **Question** - a natural-language query about spatial relationships between objects in the image.
3. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Based on the image and question, provide your answer.
Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply “insufficient information.”
"""

    ZERO_SHOT_COT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive
1. **Image** - a single RGB frame depicting a scene.
2. **Question** - a natural-language query about spatial relationships between objects in the image.
3. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Think step by step and provide the answer.
Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply “insufficient information.”
"""

    MANUAL_COT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive
1. **Image** - a single RGB frame depicting a scene.
2. **Question** - a natural-language query about spatial relationships between objects in the image.
3. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Guidelines
----------
Please follow these steps to analyze the image and answer the question:
1. First, carefully observe the image and identify all relevant objects and their spatial relationships.
2. Next, break down the question into key components that need to be addressed.
3. Think through the spatial reasoning step-by-step to arrive at your answer. It may be necessary to transfer perspective to better understand the scene.   # noqa: E501
4. Finally, select the most appropriate option (A, B, C, or D) based on your analysis.

Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply “insufficient information.”
"""  # noqa: E501

    DATASET_URL = {}
    DATASET_MD5 = {}

    for name in VARIANTS:
        DATASET_URL[name] = OMNI_TSV_URL
        DATASET_MD5[name] = OMNI_TSV_MD5

    SYS_PROMPTS = {
        'default': DEFAULT_SYSTEM_PROMPT,
        'zeroshot_cot': ZERO_SHOT_COT_SYSTEM_PROMPT,
        'manual_cot': MANUAL_COT_SYSTEM_PROMPT,
    }

    def __init__(self, dataset, skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)
        self.prompt_mode = self.parse_dataset_name(dataset)

    def prepare_tsv(self, url, file_md5=None, img_zip=IMG_ZIP, img_zip_md5=IMG_ZIP_MD5):
        return super().prepare_tsv(url, file_md5, img_zip, img_zip_md5)

    def parse_dataset_name(self, name: str) -> str:
        if not isinstance(name, str):
            return ''

        lower = name.lower()

        for key in self.SYS_PROMPTS.keys():
            if lower.endswith(f'_{key}'.lower()):
                return key

        return ''

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
        option_text = ''
        for key, item in options.items():
            option_text += f'\n{key}. {item}'

        # prompt format from OmniSpatial codebase
        if self.prompt_mode in self.SYS_PROMPTS.keys():
            system_prompt = self.SYS_PROMPTS[self.prompt_mode]
            prompt = system_prompt + '\n' + self.RE_FORMAT + '\n\n' + question + option_text

        # EASI also provides direct QA format
        else:
            prompt = question + option_text + '\nAnswer directly with the option letter from the given choices. '

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def report_acc(self, data):
        return report_acc(data, inds=['task_type', 'sub_task_type'])


class ViewSpatialBench(ImageMCQDataset):
    """
    ViewSpatial-Bench.

    Reference:
      ViewSpatial-Bench: Evaluating Multi-perspective Spatial Localization in Vision-Language Models
      https://arxiv.org/abs/2505.21500
    """

    TYPE = 'MCQ'

    DATASET_URL = {
        'ViewSpatialBench': 'https://opencompass.openxlab.space/utils/Spatial/ViewSpatialBench.tsv'
    }
    DATASET_MD5 = {
        'ViewSpatialBench': 'b64becd339fefd14d4e4336459ef2be9'
    }
    DEFAULT_JUDGE = 'gpt-4o-mini'
    IMG_ZIP = 'https://opencompass.openxlab.space/utils/Spatial/ViewSpatial.zip'
    IMG_ZIP_MD5 = 'bb23be2ea83a23aa6ca7472c509a2711'

    def prepare_tsv(self, url, file_md5=None, img_zip=IMG_ZIP, img_zip_md5=IMG_ZIP_MD5):
        return super().prepare_tsv(url, file_md5, img_zip, img_zip_md5)

    def build_prompt(self, line):
        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        choices = line['candidates']

        # prompt format from viewspatial bench paper
        question_text = f"Question: {question}\n"
        choices_text = f"Choices: {choices}\n"
        post_prompt = "Answer: "

        prompt = question_text + choices_text + post_prompt

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def report_acc(self, data):
        return report_acc(data, inds=['question_type'])
