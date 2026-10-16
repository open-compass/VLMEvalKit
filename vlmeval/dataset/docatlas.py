import json
import os
import os.path as osp
from collections import defaultdict

import pandas as pd

from ..smp import LMUDataRoot, dump, get_intermediate_file_path, load
from ..smp.file import INFER_FAIL_MSG
from ..utils import track_progress_rich
from .image_base import ImageBaseDataset

# Prompt of the official script (https://github.com/ahmedheakl/DocAtlas/tree/main/eval).
PROMPT = r"""You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
- Enclose block formulas with \[ \]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

3. Table Processing:
- Convert tables to HTML format.
- Wrap the entire table with <table> and </table>.

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

5. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments."""  # noqa: E501


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float('nan')


class DocAtlasBench(ImageBaseDataset):
    """DocAtlas-Bench: page parsing in 80+ languages (https://arxiv.org/abs/2605.12623).

    Scoring needs the packages in ``vlmeval/dataset/OmniDocBench/requirements.txt``. Run with
    ``PRED_FORMAT=tsv``: the default xlsx prediction file cuts cells at 32,767 characters.
    """

    TYPE = 'QA'
    DATASET_URL = {'DocAtlasBench': ''}
    DATASET_MD5 = {}
    HF_REPO = 'ahmedheakl/DocAtlas-Bench'

    def load_data(self, dataset):
        tsv_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        if osp.exists(tsv_path):
            return load(tsv_path)

        from datasets import Image, load_dataset
        pages = load_dataset(self.HF_REPO, split='test').cast_column('image', Image(decode=False))
        os.makedirs(self.img_root, exist_ok=True)
        rows = []
        for index, page in enumerate(pages):
            image_path = page['id'] + '.jpg'
            with open(osp.join(self.img_root, image_path), 'wb') as f:
                f.write(page['image']['bytes'])
            rows.append(dict(index=index, image_path=image_path, answer=page['annotation']))
        data = pd.DataFrame(rows)
        dump(data, tsv_path)
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return [dict(type='image', value=self.dump_image(line)[0]), dict(type='text', value=PROMPT)]

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.docatlas import score_page

        data = load(eval_file)
        predictions = data['prediction'].fillna('').astype(str)
        # The API wrappers report an empty answer as a failed request; both are scored as an empty page.
        predictions = predictions.mask(predictions.str.contains(INFER_FAIL_MSG, regex=False), '')
        predictions = dict(zip(data['index'], predictions))
        tasks = [(json.loads(line['answer']), predictions[line['index']]) for _, line in self.data.iterrows()]
        # Scoring is CPU-bound, and the matcher falls back to a simpler one after 30 s of wall-clock time.
        nproc = min(judge_kwargs.get('nproc', 4), os.cpu_count() or 1)
        pages = track_progress_rich(score_page, tasks, nproc=nproc, use_process=True)

        # Text and table scores are averaged per language and then over languages, reading order over pages.
        text, tables = defaultdict(list), defaultdict(list)
        for page in pages:
            text[page['language']].extend(page['text'])
            tables[page['language']].extend(page['tables'])
        text_edit = _mean(_mean(scores) for scores in text.values() if scores)
        table_teds = _mean(100 * _mean(scores) for scores in tables.values() if scores)
        reading_order = _mean(page['reading_order'] for page in pages if page['reading_order'] is not None)
        result = pd.DataFrame([{
            'Overall': ((1 - text_edit) * 100 + table_teds) / 2,
            'Text_Edit': text_edit,
            'Table_TEDS': table_teds,
            'Reading_Order_Edit': reading_order,
        }])
        dump(result, get_intermediate_file_path(eval_file, '_acc'))
        return result
