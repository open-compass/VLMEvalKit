"""
VLMEvalKit dataset class for xGQA.
"""

import re
import string

import pandas as pd

from ..smp import load
from .image_base import ImageBaseDataset

LANGUAGES = ['bn', 'de', 'en', 'id', 'ko', 'pt', 'ru', 'zh']

DATASET_URL = {k: '' for k in ['xGQA'] + [f'xGQA_{lang_code}' for lang_code in LANGUAGES]}
DATASET_MD5 = {k: None for k in ['xGQA'] + [f'xGQA_{lang_code}' for lang_code in LANGUAGES]}


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = str(text).lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _exact_match(prediction: str, answer: str) -> bool:
    return _normalise(prediction) == _normalise(answer)


def _accuracy(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    return round(df['correct'].sum() / len(df) * 100, 2)


class xGQADataset(ImageBaseDataset):
    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    DATASET_URL = DATASET_URL
    DATASET_MD5 = DATASET_MD5

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        img_paths = self.dump_image(line)
        if not isinstance(img_paths, list):
            img_paths = [img_paths]

        question = str(line['question'])
        prompt = (
            f'{question}\n'
            'Answer the question using a single word or short phrase.'
        )

        msgs = [dict(type='image', value=p) for p in img_paths]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data['correct'] = data.apply(
            lambda row: _exact_match(row['prediction'], row['answer']),
            axis=1,
        )

        rows = []
        if 'category' in data.columns:
            for lang in sorted(data['category'].unique()):
                sub = data[data['category'] == lang]
                rows.append({
                    'dataset': self.dataset_name,
                    'lang': lang,
                    'total': len(sub),
                    'correct': int(sub['correct'].sum()),
                    'accuracy (%)': _accuracy(sub),
                })

        rows.append({
            'dataset': self.dataset_name,
            'lang': 'overall',
            'total': len(data),
            'correct': int(data['correct'].sum()),
            'accuracy (%)': _accuracy(data),
        })

        result_df = pd.DataFrame(rows)
        result_path = eval_file.replace('.xlsx', '_xGQA_results.csv')
        result_df.to_csv(result_path, index=False)
        print(f'\nxGQA results  ->  {result_path}')
        print(result_df.to_string(index=False))
        return result_df


def _make_lang_class(lang: str):
    name = f'xGQA_{lang}'
    return type(
        name,
        (xGQADataset,),
        {
            '__doc__': f'xGQA benchmark - language: {lang}',
            'DATASET_URL': {name: DATASET_URL.get(name, '')},
            'DATASET_MD5': {name: DATASET_MD5.get(name)},
        },
    )


xGQA_bn = _make_lang_class('bn')
xGQA_de = _make_lang_class('de')
xGQA_en = _make_lang_class('en')
xGQA_id = _make_lang_class('id')
xGQA_ko = _make_lang_class('ko')
xGQA_pt = _make_lang_class('pt')
xGQA_ru = _make_lang_class('ru')
xGQA_zh = _make_lang_class('zh')


class xGQA(xGQADataset):
    DATASET_URL = {'xGQA': DATASET_URL.get('xGQA', '')}
    DATASET_MD5 = {'xGQA': DATASET_MD5.get('xGQA')}


XGQA_DATASETS = ['xGQA'] + [f'xGQA_{lang}' for lang in LANGUAGES]
