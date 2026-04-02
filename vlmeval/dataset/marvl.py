"""
VLMEvalKit dataset class for MaRVL.
"""

import re
import string

import pandas as pd

from ..smp import load
from .image_base import ImageBaseDataset

LANGUAGES = ['id', 'sw', 'ta', 'tr', 'zh']

DATASET_URL = {
    'MaRVL': '',
    'MaRVL_id': '',
    'MaRVL_sw': '',
    'MaRVL_ta': '',
    'MaRVL_tr': '',
    'MaRVL_zh': '',
}

DATASET_MD5 = {
    'MaRVL': None,
    'MaRVL_id': None,
    'MaRVL_sw': None,
    'MaRVL_ta': None,
    'MaRVL_tr': None,
    'MaRVL_zh': None,
}

_TRUE_TOKENS = {'true', 'yes', 'correct', 'right', '1'}
_FALSE_TOKENS = {'false', 'no', 'wrong', 'incorrect', '0'}


def _extract_answer(prediction: str) -> str:
    """Parse a free-form model prediction into 'True' or 'False'."""
    clean = str(prediction).strip().strip(string.punctuation).lower()
    for tok in _TRUE_TOKENS:
        if re.search(rf'\b{tok}\b', clean):
            return 'True'
    for tok in _FALSE_TOKENS:
        if re.search(rf'\b{tok}\b', clean):
            return 'False'
    first = clean.split()[0] if clean.split() else clean
    return first.capitalize()


def _normalise_binary_label(value) -> str:
    """Normalise saved booleans / strings to canonical labels."""
    text = str(value).strip()
    if text in {'True', 'False'}:
        return text
    return _extract_answer(text)


def _accuracy(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    return round(df['correct'].sum() / len(df) * 100, 2)


class MaRVLDataset(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = DATASET_URL
    DATASET_MD5 = DATASET_MD5

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        img_paths = self.dump_image(line)
        if not isinstance(img_paths, list):
            img_paths = [img_paths]

        question = str(line['question'])
        hint = str(line.get('hint', '')) if 'hint' in line.index else ''

        prompt = (
            'You are shown two images placed side by side.\n'
            f'Hypothesis: {question}\n'
        )
        if hint and hint.lower() not in ('', 'nan', 'none'):
            prompt += f'(English translation: {hint})\n'

        prompt += (
            '\nBased on the two images, is the hypothesis TRUE or FALSE?\n'
            'Answer with a single word: True or False.'
        )

        msgs = [dict(type='image', value=p) for p in img_paths]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data['prediction_normalized'] = data['prediction'].apply(_normalise_binary_label)
        data['answer_normalized'] = data['answer'].apply(_normalise_binary_label)
        data['correct'] = data['prediction_normalized'] == data['answer_normalized']

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
        result_path = eval_file.replace('.xlsx', '_MaRVL_results.csv')
        result_df.to_csv(result_path, index=False)
        print(f'\nMaRVL results  ->  {result_path}')
        print(result_df.to_string(index=False))
        return result_df


def _make_lang_class(lang: str):
    name = f'MaRVL_{lang}'
    return type(
        name,
        (MaRVLDataset,),
        {
            '__doc__': f'MaRVL benchmark - language: {lang}',
            'DATASET_URL': {name: DATASET_URL.get(name, '')},
            'DATASET_MD5': {name: DATASET_MD5.get(name)},
        },
    )


MaRVL_id = _make_lang_class('id')
MaRVL_sw = _make_lang_class('sw')
MaRVL_ta = _make_lang_class('ta')
MaRVL_tr = _make_lang_class('tr')
MaRVL_zh = _make_lang_class('zh')


class MaRVL(MaRVLDataset):
    DATASET_URL = {'MaRVL': DATASET_URL.get('MaRVL', '')}
    DATASET_MD5 = {'MaRVL': DATASET_MD5.get('MaRVL')}


MARVL_DATASETS = ['MaRVL'] + [f'MaRVL_{lang}' for lang in LANGUAGES]
