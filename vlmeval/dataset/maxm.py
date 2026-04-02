"""
VLMEvalKit dataset class for MaXM.
"""

import json
import re
import string

import pandas as pd

from ..smp import load
from .image_base import ImageBaseDataset

LANGUAGES = ['en', 'fr', 'hi', 'iw', 'ro', 'th', 'zh']

HF_ROOT = 'https://huggingface.co/datasets/inigopm/vlmevalkit-maxm-tsv/resolve/main'
DATASET_URL = {
    'MaXM': f'{HF_ROOT}/MaXM.tsv',
    'MaXM_en': f'{HF_ROOT}/MaXM_en.tsv',
    'MaXM_fr': f'{HF_ROOT}/MaXM_fr.tsv',
    'MaXM_hi': f'{HF_ROOT}/MaXM_hi.tsv',
    'MaXM_iw': f'{HF_ROOT}/MaXM_iw.tsv',
    'MaXM_ro': f'{HF_ROOT}/MaXM_ro.tsv',
    'MaXM_th': f'{HF_ROOT}/MaXM_th.tsv',
    'MaXM_zh': f'{HF_ROOT}/MaXM_zh.tsv',
}
DATASET_MD5 = {
    'MaXM': 'edc625b8627bd2c2b2054c1c1598c7b6',
    'MaXM_en': 'e9829f8289c9957b142f8daee68634e9',
    'MaXM_fr': 'c940238668b9e6cb94797d3b64624890',
    'MaXM_hi': 'e4c1fc7402ea0fa475c66437c5f8063c',
    'MaXM_iw': '1cf0d5ee2544c300ae620d891cf3f84a',
    'MaXM_ro': '351ab40f15968819df0ff4f4945160f2',
    'MaXM_th': '67850a5d069529875cf2ed776fc2e39e',
    'MaXM_zh': '909f9aba140b072d6ef7db2bdf3a38f4',
}


def _normalise(text: str) -> str:
    text = str(text).lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _parse_answer_list(raw) -> list[str]:
    """Parse list-like answer fields stored as strings or JSON arrays."""
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        if '|' in raw:
            return [part.strip() for part in raw.split('|') if part.strip()]
        matches = re.findall(r"'([^']*)'", raw)
        if matches:
            return matches
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [raw]
    return [str(raw)]


def _vqa_score(prediction: str, answers: list[str]) -> float:
    """Compute VQA-style soft scoring: min(1, matches / 3)."""
    pred_norm = _normalise(prediction)
    matches = sum(pred_norm == _normalise(ans) for ans in answers)
    return min(1.0, matches / 3.0)


def _avg_score(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    return round(df['score'].mean() * 100, 2)


class MaXMDataset(ImageBaseDataset):
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
        prompt = (
            f'{question}\n'
            'Answer the question using a single word or short phrase.'
        )

        msgs = [dict(type='image', value=p) for p in img_paths]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        answer_col = 'processed_answers' if 'processed_answers' in data.columns else 'answers'

        data['score'] = data.apply(
            lambda row: _vqa_score(row['prediction'], _parse_answer_list(row[answer_col])),
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
                    'score_sum': round(sub['score'].sum(), 2),
                    'accuracy (%)': _avg_score(sub),
                })

        rows.append({
            'dataset': self.dataset_name,
            'lang': 'overall',
            'total': len(data),
            'score_sum': round(data['score'].sum(), 2),
            'accuracy (%)': _avg_score(data),
        })

        result_df = pd.DataFrame(rows)
        result_path = eval_file.replace('.xlsx', '_MaXM_results.csv')
        result_df.to_csv(result_path, index=False)
        print(f'\nMaXM results  ->  {result_path}')
        print(result_df.to_string(index=False))
        return result_df


def _make_lang_class(lang: str):
    name = f'MaXM_{lang}'
    return type(
        name,
        (MaXMDataset,),
        {
            '__doc__': f'MaXM benchmark - language: {lang}',
            'DATASET_URL': {name: DATASET_URL.get(name, '')},
            'DATASET_MD5': {name: DATASET_MD5.get(name)},
        },
    )


MaXM_en = _make_lang_class('en')
MaXM_fr = _make_lang_class('fr')
MaXM_hi = _make_lang_class('hi')
MaXM_iw = _make_lang_class('iw')
MaXM_ro = _make_lang_class('ro')
MaXM_th = _make_lang_class('th')
MaXM_zh = _make_lang_class('zh')


class MaXM(MaXMDataset):
    DATASET_URL = {'MaXM': DATASET_URL.get('MaXM', '')}
    DATASET_MD5 = {'MaXM': DATASET_MD5.get('MaXM')}


MAXM_DATASETS = ['MaXM'] + [f'MaXM_{lang}' for lang in LANGUAGES]
