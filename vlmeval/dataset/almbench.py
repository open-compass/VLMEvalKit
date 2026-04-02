"""
VLMEvalKit dataset class for ALM-Bench.
"""

import re
import string

import pandas as pd

from ..smp import load
from .image_base import ImageBaseDataset

LANGUAGES = [
    'Afrikaans', 'Albanian', 'Amharic', 'Armenian', 'Assamese', 'Azerbaijani',
    'Basque', 'Belarusian', 'Bengali', 'Bhojpuri', 'Bosnian', 'Bulgarian',
    'Catalan', 'Cebuano', 'Chinese_Simplified', 'Chinese_Traditional', 'Croatian',
    'Czech', 'Danish', 'Dutch', 'Egyptian_Arabic', 'Emirati_Arabic', 'English',
    'Estonian', 'Filipino', 'Finnish', 'French', 'Galician', 'Georgian',
    'German', 'Greek', 'Gujarati', 'Hausa', 'Hawaiian', 'Hebrew', 'Hindi',
    'Hungarian', 'Icelandic', 'Igbo', 'Indonesian', 'Irish', 'Italian',
    'Japanese', 'Javanese', 'Kannada', 'Kazakh', 'Kinyarwanda', 'Korean',
    'Kurdish', 'Kyrgyz', 'Lao', 'Latin', 'Latvian', 'Lithuanian',
    'Luxembourgish', 'Macedonian', 'Malagasy', 'Malay', 'Malayalam', 'Maltese',
    'Marathi', 'Mongolian', 'Myanmar_Burmese', 'Nepali', 'Norwegian',
    'Odia_Oriya', 'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi',
    'Romanian', 'Russian', 'Sanskrit', 'Saudi_Arabic', 'Scots_Gaelic',
    'Serbian', 'Shona', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian', 'Somali',
    'Spanish', 'Sundanese', 'Swahili', 'Swedish', 'Tajik', 'Tamil', 'Telugu',
    'Thai', 'Turkish', 'Ukrainian', 'Urdu', 'Uyghur', 'Uzbek', 'Vietnamese',
    'Welsh', 'Yiddish', 'Yoruba',
]


def _make_url_dicts():
    names = ['ALMBench'] + [f'ALMBench_{lang}' for lang in LANGUAGES]
    return {name: '' for name in names}, {name: None for name in names}


DATASET_URL, DATASET_MD5 = _make_url_dicts()


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = str(text).lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _question_family(question_type: str) -> str:
    qtype = _normalise(question_type)
    if qtype in ('t/f', 'true/false', 'tf', 'true false question'):
        return 'tf'
    if qtype in ('mcqs', 'mcq', 'multiple choice', 'multiple choice questions'):
        return 'mcq'
    if qtype in ('svqas', 'svqa', 'short questions', 'short'):
        return 'short'
    if qtype in ('lvqas', 'lvqa', 'long question', 'long questions', 'long'):
        return 'long'
    return 'open'


def _extract_tf(text: str):
    """Extract True/False from a model prediction."""
    norm = _normalise(text)
    if re.search(r'\btrue\b|\byes\b|\bcorrect\b', norm):
        return 'true'
    if re.search(r'\bfalse\b|\bno\b|\bincorrect\b', norm):
        return 'false'
    return None


def _extract_mcq_answer(answer: str) -> str:
    text = str(answer).strip()
    for delimiter in (' (', '\n('):
        if delimiter in text:
            return text.split(delimiter, 1)[0].strip()
    return text


def _soft_exact_match(prediction: str, answer: str) -> bool:
    return _normalise(prediction) == _normalise(answer)


def _tf_match(prediction: str, answer: str, english_answer: str = '') -> bool:
    pred_label = _extract_tf(prediction)
    ans_label = _extract_tf(english_answer) if str(english_answer).strip() else None
    if ans_label is None:
        ans_label = _extract_tf(answer)
    if pred_label is None or ans_label is None:
        if english_answer and _soft_exact_match(prediction, english_answer):
            return True
        return _soft_exact_match(prediction, answer)
    return pred_label == ans_label


def _accuracy(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    return round(df['correct'].sum() / len(df) * 100, 2)


def _evaluate_row(row) -> bool:
    qtype = _question_family(str(row.get('question_type', '')))
    prediction = str(row['prediction'])
    answer = str(row['answer'])
    english_answer = str(row.get('english_answer', ''))

    if qtype == 'tf':
        return _tf_match(prediction, answer, english_answer)
    if qtype == 'mcq':
        return _soft_exact_match(prediction, _extract_mcq_answer(answer))
    return _soft_exact_match(prediction, answer)


class ALMBenchDataset(ImageBaseDataset):
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
        family = _question_family(str(line.get('question_type', '')).strip().lower())

        if family == 'tf':
            instruction = 'Answer with True or False only.'
        elif family == 'mcq':
            instruction = 'Answer using only the text of the correct option.'
        elif family == 'short':
            instruction = 'Answer the question using a single word or short phrase.'
        else:
            instruction = 'Answer the question as accurately as possible.'

        prompt = f'{question}\n{instruction}'
        msgs = [dict(type='image', value=p) for p in img_paths]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data['correct'] = data.apply(_evaluate_row, axis=1)

        rows = []

        def add_rows(col_name, split_label):
            if col_name not in data.columns:
                return
            for value in sorted(data[col_name].dropna().unique()):
                sub = data[data[col_name] == value]
                rows.append({
                    'dataset': self.dataset_name,
                    'split_by': split_label,
                    'value': value,
                    'total': len(sub),
                    'correct': int(sub['correct'].sum()),
                    'accuracy (%)': _accuracy(sub),
                })

        add_rows('language', 'language')
        add_rows('category', 'category')
        add_rows('question_type', 'question_type')
        rows.append({
            'dataset': self.dataset_name,
            'split_by': 'overall',
            'value': 'all',
            'total': len(data),
            'correct': int(data['correct'].sum()),
            'accuracy (%)': _accuracy(data),
        })

        result_df = pd.DataFrame(rows)
        result_path = eval_file.replace('.xlsx', '_ALMBench_results.csv')
        if result_path == eval_file:
            result_path = eval_file + '_ALMBench_results.csv'
        result_df.to_csv(result_path, index=False)
        print(f'\nALM-Bench results  ->  {result_path}')
        print(result_df.to_string(index=False))
        return result_df


def _make_lang_class(lang: str):
    name = f'ALMBench_{lang}'
    return type(
        name,
        (ALMBenchDataset,),
        {
            '__doc__': f'ALM-Bench - language: {lang}',
            'DATASET_URL': {name: DATASET_URL.get(name, '')},
            'DATASET_MD5': {name: DATASET_MD5.get(name)},
        },
    )


for _lang in LANGUAGES:
    globals()[f'ALMBench_{_lang}'] = _make_lang_class(_lang)


class ALMBench(ALMBenchDataset):
    DATASET_URL = {'ALMBench': DATASET_URL.get('ALMBench', '')}
    DATASET_MD5 = {'ALMBench': DATASET_MD5.get('ALMBench')}


ALM_LANGUAGES = list(LANGUAGES)
ALM_DATASETS = ['ALMBench'] + [f'ALMBench_{lang}' for lang in LANGUAGES]
