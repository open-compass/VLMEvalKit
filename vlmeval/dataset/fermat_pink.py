"""FERMAT-PINK: over-correction-aware evaluation of handwritten math OCR.

Seong et al., "When VLMs 'Fix' Students: Identifying and Penalizing Over-Correction in the
Evaluation of Multi-line Handwritten Math OCR", EMNLP 2026. https://github.com/9115jin/PINK

The VLM transcribes a handwritten student solution that contains an error. An LLM judge
grades the transcription and the oracle transcription (what the student actually wrote)
with a 5-component rubric; components on which the transcription scores higher than the
oracle are over-corrections and are penalized. Scoring uses the ``pink-eval`` package.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, get_logger, load
from .image_base import ImageBaseDataset

logger = get_logger(__name__)

# OCR prompt of the paper (Fig. 18): system instruction followed by the user request.
OCR_SYSTEM_PROMPT = (
    'You are a math assistant specializing in extracting mathematical question and answer '
    'content from handwritten images of math problems by middle or high school students. '
    'Your task is to analyze the given Image, which contains the handwritten math '
    'Question-Answer pair, and convert it into LaTeX format. Follow the specific instructions '
    'provided carefully and output your response strictly in the requested format. Ensure '
    'that your evaluation is precise, accurate, and based solely on the information visible '
    'in the image.')
OCR_USER_PROMPT = (
    'The image provided contains a handwritten math problem with both a Question and an '
    'Answer at a middle or high school level. Your task is to explicitly perform OCR on the '
    'handwritten text and extract the content in LaTeX format. Return only the extracted '
    'content, formatted in LaTeX, exactly as it appears in the image, in the **Question:** '
    'and **Answer:** fields.\n\n'
    'Ensure that no extra information is added that is not in the image.\n\n'
    'Please return the LaTeX output as follows:\n'
    '**Question:**<Extracted Question text in LaTeX>\n'
    '**Answer:**<Extracted Answer text in LaTeX>')

# FERMAT_PINK_MINI: a fixed random subset for quick, cheap checks. Its scores are not
# comparable to full-set FERMAT_PINK scores reported in the paper.
MINI_SIZE = 100
MINI_SEED = 0
CHAT_SUFFIX = '/chat/completions'


def _pink():
    try:
        import pink  # noqa: F401
    except ImportError as e:
        raise ImportError("FERMAT_PINK needs the PINK scorer: pip install 'pink-eval>=1.2'") from e
    from pink import data, judge, ocr, rubric, score
    return data, judge, ocr, score, rubric


def _image_suffix(image_bytes, path):
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return '.png'
    if image_bytes[:3] == b'\xff\xd8\xff':
        return '.jpg'
    return Path(path).suffix if path else '.png'


def _is_true(value):
    return str(value).lower() == 'true'


class FERMAT_PINK(ImageBaseDataset):
    """FERMAT handwritten solutions (CC-BY-4.0, gated on the Hub), scored with PINK."""

    TYPE = 'VQA'
    force_use_dataset_prompt = True
    DEFAULT_JUDGE_MODEL = 'gpt-5'

    @classmethod
    def supported_datasets(cls):
        return ['FERMAT_PINK', 'FERMAT_PINK_MINI']

    def load_data(self, dataset):
        from datasets import Image, load_dataset

        pink_data = _pink()[0]
        # Keep the original image bytes: handwriting details matter for faithful transcription.
        source = load_dataset('ai4bharat/FERMAT', split='train')
        source = source.cast_column('image', Image(decode=False))
        if dataset == 'FERMAT_PINK_MINI':
            source = source.shuffle(seed=MINI_SEED).select(range(MINI_SIZE))
        image_root = Path(self.img_root)
        image_root.mkdir(parents=True, exist_ok=True)
        records = []
        for item in source:
            image = item['image']
            image_bytes = image['bytes']
            if image_bytes is None:
                image_bytes = Path(image['path']).read_bytes()
            suffix = _image_suffix(image_bytes, image.get('path'))
            image_path = image_root / f"{item['new_custom_id']}{suffix}"
            if not image_path.exists():
                image_path.write_bytes(image_bytes)
            records.append({
                'index': len(records),
                'sample_id': item['new_custom_id'],
                'image_path': str(image_path),
                'question': OCR_USER_PROMPT,
                'problem': str(item['orig_q']).replace('\r\n', '\n'),
                'answer': pink_data.oracle_transcription(item['pert_a']),
                'category': item['domain_code'],
            })
        return pd.DataFrame(records)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return [
            dict(type='image', value=line['image_path']),
            dict(type='text', value=f'{OCR_SYSTEM_PROMPT}\n\n{OCR_USER_PROMPT}'),
        ]

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if str(key).split('|')[-1] == 'PINK':
                    return {'PINK': value}
        return {}

    @classmethod
    def judge_config(cls, judge_kwargs):
        """Map VLMEvalKit judge arguments to a PINK judge and its oracle grades."""
        _, judge, _, score, _ = _pink()
        name = judge_kwargs.get('model') or cls.DEFAULT_JUDGE_MODEL
        base_url = judge_kwargs.get('api_base')
        if base_url and base_url.endswith(CHAT_SUFFIX):
            base_url = base_url[:-len(CHAT_SUFFIX)]
        overrides = {
            k: v
            for k, v in dict(base_url=base_url, api_key=judge_kwargs.get('key')).items()
            if v is not None
        }
        oracle_spec = None
        if name in judge.JUDGE_PRESETS:
            cfg = judge.judge_from_preset(name, **overrides)
            if f'fermat:{name}' in score.PACKAGED_ORACLES:
                oracle_spec = f'fermat:{name}'
        else:
            cfg = judge.JudgeConfig(model=name, **overrides)
        return name, cfg, os.environ.get('PINK_ORACLE_GRADES') or oracle_spec

    def evaluate(self, eval_file, **judge_kwargs):
        _, judge, ocr, score, rubric = _pink()
        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        name, cfg, oracle_spec = self.judge_config(judge_kwargs)
        client = judge.make_client(cfg)
        workers = int(judge_kwargs.get('nproc') or 16)
        tag = name.replace('/', '_')

        def grade_all(texts, cache_name):
            # Judge results are cached next to the predictions, so a rerun resumes.
            cache_file = get_intermediate_file_path(eval_file, f'_{tag}_{cache_name}', 'csv')
            done = {}
            if os.path.exists(cache_file):
                for row in load(cache_file).to_dict('records'):
                    if _is_true(row.get('success')):
                        done[str(row['sample_id'])] = row
            todo = [item for item in texts if item[0] not in done]

            def grade_one(item):
                sample_id, problem, text = item
                if not str(text or '').strip():  # an empty transcription is a failed one
                    zeros = {col: 0 for col in rubric.SCORE_COLUMNS}
                    return {'sample_id': sample_id, 'success': True, **zeros}
                return judge.grade(client, cfg, problem, text, sample_id)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for row in pool.map(grade_one, todo):
                    done[str(row['sample_id'])] = row
            rows = pd.DataFrame(list(done.values()))
            dump(rows, cache_file)
            graded = rows[rows['success'].map(_is_true)].set_index('sample_id')
            return graded[rubric.SCORE_COLUMNS].astype(float), len(rows) - len(graded)

        predictions = []
        for _, row in data.iterrows():
            prediction = '' if pd.isna(row['prediction']) else row['prediction']
            predictions.append(
                (str(row['sample_id']), row['problem'], ocr.clean_transcription(prediction)))
        model_grades, n_failed = grade_all(predictions, 'pink_model_grades')

        if oracle_spec:
            oracle_grades, source = score.load_grades(oracle_spec), oracle_spec
        else:
            oracle_texts = [(str(r['sample_id']), r['problem'], r['answer'])
                            for _, r in data.iterrows()]
            oracle_grades, _ = grade_all(oracle_texts, 'pink_oracle_grades')
            source = f'graded with {cfg.model}'

        result = score.compute_pink(model_grades, oracle_grades)
        table = pd.DataFrame([{
            'PINK': result.pink,
            'over_correction_rate': result.over_correction_rate,
            'raw_score_ratio': result.raw_ratio,
            'n': result.n,
            'judge_failures': n_failed,
        }])
        details = table.assign(judge=cfg.model, oracle=source)
        dump(details, get_intermediate_file_path(eval_file, '_score', 'csv'))
        logger.info(f'FERMAT_PINK: {details.to_dict("records")[0]}')
        return table
