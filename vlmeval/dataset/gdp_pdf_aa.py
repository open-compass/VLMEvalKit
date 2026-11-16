"""Artificial Analysis GDP.pdf document preparation and single-attempt scoring.

Reference: https://artificialanalysis.ai/methodology/intelligence-benchmarking#gdp-pdf
AA does not publish its exact inference/judge prompt templates. The templates
below implement the documented inputs; they are not verbatim AA prompts.
Endpoint image limits must be supplied in the dataset configuration. Repeat
orchestration is deliberately left to the runner.
"""

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from vlmeval.smp import dump, load, md5, read_ok
from .gdp_pdf import FAIL_MSG, GDPPDF, _sample_scores, _score_criterion
from .utils.judge_cache import (get_judge_cache_file, get_judge_detail_file, get_judge_score_file,
                                load_judge_cache, run_cached_tasks)
from .utils.judge_util import build_judge


class _TaskAwareJudge:
    def __init__(self, model, question):
        self.model = model
        self.question = question

    def generate(self, prompt, **kwargs):
        return self.model.generate(f'Task prompt:\n{self.question}\n\n{prompt}', **kwargs)


def _score_aa_criterion(model, response, criterion, question):
    return _score_criterion(_TaskAwareJudge(model, question), response, criterion)


class GDPPDFAA(GDPPDF):
    """AA GDP.pdf, one attempt per task, with complete LiteParse OCR text."""

    DEFAULT_JUDGE_MODEL = 'gpt-5.6-luna'

    def __init__(self, dataset='GDP_pdf_AA', dpi=150, max_images=None):
        if isinstance(dpi, bool) or not isinstance(dpi, int) or not 72 <= dpi <= 150:
            raise ValueError('AA GDP.pdf dpi must be an integer between 72 and 150')
        if max_images is not None and (
            isinstance(max_images, bool) or not isinstance(max_images, int) or max_images <= 0
        ):
            raise ValueError('max_images must be a positive integer or None')
        self.max_images = max_images
        self._liteparse_executable = None
        super().__init__(dataset=dataset, dpi=dpi)

    @classmethod
    def supported_datasets(cls):
        return ['GDP_pdf_AA', 'GDP_pdf_AA_text']

    def _get_liteparse(self):
        if self._liteparse_executable is None:
            command = shutil.which('lit')
            if command is None:
                raise ImportError(
                    'LiteParse is not installed or its lit executable is not on PATH. '
                    'Install @llamaindex/liteparse@2.5.0 with English OCR support.'
                )
            # The 2.5.0 npm release still prints 2.0.0 for --version. Check the
            # package manifest beside the resolved dist/cli.js instead.
            manifest = Path(command).resolve().parent.parent / 'package.json'
            if not manifest.is_file():
                raise ValueError('Expected the lit executable from @llamaindex/liteparse@2.5.0')
            package = json.loads(manifest.read_text(encoding='utf-8'))
            if (package.get('name'), package.get('version')) != ('@llamaindex/liteparse', '2.5.0'):
                raise ValueError('GDP_pdf_AA requires @llamaindex/liteparse version 2.5.0')
            self._liteparse_executable = command
        return self._liteparse_executable

    def _extract_pages(self, filename):
        import fitz

        pdf = self.data_root / filename
        with fitz.open(str(pdf)) as document:
            page_count = len(document)
        if not page_count:
            raise ValueError(f'PDF has no pages: {pdf}')
        cache = self.data_root / 'liteparse_2.5.0_eng' / Path(filename).with_suffix('.json')
        pdf_md5 = md5(str(pdf))
        parsed = load(str(cache)) if cache.is_file() else None
        needs_parse = parsed is None or parsed.get('pdf_md5') != pdf_md5
        if needs_parse:
            result = subprocess.run(
                [self._get_liteparse(), 'parse', str(pdf), '--format', 'json',
                 '--ocr-language', 'eng', '--max-pages', str(page_count),
                 '--dpi', '150', '--quiet'],
                check=True, capture_output=True, text=True, encoding='utf-8',
            )
            parsed = json.loads(result.stdout)
            parsed['pdf_md5'] = pdf_md5
        pages = parsed['pages']
        if (
            [page['page'] for page in pages] != list(range(1, page_count + 1))
            or any(not isinstance(page.get('text'), str) for page in pages)
        ):
            raise ValueError(f'LiteParse must return text for every page in order: {pdf}')
        if needs_parse:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=cache.parent, suffix='.json', delete=False
            ) as stream:
                temporary = Path(stream.name)
            try:
                dump(parsed, stream.name)
                temporary.replace(cache)
            finally:
                temporary.unlink(missing_ok=True)
        return pages

    def load_data(self, dataset):
        data = super().load_data(dataset)
        documents = {}
        for filename in data['pdf_path'].unique():
            pages = self._extract_pages(filename)
            documents[filename] = '\n\n'.join(
                f'--- Page {page["page"]} ---\n{page["text"]}' for page in pages
            )
        data['document_text'] = data['pdf_path'].map(documents)
        return data

    def _image_inputs(self, line):
        if self.dataset_name == 'GDP_pdf_AA_text':
            return [], 0, 1
        originals = super().dump_image(line)
        per_image = 1
        if self.max_images is not None:
            while math.ceil(len(originals) / per_image) > self.max_images and per_image < 4:
                per_image *= 2
        covered = min(len(originals), (self.max_images or len(originals)) * per_image)
        originals = originals[:covered]
        return self._adapt_images(originals, per_image), covered, per_image

    def _adapt_images(self, originals, per_image):
        if per_image == 1:
            return originals
        directory = Path(originals[0]).parent / f'aa_{self.dpi}dpi_{per_image}pp_png'
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for start in range(0, len(originals), per_image):
            path = directory / f'page_{start + 1:05d}.png'
            if not read_ok(str(path)):
                images = []
                for source in originals[start:start + per_image]:
                    with Image.open(source) as image:
                        images.append(image.convert('RGB'))
                width = max(image.width for image in images)
                height = max(image.height for image in images)
                label_height = 24
                output = Image.new(
                    'RGB', (2 * width, math.ceil(len(images) / 2) * (height + label_height)),
                    'white',
                )
                draw = ImageDraw.Draw(output)
                for offset, image in enumerate(images):
                    x, y = offset % 2 * width, offset // 2 * (height + label_height)
                    draw.text((x + 4, y + 4), f'Page {start + offset + 1}', fill='black')
                    output.paste(image, (x, y + label_height))
                with tempfile.NamedTemporaryFile(
                    dir=directory, suffix='.png', delete=False
                ) as stream:
                    temporary = stream.name
                    output.save(stream, format='PNG')
                Path(temporary).replace(path)
            paths.append(str(path))
        return paths

    def dump_image(self, line):
        return self._image_inputs(line)[0]

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        paths, covered, per_image = self._image_inputs(line)
        notes = []
        if per_image > 1:
            notes.append('Page images are composites; each cell is labeled with its page number.')
        if paths:
            notes.append(
                f'Page images cover pages 1 through {covered}. '
                'The extracted text below includes every page.'
            )
        text = line['question'] + '\n\n'
        if notes:
            text += '\n'.join(notes) + '\n\n'
        text += 'Document text:\n' + line['document_text']
        return [dict(type='image', value=path) for path in paths] + [
            dict(type='text', value=text)
        ]

    def evaluate(self, eval_file, **judge_kwargs):
        predictions = load(eval_file)
        predictions['index'] = predictions['index'].astype(str)
        metadata = self.data.copy()
        metadata['index'] = metadata['index'].astype(str)
        if (
            predictions['index'].duplicated().any()
            or not set(predictions['index']) <= set(metadata['index'])
        ):
            raise ValueError('AA GDP.pdf predictions must have unique, known dataset indices')
        # Preserve every task, taking its question and rubrics from canonical data.
        data = metadata.merge(
            predictions[['index', 'prediction']], on='index', how='left', validate='one_to_one'
        )
        missing = (
            data['prediction'].isna() | data['prediction'].astype(str).str.strip().eq('')
        )
        data.loc[missing, 'prediction'] = FAIL_MSG
        judge_name = judge_kwargs.pop('model', self.DEFAULT_JUDGE_MODEL)
        nproc = judge_kwargs.pop('nproc', 4)
        stage = 'aa_rubric_v2'
        cache_file = get_judge_cache_file(eval_file, stage, judge_name)
        cache = load_judge_cache(cache_file)
        tasks, keys, sample_keys = [], [], []
        for _, row in data.iterrows():
            response = str(row['prediction'])
            row_keys = []
            for rubric in json.loads(row['rubrics']):
                fingerprint = json.dumps(
                    [row['index'], response, rubric, row['question']], ensure_ascii=False
                )
                key = hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()
                row_keys.append((key, rubric['id']))
                if FAIL_MSG in response:
                    cache[key] = dict(score='0', passed=False, rationale='Inference failed.',
                                      judge_response=[])
                elif key not in cache or cache[key]['passed'] is None:
                    tasks.append((response, rubric['criterion'], row['question']))
                    keys.append(key)
            sample_keys.append(row_keys)
        if tasks:
            judge_kwargs.setdefault('max_tokens', 65535)
            judge_kwargs.setdefault('reasoning_effort', 'medium')
            model = build_judge(model=judge_name, **judge_kwargs)
            cache.update(run_cached_tasks(
                _score_aa_criterion, [(model, *task) for task in tasks], keys,
                cache_file, nproc=nproc,
            ))
        grades = [
            [dict(rubric_id=rubric_id, **cache[key]) for key, rubric_id in row_keys]
            for row_keys in sample_keys
        ]
        scores = pd.DataFrame([_sample_scores(row) for row in grades])
        for column in scores:
            data[column] = scores[column].values
        data['judge_log'] = [json.dumps(row, ensure_ascii=False) for row in grades]
        self._judge_name = judge_name
        dump(data, get_judge_detail_file(eval_file, stage, judge_name, 'tsv'))
        if data['num_unscored_criteria'].sum():
            raise ValueError(
                'AA GDP.pdf grading is incomplete; retry evaluation to obtain every '
                'criterion verdict before reporting scores.'
            )

        def summarize(frame):
            return dict(
                mean_pass=float(frame['mean_criteria'].mean()),
                num_samples=len(frame), num_fully_scored=int(frame['all_pass'].notna().sum()),
                num_scored_criteria=int(frame['num_scored_criteria'].sum()),
                num_unscored_criteria=int(frame['num_unscored_criteria'].sum()),
            )

        result = dict(all_pass=float(data['all_pass'].mean()), **summarize(data))
        for category, frame in data.groupby('category', dropna=False):
            result.update({f'domain/{category}/{key}': value
                           for key, value in summarize(frame).items()})
        dump(result, get_judge_score_file(eval_file, judge_name, 'json'))
        return result

    def get_pass_k_detail_paths(self, sub_file):
        return [get_judge_detail_file(sub_file, 'aa_rubric_v2', self._judge_name, 'tsv')]
