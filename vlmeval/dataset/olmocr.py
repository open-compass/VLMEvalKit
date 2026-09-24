import hashlib
import os
import os.path as osp
from collections import defaultdict
from pathlib import Path

import pandas as pd

from vlmeval.smp import LMUDataRoot, d2df, dump, get_intermediate_file_path, load
from .image_base import ImageBaseDataset


class OLMOCRBench(ImageBaseDataset):
    """Adapter for the official olmOCR-Bench unit-test OCR benchmark."""

    MODALITY = 'IMAGE'
    TYPE = 'QA'

    DATASET_URL = {
        'OLMOCRBench': '',
        'OLMOCRBench_TEST': '',
    }
    DATASET_MD5 = {}

    DATASET_REPO = 'allenai/olmOCR-bench'
    SYSTEM_PROMPT = (
        'Please provide a natural, plain text representation of the document, '
        'formatted in Markdown. Skip any headers and footers. '
        'For ALL mathematical expressions, use LaTeX notation with \\( and \\) '
        'for inline equations and \\[ and \\] for display equations. '
        'Convert any tables into Markdown format.'
    )

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        self.data_path = data_path
        if osp.exists(data_path) and not os.environ.get('OLMOCR_REBUILD'):
            return load(data_path)

        data_dir = self._resolve_data_dir()
        tests, _ = self._load_tests(data_dir)
        if dataset.endswith('_TEST'):
            tests = tests[: min(100, len(tests))]

        pdf_page_tests = defaultdict(list)
        for test in tests:
            pdf_page_tests[(test.pdf, int(test.page))].append(test.id)

        rows = []
        for (pdf_name, page), test_ids in sorted(pdf_page_tests.items()):
            pdf_path = data_dir / 'pdfs' / pdf_name
            image_path = self._render_page(pdf_path, page, dataset)
            rows.append({
                'index': f'{pdf_name}::page{page}',
                'image_path': image_path,
                'question': self.SYSTEM_PROMPT,
                'pdf': pdf_name,
                'page': page,
                'test_ids': ','.join(test_ids),
            })

        if not rows:
            raise RuntimeError(f'No olmOCR-Bench tests found under {data_dir}')

        data = pd.DataFrame(rows)
        dump(data, data_path)
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return [
            dict(type='image', value=self.dump_image(line)[0]),
            dict(type='text', value=line['question']),
        ]

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        required = {'prediction', 'pdf', 'page', 'test_ids'}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f'OLMOCRBench evaluation requires columns: {sorted(missing)}')

        data_dir = self._resolve_data_dir()
        tests, test_to_jsonl = self._load_tests(data_dir)
        test_map = {test.id: test for test in tests}

        details = []
        type_scores = defaultdict(list)
        jsonl_scores = defaultdict(list)

        for _, row in data.iterrows():
            prediction = '' if pd.isna(row['prediction']) else str(row['prediction'])
            for test_id in str(row['test_ids']).split(','):
                test = test_map.get(test_id)
                if test is None:
                    details.append({
                        'pdf': row['pdf'],
                        'page': row['page'],
                        'test_id': test_id,
                        'type': 'unknown',
                        'score': 0.0,
                        'passed': False,
                        'reason': 'test id not found',
                    })
                    continue
                try:
                    passed, reason = test.run(prediction)
                except Exception as e:
                    passed, reason = False, str(e)
                score = 1.0 if passed else 0.0
                details.append({
                    'pdf': row['pdf'],
                    'page': row['page'],
                    'test_id': test.id,
                    'type': test.type,
                    'score': score,
                    'passed': passed,
                    'reason': reason,
                })
                type_scores[test.type].append(score)
                jsonl_scores[test_to_jsonl.get(test.id, 'unknown')].append(score)

        detail_df = pd.DataFrame(details)
        detail_file = get_intermediate_file_path(eval_file, '_olmocr_eval')
        dump(detail_df, detail_file)

        summary = {}
        category_scores = []
        for jsonl_file, scores in sorted(jsonl_scores.items()):
            if scores:
                value = sum(scores) / len(scores) * 100
                summary[jsonl_file] = value
                category_scores.append(value)
        summary['Overall'] = sum(category_scores) / len(category_scores) if category_scores else 0.0
        for test_type, scores in sorted(type_scores.items()):
            if scores:
                summary[f'{test_type}_avg'] = sum(scores) / len(scores) * 100
        summary['Tests'] = len(details)
        summary['Pages'] = len(data)

        ret = d2df(summary)
        score_file = get_intermediate_file_path(eval_file, '_score')
        dump(ret, score_file)
        return ret

    def _resolve_data_dir(self):
        env_dir = os.environ.get('OLMOCR_DATA_DIR')
        if env_dir:
            return Path(env_dir).expanduser().resolve()

        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                'OLMOCRBench requires huggingface_hub to download allenai/olmOCR-bench. '
                'Install huggingface_hub or set OLMOCR_DATA_DIR to a local benchmark directory.'
            ) from e

        target = Path(LMUDataRoot()) / 'OLMOCRBench'
        snapshot_download(
            repo_id=self.DATASET_REPO,
            repo_type='dataset',
            local_dir=str(target),
        )
        return target

    @staticmethod
    def _load_tests(data_dir):
        try:
            from olmocr.bench.tests import BaselineTest, load_tests
        except ImportError as e:
            raise ImportError(
                'OLMOCRBench evaluation requires the official olmocr package with bench dependencies. '
                'Install it from the olmOCR repository, e.g. `pip install -e .[bench]`.'
            ) from e

        jsonl_files = sorted(Path(data_dir).glob('*.jsonl'))
        if not jsonl_files:
            raise RuntimeError(f'No olmOCR-Bench jsonl files found under {data_dir}')

        tests = []
        test_to_jsonl = {}
        for jsonl_file in jsonl_files:
            loaded = load_tests(str(jsonl_file))
            for test in loaded:
                test_to_jsonl[test.id] = jsonl_file.name
            tests.extend(loaded)

        pdf_dir = Path(data_dir) / 'pdfs'
        for pdf_path in sorted(pdf_dir.rglob('*.pdf')):
            pdf_name = str(pdf_path.relative_to(pdf_dir))
            if not any(test.type == 'baseline' for test in tests if test.pdf == pdf_name):
                baseline = BaselineTest(
                    id=f'{pdf_name}_baseline',
                    pdf=pdf_name,
                    page=1,
                    type='baseline',
                )
                tests.append(baseline)
                test_to_jsonl[baseline.id] = 'baseline'

        tests.sort(key=lambda test: (test.pdf, int(test.page), test.id))
        return tests, test_to_jsonl

    @staticmethod
    def _render_page(pdf_path, page, dataset):
        if not pdf_path.exists():
            raise FileNotFoundError(f'Cannot find olmOCR-Bench PDF: {pdf_path}')

        image_root = Path(LMUDataRoot()) / 'images' / dataset
        image_root.mkdir(parents=True, exist_ok=True)
        key = f'{pdf_path}:{page}'
        path_hash = hashlib.md5(key.encode('utf-8')).hexdigest()[:10]
        out_path = image_root / f'{pdf_path.stem}_pg{page}_{path_hash}.png'
        if out_path.exists():
            return str(out_path)

        try:
            import fitz
        except ImportError as e:
            raise ImportError('OLMOCRBench PDF rendering requires pymupdf. Install with `pip install pymupdf`.') from e

        with fitz.open(str(pdf_path)) as doc:
            page_idx = int(page) - 1
            if page_idx < 0 or page_idx >= len(doc):
                raise ValueError(f'Invalid page {page} for {pdf_path}')
            pix = doc.load_page(page_idx).get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pix.save(str(out_path))
        return str(out_path)
