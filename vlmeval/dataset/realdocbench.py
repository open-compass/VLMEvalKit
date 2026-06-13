import hashlib
import json
import os
import os.path as osp
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from vlmeval.smp import LMUDataRoot, d2df, dump, get_intermediate_file_path, load
from .image_base import ImageBaseDataset


class RealDocBench(ImageBaseDataset):
    """RealDocBench adapter with separate Document QA and Layout tracks."""

    MODALITY = 'IMAGE'
    TYPE = 'QA'

    DATASET_URL = {
        'RealDocBench_QA': '',
        'RealDocBench_QA_TEST': '',
        'RealDocBench_Layout': '',
        'RealDocBench_Layout_TEST': '',
    }
    DATASET_MD5 = {}

    QA_REPO = 'Extend-AI/RealDoc-Bench'
    LAYOUT_REPO = 'Extend-AI/RealDoc-Bench-Layout'
    QA_TEST_LIMIT = 100
    LAYOUT_TEST_LIMIT = 100

    QA_PROMPT = (
        'Convert this document into faithful Markdown. Preserve the natural reading order, '
        'all key values, tables, labels, dates, totals, and handwritten or stamped text. '
        'Do not answer questions and do not add explanations. Return only Markdown.'
    )
    LAYOUT_PROMPT = """Detect document layout blocks in this page image.

Return only valid JSON with this schema:
{
  "chunks": [
    {
      "blocks": [
        {
          "id": "block_0",
          "type": "text|heading|section_heading|header|footer|page_number|figure|table|key_value",
          "content": "recognized text, if any",
          "bounding_box": {"left": 0, "top": 0, "right": 100, "bottom": 100},
          "metadata": {"page": {"number": 1, "width": image_width, "height": image_height}, "reading_order": 0}
        }
      ]
    }
  ]
}

Use absolute pixel coordinates in the input image coordinate system. Do not include prose or code fences."""

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        self.data_path = data_path
        if osp.exists(data_path) and not os.environ.get('REALDOC_REBUILD'):
            return load(data_path)

        if self._is_layout(dataset):
            data = self._load_layout_data(dataset)
        else:
            data = self._load_qa_data(dataset)
        dump(data, data_path)
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        paths = self.dump_image(line)
        msgs = [dict(type='image', value=p) for p in paths]
        msgs.append(dict(type='text', value=line['question']))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        if self._is_layout(self.dataset_name):
            return self._evaluate_layout(eval_file)
        return self._evaluate_qa(eval_file)

    @staticmethod
    def _is_layout(dataset):
        return 'Layout' in dataset

    def _load_qa_data(self, dataset):
        data_dir = self._resolve_qa_dir(dataset)
        items = self._load_bank(data_dir)
        if dataset.endswith('_TEST'):
            items = items[: min(self.QA_TEST_LIMIT, len(items))]

        by_doc = defaultdict(list)
        for item in items:
            by_doc[item['source_file']].append(item['question_id'])

        rows = []
        for source_file, qids in sorted(by_doc.items()):
            pdf_path = self._qa_pdf_path(data_dir, source_file)
            image_paths = self._render_pdf(pdf_path, dataset)
            rows.append({
                'index': source_file,
                'image_path': image_paths,
                'question': self.QA_PROMPT,
                'source_file': source_file,
                'question_ids': ','.join(qids),
                'track': 'qa',
            })
        if not rows:
            raise RuntimeError(f'No RealDocBench QA items found under {data_dir}')
        return pd.DataFrame(rows)

    def _load_layout_data(self, dataset):
        samples = self._load_layout_samples(dataset)
        rows = []
        for sample in samples:
            rows.append({
                'index': sample.page_id,
                'image_path': str(sample.image_path),
                'question': self.LAYOUT_PROMPT,
                'page_id': sample.page_id,
                'domain': sample.domain,
                'track': 'layout',
            })
        if not rows:
            raise RuntimeError(f'No RealDocBench layout samples found for {dataset}')
        return pd.DataFrame(rows)

    def _evaluate_qa(self, eval_file):
        data = load(eval_file)
        required = {'prediction', 'source_file', 'question_ids'}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f'RealDocBench_QA evaluation requires columns: {sorted(missing)}')

        data_dir = self._resolve_qa_dir(self.dataset_name)
        items = self._load_bank(data_dir)
        item_map = {item['question_id']: item for item in items}

        details = []
        for _, row in data.iterrows():
            markdown = '' if pd.isna(row['prediction']) else str(row['prediction'])
            for qid in str(row['question_ids']).split(','):
                item = item_map.get(qid)
                if item is None:
                    details.append(self._qa_detail(row['source_file'], qid, False, {}, 'question id not found'))
                    continue
                try:
                    answer = self._gemini_extract(item['question'], item['template'], markdown)
                    field_matches, match = self._score_typed(answer, item['gold_dict'], item['str_keys'])
                    details.append(self._qa_detail(row['source_file'], qid, match, field_matches, ''))
                except Exception as e:
                    details.append(self._qa_detail(row['source_file'], qid, False, {}, str(e)))

        detail_df = pd.DataFrame(details)
        detail_file = get_intermediate_file_path(eval_file, '_realdoc_qa_eval')
        dump(detail_df, detail_file)

        ok = detail_df[detail_df['error'] == ''] if len(detail_df) else detail_df
        field_ok = int(ok['field_ok'].sum()) if len(ok) else 0
        field_total = int(ok['field_total'].sum()) if len(ok) else 0
        q_ok = int(ok['match'].sum()) if len(ok) else 0
        q_total = len(ok)
        ret = d2df({
            'Per-field Accuracy': 100 * field_ok / field_total if field_total else 0.0,
            'Per-question Accuracy': 100 * q_ok / q_total if q_total else 0.0,
            'Questions': len(detail_df),
            'Errors': int((detail_df['error'] != '').sum()) if len(detail_df) else 0,
        })
        score_file = get_intermediate_file_path(eval_file, '_score')
        dump(ret, score_file)
        return ret

    @staticmethod
    def _qa_detail(source_file, qid, match, field_matches, error):
        return {
            'source_file': source_file,
            'question_id': qid,
            'match': bool(match),
            'field_ok': sum(1 for value in field_matches.values() if value),
            'field_total': len(field_matches),
            'field_matches': json.dumps(field_matches, ensure_ascii=False),
            'error': error,
        }

    def _evaluate_layout(self, eval_file):
        data = load(eval_file)
        required = {'prediction', 'page_id'}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f'RealDocBench_Layout evaluation requires columns: {sorted(missing)}')

        sample_map = {sample.page_id: sample for sample in self._load_layout_samples(self.dataset_name)}
        cells = []
        details = []
        for _, row in data.iterrows():
            page_id = row['page_id']
            sample = sample_map.get(page_id)
            if sample is None:
                details.append({'page_id': page_id, 'status': 'error', 'error': 'sample not found'})
                continue
            try:
                pred_doc = self._prediction_to_layout_document(row['prediction'], source=page_id)
                strict, per_cls, structural, adjusted = self._evaluate_layout_cell(sample, pred_doc)
                cells.append((strict, per_cls, adjusted))
                details.append({
                    'page_id': page_id,
                    'domain': getattr(sample, 'domain', ''),
                    'status': 'ok',
                    'strict_f1': strict.f1,
                    'adjusted_f1': adjusted.f1 if adjusted is not None else None,
                    'precision': strict.precision,
                    'recall': strict.recall,
                    'pred_count': len(pred_doc.blocks),
                    'gt_count': len(sample.ground_truth.blocks),
                    'error': '',
                })
            except Exception as e:
                details.append({'page_id': page_id, 'status': 'error', 'error': str(e)})

        detail_df = pd.DataFrame(details)
        detail_file = get_intermediate_file_path(eval_file, '_realdoc_layout_eval')
        dump(detail_df, detail_file)

        if cells:
            strict_total, macro_f1, adjusted_total = self._aggregate_layout_cells(cells)
            summary = {
                'Strict F1': strict_total.f1,
                'Adjusted F1': adjusted_total.f1 if adjusted_total is not None else 0.0,
                'Macro F1': macro_f1,
                'Precision': strict_total.precision,
                'Recall': strict_total.recall,
                'Pages': len(cells),
                'Errors': int((detail_df['status'] == 'error').sum()) if len(detail_df) else 0,
            }
        else:
            summary = {'Strict F1': 0.0, 'Adjusted F1': 0.0, 'Macro F1': 0.0,
                       'Precision': 0.0, 'Recall': 0.0, 'Pages': 0, 'Errors': len(detail_df)}
        ret = d2df(summary)
        score_file = get_intermediate_file_path(eval_file, '_score')
        dump(ret, score_file)
        return ret

    @staticmethod
    def _resolve_qa_dir(dataset):
        env_dir = os.environ.get('REALDOC_QA_DATA_DIR') or os.environ.get('REALDOC_DATA_DIR')
        if env_dir:
            return Path(env_dir).expanduser().resolve()

        try:
            from realdoc_bench.evaluate.download import download_dataset
            from realdoc_bench.evaluate.runs import RunLayout
        except ImportError as e:
            raise ImportError(
                'RealDocBench_QA requires the official realdoc_bench package. '
                'Install realdoc-bench or set REALDOC_QA_DATA_DIR to a local dataset directory.'
            ) from e

        target = Path(LMUDataRoot()) / dataset
        limit = RealDocBench.QA_TEST_LIMIT if dataset.endswith('_TEST') else None
        download_dataset(RunLayout.at(target), repo_id=RealDocBench.QA_REPO, limit=limit)
        return target

    @staticmethod
    def _load_bank(data_dir):
        try:
            from realdoc_bench.evaluate.score import load_bank
        except ImportError as e:
            raise ImportError('RealDocBench_QA evaluation requires realdoc_bench.evaluate.score.') from e
        bank_path = Path(data_dir) / 'qa_bank.json'
        if not bank_path.exists():
            raise FileNotFoundError(f'No RealDocBench QA bank found at {bank_path}')
        return load_bank(bank_path)

    @staticmethod
    def _qa_pdf_path(data_dir, source_file):
        docs_dir = Path(data_dir) / 'docs'
        candidates = [docs_dir / f'{source_file}.pdf', docs_dir / str(source_file)]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f'No RealDocBench QA PDF found for source_file={source_file}')

    @staticmethod
    def _load_layout_samples(dataset):
        try:
            from realdoc_bench.layout.data.loader import load
        except ImportError as e:
            raise ImportError(
                'RealDocBench_Layout requires the official realdoc_bench package. '
                'Install realdoc-bench before evaluating this track.'
            ) from e
        limit = RealDocBench.LAYOUT_TEST_LIMIT if dataset.endswith('_TEST') else None
        return list(load(hf_dataset=RealDocBench.LAYOUT_REPO, limit=limit))

    @staticmethod
    def _render_pdf(pdf_path, dataset):
        image_root = Path(LMUDataRoot()) / 'images' / dataset
        image_root.mkdir(parents=True, exist_ok=True)
        try:
            import fitz
        except ImportError as e:
            raise ImportError(
                'RealDocBench PDF rendering requires pymupdf. Install with `pip install pymupdf`.'
            ) from e

        paths = []
        with fitz.open(str(pdf_path)) as doc:
            for page_idx in range(len(doc)):
                key = f'{pdf_path}:{page_idx + 1}'
                path_hash = hashlib.md5(key.encode('utf-8')).hexdigest()[:10]
                out_path = image_root / f'{pdf_path.stem}_pg{page_idx + 1}_{path_hash}.png'
                if not out_path.exists():
                    pix = doc.load_page(page_idx).get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    pix.save(str(out_path))
                paths.append(str(out_path))
        return paths

    @staticmethod
    def _gemini_extract(question, template, markdown):
        from realdoc_bench.evaluate.score import gemini_extract
        return gemini_extract(question, template, markdown)

    @staticmethod
    def _score_typed(answer, gold_dict, str_keys):
        from realdoc_bench.evaluate.score import score_typed
        return score_typed(answer, gold_dict, str_keys)

    @staticmethod
    def _evaluate_layout_cell(sample, pred_doc):
        from realdoc_bench.layout.runner import _evaluate_cell
        return _evaluate_cell(sample, pred_doc, scorer='adjacency')

    @staticmethod
    def _aggregate_layout_cells(cells):
        from realdoc_bench.layout.metrics.f1 import aggregate, aggregate_per_class

        strict_total = aggregate([strict for strict, _, _ in cells])
        adjusted = [adjusted for _, _, adjusted in cells if adjusted is not None]
        adjusted_total = aggregate(adjusted) if adjusted else None
        per_class_total = aggregate_per_class([per_cls for _, per_cls, _ in cells])
        macro_values = [score.f1 for score in per_class_total.values()]
        macro_f1 = sum(macro_values) / len(macro_values) if macro_values else 0.0
        return strict_total, macro_f1, adjusted_total

    @staticmethod
    def _prediction_to_layout_document(prediction, source=None):
        from realdoc_bench.layout.normalizers.base import LayoutDocument
        from realdoc_bench.layout.normalizers.vlm_json import normalize_vlm_json

        payload = RealDocBench._extract_json(prediction)
        if not isinstance(payload, dict):
            raise ValueError('model output is not valid JSON')
        if 'pages' in payload:
            return LayoutDocument.model_validate(payload)
        if 'blocks' in payload:
            payload = {'chunks': [{'blocks': payload['blocks']}]}
        return normalize_vlm_json(payload, source=source)

    @staticmethod
    def _extract_json(text):
        text = '' if text is None else str(text).strip()
        fence = re.search(r'```(?:json)?\s*(.*?)```', text, flags=re.S | re.I)
        if fence:
            text = fence.group(1).strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError('no JSON object found in model output')
