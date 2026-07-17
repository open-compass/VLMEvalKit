import hashlib
import json
import os
import os.path as osp
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from vlmeval.smp import LMUDataRoot, d2df, dump, get_intermediate_file_path, load
from .image_base import ImageBaseDataset


class ParseBench(ImageBaseDataset):
    """ParseBench document parsing benchmark.

    This adapter evaluates VLMs as page-level document parsers. It reuses the
    official ParseBench test cases and deterministic evaluators when the
    ``parse_bench`` package is installed.
    """

    MODALITY = 'IMAGE'
    TYPE = 'QA'

    DATASET_URL = {
        'ParseBench': '',
        'ParseBench_TEST': '',
    }
    DATASET_MD5 = {}

    SYSTEM_PROMPT = """Convert the document page image into structured parsing output.

Return only valid JSON with this schema:
{
  "markdown": "complete Markdown transcription of the page",
  "pages": [{"page_index": 0, "markdown": "page Markdown"}],
  "layout_pages": [
    {
      "page_number": 1,
      "width": image_width,
      "height": image_height,
      "items": [
        {
          "type": "text|title|table|figure|list|formula|other",
          "md": "item Markdown",
          "bbox": {"x": x, "y": y, "w": width, "h": height}
        }
      ]
    }
  ]
}

If exact layout coordinates are unavailable, still return accurate Markdown and
use an empty layout_pages list. Do not include explanations or code fences."""

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        self.data_path = data_path

        if osp.exists(data_path) and not os.environ.get('PARSEBENCH_REBUILD'):
            return load(data_path)

        data_dir = self._resolve_data_dir(dataset)
        test_cases = self._load_parsebench_cases(data_dir)
        rows = []
        for test_case in test_cases:
            if getattr(test_case, 'group', None) not in {
                'chart', 'layout', 'table', 'text_content', 'text_formatting'
            }:
                continue
            image_path = self._render_first_page(Path(test_case.file_path), dataset)
            rows.append({
                'index': test_case.test_id,
                'image_path': image_path,
                'question': self.SYSTEM_PROMPT,
                'test_id': test_case.test_id,
                'group': test_case.group,
                'source_file_path': str(test_case.file_path),
            })

        if not rows:
            raise RuntimeError(f'No ParseBench test cases found under {data_dir}')

        data = pd.DataFrame(rows)
        dump(data, data_path)
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        image_path = self.dump_image(line)[0]
        return [
            dict(type='image', value=image_path),
            dict(type='text', value=line['question']),
        ]

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        required = {'prediction', 'test_id', 'source_file_path', 'group'}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f'ParseBench evaluation requires columns: {sorted(missing)}')

        data_dir = self._resolve_data_dir(self.dataset_name)
        test_cases = self._load_parsebench_cases(data_dir)
        test_case_map = {tc.test_id: tc for tc in test_cases}

        results = []
        detailed_rows = []
        pipeline_name = osp.splitext(osp.basename(eval_file))[0].replace(f'_{self.dataset_name}', '')

        for _, line in data.iterrows():
            test_id = str(line['test_id'])
            test_case = test_case_map.get(test_id)
            if test_case is None:
                detailed_rows.append({
                    'test_id': test_id,
                    'group': line['group'],
                    'success': False,
                    'error': 'No matching ParseBench test case',
                })
                continue

            try:
                inference_result = self._build_inference_result(line, pipeline_name)
                result = self._evaluate_one(inference_result, test_case)
                results.append(result)
                metric_values = {m.metric_name: m.value for m in result.metrics}
                detailed_rows.append({
                    'test_id': test_id,
                    'group': line['group'],
                    'success': result.success,
                    'error': result.error,
                    **metric_values,
                })
            except Exception as e:
                detailed_rows.append({
                    'test_id': test_id,
                    'group': line['group'],
                    'success': False,
                    'error': str(e),
                })

        detailed = pd.DataFrame(detailed_rows)
        detail_file = get_intermediate_file_path(eval_file, '_parsebench_eval')
        dump(detailed, detail_file)

        summary = self._summarize_results(results, detailed)
        ret = d2df(summary)
        score_file = get_intermediate_file_path(eval_file, '_score')
        dump(ret, score_file)
        return ret

    def _resolve_data_dir(self, dataset):
        env_dir = os.environ.get('PARSEBENCH_DATA_DIR')
        if env_dir:
            return Path(env_dir).expanduser().resolve()

        try:
            from parse_bench.data.download import download_dataset
        except ImportError as e:
            raise ImportError(
                'ParseBench requires the official parse_bench package. '
                'Install it from the ParseBench repository, or set PARSEBENCH_DATA_DIR '
                'to an already downloaded ParseBench dataset directory.'
            ) from e

        root = Path(LMUDataRoot()) / 'ParseBench'
        return download_dataset(root / ('test' if dataset.endswith('_TEST') else 'full'),
                                test=dataset.endswith('_TEST'))

    @staticmethod
    def _load_parsebench_cases(data_dir):
        try:
            from parse_bench.test_cases import load_test_cases
        except ImportError as e:
            raise ImportError(
                'ParseBench evaluation requires parse_bench. '
                'Install the ParseBench package before building or evaluating this dataset.'
            ) from e
        return load_test_cases(root_dir=Path(data_dir), require_test_json=False, product_type='parse')

    def _render_first_page(self, source_file, dataset):
        image_root = Path(LMUDataRoot()) / 'images' / dataset
        image_root.mkdir(parents=True, exist_ok=True)
        path_hash = hashlib.md5(str(source_file).encode('utf-8')).hexdigest()[:10]
        out_path = image_root / f'{source_file.stem}_{path_hash}.png'
        if out_path.exists():
            return str(out_path)

        suffix = source_file.suffix.lower()
        if suffix in {'.png', '.jpg', '.jpeg', '.jfif'}:
            return str(source_file)
        if suffix != '.pdf':
            raise ValueError(f'Unsupported ParseBench source file: {source_file}')

        try:
            import fitz
        except ImportError as e:
            raise ImportError('ParseBench PDF rendering requires pymupdf. Install with `pip install pymupdf`.') from e

        with fitz.open(str(source_file)) as doc:
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pix.save(str(out_path))
        return str(out_path)

    @staticmethod
    def _extract_json_object(text):
        text = str(text).strip()
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
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
        return None

    def _prediction_to_parse_output(self, prediction, example_id, pipeline_name):
        from parse_bench.schemas.parse_output import PageIR, ParseOutput, ParseLayoutPageIR

        obj = self._extract_json_object(prediction)
        if isinstance(obj, dict):
            markdown = str(obj.get('markdown') or obj.get('md') or '')
            pages_payload = obj.get('pages') or []
            layout_payload = obj.get('layout_pages') or obj.get('layout') or []
            if not markdown and pages_payload:
                markdown = '\n\n'.join(str(p.get('markdown') or p.get('md') or '') for p in pages_payload)
            if not markdown:
                markdown = str(prediction)

            pages = []
            for i, page in enumerate(pages_payload):
                if isinstance(page, dict):
                    pages.append(PageIR(
                        page_index=int(page.get('page_index', i)),
                        markdown=str(page.get('markdown') or page.get('md') or ''),
                    ))
            if not pages:
                pages = [PageIR(page_index=0, markdown=markdown)]

            layout_pages = []
            for page in layout_payload if isinstance(layout_payload, list) else []:
                if isinstance(page, dict):
                    layout_pages.append(ParseLayoutPageIR.model_validate(page))

            return ParseOutput(
                example_id=example_id,
                pipeline_name=pipeline_name,
                pages=pages,
                layout_pages=layout_pages,
                markdown=markdown,
            )

        markdown = str(prediction)
        return ParseOutput(
            example_id=example_id,
            pipeline_name=pipeline_name,
            pages=[PageIR(page_index=0, markdown=markdown)],
            markdown=markdown,
        )

    def _build_inference_result(self, line, pipeline_name):
        from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
        from parse_bench.schemas.product import ProductType

        test_id = str(line['test_id'])
        now = datetime.now()
        output = self._prediction_to_parse_output(line['prediction'], test_id, pipeline_name)
        return InferenceResult(
            request=InferenceRequest(
                example_id=test_id,
                source_file_path=str(line['source_file_path']),
                product_type=ProductType.PARSE,
            ),
            pipeline_name=pipeline_name,
            product_type=ProductType.PARSE,
            raw_output={'prediction': str(line['prediction'])},
            output=output,
            started_at=now,
            completed_at=now,
            latency_in_ms=0,
        )

    @staticmethod
    def _evaluate_one(inference_result, test_case):
        from parse_bench.evaluation.evaluators.layoutdet import LayoutDetectionEvaluator
        from parse_bench.evaluation.evaluators.parse import ParseEvaluator
        from parse_bench.evaluation.layout_adapters import create_layout_adapter_for_result
        from parse_bench.schemas.pipeline_io import InferenceResult
        from parse_bench.schemas.product import ProductType
        from parse_bench.test_cases.schema import LayoutDetectionTestCase

        if isinstance(test_case, LayoutDetectionTestCase):
            adapter = create_layout_adapter_for_result(inference_result)
            layout_output = adapter.to_layout_output(
                inference_result,
                page_filter=getattr(test_case, 'page_index', 0) + 1,
            )
            layout_result = InferenceResult(
                request=inference_result.request,
                pipeline_name=inference_result.pipeline_name,
                product_type=ProductType.LAYOUT_DETECTION,
                raw_output=inference_result.raw_output,
                output=layout_output,
                started_at=inference_result.started_at,
                completed_at=inference_result.completed_at,
                latency_in_ms=inference_result.latency_in_ms,
            )
            return LayoutDetectionEvaluator().evaluate(layout_result, test_case)

        return ParseEvaluator().evaluate(inference_result, test_case)

    @staticmethod
    def _summarize_results(results, detailed):
        metric_rows = []
        for result in results:
            if not result.success:
                continue
            group = result.test_id.split('/')[0]
            for metric in result.metrics:
                metric_rows.append({
                    'group': group,
                    'metric': metric.metric_name,
                    'value': metric.value,
                })

        summary = {
            'Overall': 0.0,
            'Tables': 0.0,
            'Charts': 0.0,
            'Content Faithfulness': 0.0,
            'Semantic Formatting': 0.0,
            'Visual Grounding': 0.0,
            'Evaluated': int(len(results)),
            'Failed': int((detailed['success'] == False).sum()) if 'success' in detailed else 0,  # noqa: E712
        }
        if not metric_rows:
            return summary

        metrics = pd.DataFrame(metric_rows)
        group_name = {
            'table': 'Tables',
            'chart': 'Charts',
            'text_content': 'Content Faithfulness',
            'text_formatting': 'Semantic Formatting',
            'layout': 'Visual Grounding',
        }
        preferred = {
            'table': ('grits_trm_composite', 'table_record_match', 'grits_con'),
            'chart': ('rule_pass_rate', 'chart_data', 'content_faithfulness'),
            'text_content': ('content_faithfulness', 'rule_based'),
            'text_formatting': ('semantic_formatting', 'rule_based'),
            'layout': ('parse_field_element_pass_rate', 'element_pass_rate', 'f1'),
        }

        group_scores = []
        for group, label in group_name.items():
            sub = metrics[metrics['group'] == group]
            if sub.empty:
                continue
            chosen = None
            for key in preferred[group]:
                cand = sub[sub['metric'].str.lower().str.contains(key)]
                if not cand.empty:
                    chosen = cand
                    break
            if chosen is None:
                chosen = sub
            score = float(np.mean(chosen['value'])) * 100
            summary[label] = round(score, 2)
            group_scores.append(score)

        if group_scores:
            summary['Overall'] = round(float(np.mean(group_scores)), 2)
        return summary
