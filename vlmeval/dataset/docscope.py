import hashlib
import json
import os
import os.path as osp
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, get_logger, load
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils.docscope_reasoning_prompts import (INFER_SYSTEM_PROMPT, page_marker_end,
                                               page_marker_start, page_prf, parse_model_output)
from .utils.judge_util import build_judge

logger = get_logger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent / 'utils' / 'docscope_prompts'
HF_REPO_ID = 'MiliLab/DocScope'
FAIL_MSG = 'Failed to obtain answer via API.'

_DUMP_LOCK = Lock()


def _import_fitz():
    try:
        import fitz
    except ImportError as err:
        logger.error(
            'Failed to import fitz from PyMuPDF: %s. Install it in the VLMEvalKit '
            'environment with `pip install PyMuPDF`.',
            err,
        )
        raise
    return fitz


@dataclass
class DocScopeBenchEntry:
    qid: str
    question: str
    doc_id: str
    answer_text: str
    is_answerable: bool
    evidences: list[dict]
    facts: list[dict]

    def gold_pages(self):
        return sorted({
            int(e['page']) for e in self.evidences
            if isinstance(e, dict) and isinstance(e.get('page'), (int, float))
        })

    def evidences_on_page(self, page):
        return [e for e in self.evidences if e.get('page') == page]

    def facts_on_page(self, page):
        ev_ids = {e['local_id'] for e in self.evidences_on_page(page) if e.get('local_id')}
        return [f for f in self.facts if f.get('evidence_local_id') in ev_ids]

    def facts_off_page(self, page):
        ev_ids = {e['local_id'] for e in self.evidences_on_page(page) if e.get('local_id')}
        return [f for f in self.facts if f.get('evidence_local_id') not in ev_ids]


def _is_missing(value):
    return value is None or (isinstance(value, float) and pd.isna(value))


def _json_dumps(value):
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value, default=None):
    if isinstance(value, (list, dict)):
        return value
    if _is_missing(value):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


def _load_jsonl(path):
    rows = []
    if not osp.exists(path):
        return rows
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _jsonl_done_keys(path, fields):
    keys = set()
    for rec in _load_jsonl(path):
        parts = []
        for field in fields:
            value = rec.get(field)
            if value is None:
                break
            if field == 'page':
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    break
            else:
                value = str(value)
            parts.append(value)
        else:
            keys.add(tuple(parts) if len(parts) > 1 else parts[0])
    return keys


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _append_jsonl(path, rec):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _append_message_dump(path, rec):
    if not path:
        return
    os.makedirs(osp.dirname(path), exist_ok=True)
    with _DUMP_LOCK:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _normalize_messages_for_dump(messages, image_path_map=None):
    image_path_map = image_path_map or {}

    def convert(obj):
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        if not isinstance(obj, dict):
            return obj
        out = {k: convert(v) for k, v in obj.items()}
        if obj.get('type') == 'image_url':
            image_url = dict(obj.get('image_url') or {})
            url = image_url.get('url', '')
            path = image_path_map.get(url) or image_url.get('path')
            if path and osp.exists(path):
                image_url['url'] = str(path)
                image_url['sha256'] = _sha256_file(path)
                image_url['bytes'] = osp.getsize(path)
            out['image_url'] = image_url
        return out

    return convert(messages)


def _parse_judge_json(text):
    text = (text or '').strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find('{')
    end = text.rfind('}')
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return None


def _font(size=16):
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/Library/Fonts/Arial.ttf',
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_box(draw, bbox_px, color, label, width=4):
    x1, y1, x2, y2 = bbox_px
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if not label:
        return
    font = _font(16)
    try:
        box = draw.textbbox((0, 0), label, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
    except Exception:
        tw, th = 20, 16
    pad = 2
    ty = max(0, y1 - th - 2 * pad)
    draw.rectangle([x1, ty, x1 + tw + 2 * pad, ty + th + 2 * pad], fill=color)
    draw.text((x1 + pad, ty + pad), label, fill='white', font=font)


def _render_pdf_page(pdf_path, page, dpi, out_path):
    fitz = _import_fitz()

    out_path = Path(out_path)
    if out_path.exists():
        if out_path.stat().st_size > 0:
            return out_path
        logger.warning(f'Removing empty DocScope page cache: {out_path}')
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    try:
        if page < 1 or page > len(doc):
            return None
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = doc[page - 1].get_pixmap(matrix=matrix, alpha=False)
        pix.save(out_path.as_posix())
    finally:
        doc.close()
    return out_path


def _render_pair(pdf_path, page, gold_bboxes_px, pred_bboxes_norm, out_path, cache_root):
    src = _render_pdf_page(pdf_path, page, 144, Path(cache_root) / Path(pdf_path).stem / f'page_{page:04d}.png')
    if src is None:
        return False
    try:
        im = Image.open(src).convert('RGB')
    except OSError as err:
        logger.warning(f'Removing corrupt DocScope page cache {src}: {err}')
        Path(src).unlink(missing_ok=True)
        src = _render_pdf_page(pdf_path, page, 144, Path(cache_root) / Path(pdf_path).stem / f'page_{page:04d}.png')
        if src is None:
            return False
        im = Image.open(src).convert('RGB')
    width, height = im.size
    draw = ImageDraw.Draw(im)
    for i, gb in enumerate(gold_bboxes_px or []):
        if not gb or len(gb) != 4:
            continue
        label = 'GOLD' if len(gold_bboxes_px) == 1 else f'GOLD[{i + 1}]'
        _draw_box(draw, tuple(gb), 'green', label)
    pred_count = len([b for b in pred_bboxes_norm if b and len(b) == 4])
    for idx, b in enumerate(pred_bboxes_norm):
        if not b or len(b) != 4:
            continue
        box_px = (b[0] * width, b[1] * height, b[2] * width, b[3] * height)
        label = 'PRED' if pred_count == 1 else f'PRED[{idx + 1}]'
        _draw_box(draw, box_px, 'red', label)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return True


def _format_gold_block(evs_on_page):
    lines = []
    g_meta = []
    for i, ev in enumerate(evs_on_page, start=1):
        gid = f'g{i}'
        bbox = ev.get('bbox') or []
        line = (
            f'id={gid} :: index_on_page={i}, '
            f"element_type={ev.get('element_type', 'unknown')}, "
            f'gold_bbox_px={bbox}'
        )
        lines.append(line)
        g_meta.append({
            'gold_id': gid,
            'evidence_local_id': ev.get('local_id'),
            'element_type': ev.get('element_type'),
            'gold_bbox_px': bbox,
        })
    return '\n'.join(lines), g_meta


def _format_facts_block(facts_on_page):
    lines = []
    meta = []
    for fa in facts_on_page:
        fid = fa.get('local_id') or f'f{len(meta) + 1}'
        text = fa.get('text_description', '') or ''
        lines.append(f'id={fid} :: {text}')
        meta.append({
            'fact_id': fid,
            'evidence_local_id': fa.get('evidence_local_id'),
            'key_entity': fa.get('key_entity'),
            'key_value': fa.get('key_value'),
            'fact_text': text,
        })
    return '\n'.join(lines), meta


def _format_siblings(facts_off_page):
    if not facts_off_page:
        return '(none)'
    return '; '.join(f.get('text_description', '') or '' for f in facts_off_page)


class DocScope(ImageBaseDataset):
    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    DEFAULT_JUDGE = 'gpt-4o-mini'
    DATASET_URL = {'DocScope': '', 'DocScope_DEV': '', 'DocScope_TEST': ''}
    DATASET_MD5 = {}
    force_use_dataset_prompt = True

    def __init__(
        self,
        dataset='DocScope_TEST',
        max_pages=100,
        dpi=72,
        auto_download=True,
        **kwargs,
    ):
        self.docscope_root = Path(LMUDataRoot()) / 'DocScope'
        self.benchmark_path = self.docscope_root / 'benchmark.json'
        self.pdf_dir = self.docscope_root / 'pdfs'
        self.max_pages = int(max_pages)
        self.dpi = int(dpi)
        self.auto_download = str(auto_download).lower() not in {'0', 'false', 'no'}
        super().__init__(dataset=dataset, skip_noimg=False)

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    @staticmethod
    def _split_for_dataset(dataset):
        name = dataset.lower()
        if name.endswith('_dev'):
            return 'dev'
        if name.endswith('_test'):
            return 'test'
        return None

    def _setup_hf_cache(self):
        cache_root = self.docscope_root / '.hf_cache'
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ['HF_HOME'] = str(cache_root / 'home')
        os.environ['HF_HUB_CACHE'] = str(cache_root / 'hub')
        os.environ['HF_XET_CACHE'] = str(cache_root / 'xet')

    def _download_hf_file(self, filename):
        self._setup_hf_cache()
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError as err:
            raise ImportError(
                'huggingface_hub is required for DocScope auto download. '
                'Install it before building the DocScope dataset.'
            ) from err
        self.docscope_root.mkdir(parents=True, exist_ok=True)
        if filename == 'benchmark.json':
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type='dataset',
                local_dir=str(self.docscope_root),
                allow_patterns=['benchmark.json'],
            )
            return self.docscope_root / 'benchmark.json'
        return Path(hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type='dataset',
            filename=filename,
            local_dir=str(self.docscope_root),
        ))

    def _ensure_benchmark(self):
        if self.benchmark_path.exists():
            return self.benchmark_path
        if not self.auto_download:
            raise FileNotFoundError(f'DocScope benchmark.json not found: {self.benchmark_path}')
        logger.warning(f'DocScope benchmark.json not found. Downloading from Hugging Face to {self.docscope_root}.')
        return self._download_hf_file('benchmark.json')

    def _ensure_pdf(self, doc_id):
        candidate = self.pdf_dir / f'{doc_id}.pdf'
        if candidate.exists():
            return candidate
        if not self.auto_download:
            raise FileNotFoundError(f'DocScope PDF not found for {doc_id}: {candidate}')
        logger.warning(f'DocScope PDF {doc_id}.pdf not found. Downloading from Hugging Face.')
        path = self._download_hf_file(f'pdfs/{doc_id}.pdf')
        if path.exists():
            return path
        raise FileNotFoundError(f'DocScope PDF download failed for {doc_id}')

    @classmethod
    def _row_from_entry(cls, entry):
        pdf_block = entry.get('pdf') or {}
        answer_block = entry.get('answer') or {}
        evidences = entry.get('evidences') or []
        facts = entry.get('facts') or []
        pages = sorted({
            int(e['page']) for e in evidences
            if isinstance(e, dict) and isinstance(e.get('page'), (int, float))
        })
        qid = entry.get('id') or entry.get('question_id') or ''
        return {
            'index': qid,
            'question_id': qid,
            'question': entry.get('question', ''),
            'answer': answer_block.get('answer_text', '') or '',
            'is_answerable': bool(answer_block.get('is_answerable', True)),
            'doc_id': pdf_block.get('doc_id_str', ''),
            'gold_pages': _json_dumps(pages),
            'evidences': _json_dumps(evidences),
            'facts': _json_dumps(facts),
            'extract_class': entry.get('extract_class', ''),
            'split': entry.get('split', ''),
        }

    def load_data(self, dataset):
        benchmark = self._ensure_benchmark()
        raw = json.loads(benchmark.read_text(encoding='utf-8'))
        rows = raw['data'] if isinstance(raw, dict) and 'data' in raw else raw
        split = self._split_for_dataset(dataset)
        if split:
            rows = [r for r in rows if r.get('split') == split]
        limit = int(os.environ.get('DOCSCOPE_LIMIT', '0') or 0)
        if limit > 0:
            rows = rows[:limit]
        return pd.DataFrame([self._row_from_entry(r) for r in rows])

    def _bench_entry_from_line(self, line):
        return DocScopeBenchEntry(
            qid=str(line['question_id']),
            question=str(line['question']),
            doc_id=str(line['doc_id']),
            answer_text=str(line.get('answer', '') or ''),
            is_answerable=bool(line.get('is_answerable', True)),
            evidences=_json_loads(line.get('evidences'), []),
            facts=_json_loads(line.get('facts'), []),
        )

    def _benchmark_index(self):
        return {
            str(line['question_id']): self._bench_entry_from_line(line)
            for _, line in self.data.iterrows()
        }

    def _render_pdf_pages(self, pdf_path):
        fitz = _import_fitz()

        cache_dir = Path(self.img_root) / 'page_cache' / Path(pdf_path).stem / f'{self.dpi}dpi'
        cache_dir.mkdir(parents=True, exist_ok=True)
        sentinel = cache_dir / '.done'
        if sentinel.exists():
            pages = sorted(cache_dir.glob('page_*.png'))
            if pages:
                return pages

        doc = fitz.open(str(pdf_path))
        out_paths = []
        try:
            matrix = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
            for i, page in enumerate(doc, start=1):
                out = cache_dir / f'page_{i:04d}.png'
                if not out.exists():
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    pix.save(out.as_posix())
                out_paths.append(out)
        finally:
            doc.close()
        sentinel.touch()
        return out_paths

    def dump_image(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        doc_id = str(line['doc_id'])
        pdf_path = self._ensure_pdf(doc_id)
        pages = self._render_pdf_pages(pdf_path)
        return [str(p) for p in pages[:self.max_pages]]

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        question = str(line['question'])
        page_paths = self.dump_image(line)
        page_numbers = list(range(1, len(page_paths) + 1))

        msgs = [dict(type='text', value=INFER_SYSTEM_PROMPT, role='system')]
        for page, img_path in zip(page_numbers, page_paths):
            msgs.append(dict(type='text', value=page_marker_start(page)))
            msgs.append(dict(type='image', value=img_path))
            msgs.append(dict(type='text', value=page_marker_end(page)))
        msgs.append(dict(type='text', value=f'Question: {question}'))
        return msgs

    def _records_from_eval(self, eval_file):
        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        elif isinstance(data, dict):
            data = pd.DataFrame(data)
        records = []
        for _, line in data.iterrows():
            qid = str(line.get('question_id', line.get('index', '')))
            prediction = str(line.get('prediction', '') or '')
            status = 'ok'
            if FAIL_MSG in prediction or prediction == '':
                status = 'error: empty prediction'
            records.append({
                'question_id': qid,
                'model': Path(eval_file).stem.split('_', 1)[0],
                'model_raw': prediction,
                'thinking': str(line.get('thinking', '') or ''),
                'parsed': parse_model_output(prediction) if status == 'ok' else {},
                'infer_usage': {},
                'status': status,
            })
        return records

    def _score_pages(self, records, bench, pages_file):
        existing = _load_jsonl(pages_file)
        done = {str(r.get('question_id')) for r in existing if r.get('question_id') is not None}
        sum_p = sum_r = sum_f1 = 0.0
        n = tp = fp = fn = 0
        for row in existing:
            if row.get('status') != 'ok':
                continue
            sum_p += _safe_float(row.get('precision'))
            sum_r += _safe_float(row.get('recall'))
            sum_f1 += _safe_float(row.get('f1'))
            tp += _safe_int(row.get('tp'))
            fp += _safe_int(row.get('fp'))
            fn += _safe_int(row.get('fn'))
            n += 1
        for record in records:
            qid = str(record.get('question_id'))
            if qid in done:
                continue
            if record.get('status') != 'ok':
                _append_jsonl(pages_file, {
                    'question_id': qid,
                    'status': record.get('status'),
                    'skipped': True,
                })
                continue
            be = bench.get(qid)
            parsed = record.get('parsed') or {}
            pred_pages = parsed.get('predicted_pages') or []
            gold_pages = be.gold_pages() if be else []
            prf = page_prf(pred_pages, gold_pages)
            _append_jsonl(pages_file, {
                'question_id': qid,
                'gold_pages': prf['gold'],
                'predicted_pages': prf['predicted'],
                'tp': prf['tp'],
                'fp': prf['fp'],
                'fn': prf['fn'],
                'precision': prf['precision'],
                'recall': prf['recall'],
                'f1': prf['f1'],
                'status': 'ok',
            })
            sum_p += prf['precision']
            sum_r += prf['recall']
            sum_f1 += prf['f1']
            tp += prf['tp']
            fp += prf['fp']
            fn += prf['fn']
            n += 1
        micro_p = tp / (tp + fp) if (tp + fp) else 0.0
        micro_r = tp / (tp + fn) if (tp + fn) else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
        return {
            'pages|macro_precision': sum_p / n if n else 0.0,
            'pages|macro_recall': sum_r / n if n else 0.0,
            'pages|macro_f1': sum_f1 / n if n else 0.0,
            'pages|micro_precision': micro_p,
            'pages|micro_recall': micro_r,
            'pages|micro_f1': micro_f1,
        }

    def _bbox_message(self, template, be, qid, page, record, imgs_dir):
        evs_on_page = be.evidences_on_page(page)
        citations = (record.get('parsed') or {}).get('citations') or []
        pred_on_page = [c for c in citations if c.get('page') == page]
        pred_norm = []
        pred_skipped = []
        for c in pred_on_page:
            bbox = c.get('bbox') or []
            if len(bbox) != 4:
                continue
            if c.get('is_normalized', False):
                pred_norm.append(bbox)
            else:
                pred_skipped.append(bbox)
        gold_block, g_meta = _format_gold_block(evs_on_page)
        img_path = Path(imgs_dir) / f"{qid.replace('::', '__')}__p{page}.png"
        pdf_path = self._ensure_pdf(be.doc_id)
        rendered = _render_pair(
            pdf_path,
            page,
            [ev.get('bbox') for ev in evs_on_page],
            pred_norm,
            img_path,
            Path(self.img_root) / 'eval_page_cache',
        )
        if not rendered:
            return None, g_meta, pred_skipped, pred_norm, img_path
        user_text = template.format(
            question=be.question,
            gold_answer=be.answer_text,
            gt_page=page,
            gold_total_on_page=len(evs_on_page),
            gold_facts_block=gold_block,
            n_pred=len(pred_norm),
            pred_bboxes_norm=pred_norm,
        )
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': str(img_path)}},
                {'type': 'text', 'text': user_text},
            ],
        }]
        return messages, g_meta, pred_skipped, pred_norm, img_path

    def _score_bbox(self, records, bench, bbox_file, imgs_dir, dump_path, dump_only, judge, judge_model, nproc):
        template = (PROMPT_DIR / 'evidence_grounding.txt').read_text(encoding='utf-8')
        done = _jsonl_done_keys(bbox_file, ('question_id', 'page'))
        tasks = []
        for record in records:
            if record.get('status') != 'ok':
                continue
            qid = str(record.get('question_id'))
            be = bench.get(qid)
            if be is None:
                continue
            gold_pages = set(be.gold_pages())
            pred_pages = set((record.get('parsed') or {}).get('predicted_pages') or [])
            for page in sorted(gold_pages & pred_pages):
                page = int(page)
                if (qid, page) in done:
                    continue
                evs = be.evidences_on_page(page)
                if not evs:
                    continue
                messages, g_meta, pred_skipped, pred_norm, img_path = self._bbox_message(
                    template, be, qid, page, record, imgs_dir)
                if messages is None:
                    for gm in g_meta:
                        _append_jsonl(
                            bbox_file,
                            {
                                'question_id': qid,
                                'page': page,
                                **gm,
                                'label': None,
                                'reason': 'render_failed',
                                'status': 'render_failed',
                                'skipped_pred_bboxes': pred_skipped,
                            },
                        )
                    continue
                dump_messages = _normalize_messages_for_dump(
                    messages, {str(img_path): str(img_path)})
                _append_message_dump(
                    dump_path,
                    {
                        'stage': 'judge_bbox',
                        'question_id': qid,
                        'page': page,
                        'messages': dump_messages,
                    },
                )
                tasks.append({
                    'qid': qid,
                    'page': page,
                    'g_meta': g_meta,
                    'pred_skipped': pred_skipped,
                    'pred_norm': pred_norm,
                    'img_path': str(img_path),
                    'user_text': messages[0]['content'][1]['text'],
                })

        def _judge_one_bbox(qid, page, g_meta, pred_skipped, pred_norm, img_path, user_text):
            if dump_only:
                return [{
                    'question_id': qid,
                    'page': page,
                    **gm,
                    'n_pred_drawn': len(pred_norm),
                    'skipped_pred_bboxes': pred_skipped,
                    'label': None,
                    'reason': 'message_dumped',
                    'judge_raw': '',
                    'judge_usage': {},
                    'status': 'message_dumped',
                } for gm in g_meta]

            try:
                response = judge.generate(
                    message=[
                        {'type': 'image', 'value': img_path},
                        {'type': 'text', 'value': user_text},
                    ],
                    dataset=self.dataset_name,
                )
                parsed = _parse_judge_json(str(response)) or {}
                items = parsed.get('items') or []
                by_id = {it.get('id'): it for it in items if isinstance(it, dict)}
                error = None
            except Exception as err:
                response = f'JUDGE_ERROR: {err}'
                by_id = {}
                error = str(err)

            rows = []
            for gm in g_meta:
                it = by_id.get(gm['gold_id']) or {}
                label = it.get('label')
                status = 'ok' if label in {'covered', 'imprecise', 'not_covered'} else 'judge_parse_error'
                if error is not None:
                    status = 'judge_error'
                rows.append({
                    'question_id': qid,
                    'page': page,
                    **gm,
                    'n_pred_drawn': len(pred_norm),
                    'skipped_pred_bboxes': pred_skipped,
                    'label': label,
                    'reason': it.get('reason') or error,
                    'judge_raw': str(response),
                    'judge_usage': {},
                    'status': status,
                })
            return rows

        if tasks:
            results = track_progress_rich(_judge_one_bbox, tasks, nproc=nproc)
            for rows in results:
                for row in rows:
                    _append_jsonl(bbox_file, row)
        return len(done) + len(tasks)

    def _score_facts(self, records, bench, facts_file, dump_path, dump_only, judge, nproc):
        template = (PROMPT_DIR / 'factucal_consistency.txt').read_text(encoding='utf-8')
        done = _jsonl_done_keys(facts_file, ('question_id', 'page'))
        tasks = []
        for record in records:
            if record.get('status') != 'ok':
                continue
            qid = str(record.get('question_id'))
            be = bench.get(qid)
            if be is None:
                continue
            gold_pages = set(be.gold_pages())
            pred_pages = set((record.get('parsed') or {}).get('predicted_pages') or [])
            for page in sorted(gold_pages & pred_pages):
                page = int(page)
                if (qid, page) in done:
                    continue
                facts_on = be.facts_on_page(page)
                if not facts_on:
                    continue
                facts_off = be.facts_off_page(page)
                facts_block, meta = _format_facts_block(facts_on)
                user_text = template.format(
                    question=be.question,
                    gt_page=page,
                    gold_facts_block=facts_block,
                    sibling_facts=_format_siblings(facts_off),
                    model_raw=record.get('model_raw', ''),
                )
                messages = [{'role': 'user', 'content': user_text}]
                _append_message_dump(dump_path, {
                    'stage': 'judge_facts',
                    'question_id': qid,
                    'page': page,
                    'messages': messages,
                })
                tasks.append({
                    'qid': qid,
                    'page': page,
                    'meta': meta,
                    'user_text': user_text,
                })

        def _judge_one_facts(qid, page, meta, user_text):
            if dump_only:
                return [{
                    'question_id': qid,
                    'page': page,
                    **m,
                    'label': None,
                    'reason': 'message_dumped',
                    'judge_raw': '',
                    'judge_usage': {},
                    'status': 'message_dumped',
                } for m in meta]

            try:
                response = judge.generate(user_text)
                parsed = _parse_judge_json(str(response)) or {}
                items = parsed.get('items') or []
                by_id = {it.get('id'): it for it in items if isinstance(it, dict)}
                error = None
            except Exception as err:
                response = f'JUDGE_ERROR: {err}'
                by_id = {}
                error = str(err)

            rows = []
            for m in meta:
                it = by_id.get(m['fact_id']) or {}
                label = it.get('label')
                status = 'ok' if label in {'consistent', 'not_consistent'} else 'judge_parse_error'
                if error is not None:
                    status = 'judge_error'
                rows.append({
                    'question_id': qid,
                    'page': page,
                    **m,
                    'label': label,
                    'reason': it.get('reason') or error,
                    'judge_raw': str(response),
                    'judge_usage': {},
                    'status': status,
                })
            return rows

        if tasks:
            results = track_progress_rich(_judge_one_facts, tasks, nproc=nproc)
            for rows in results:
                for row in rows:
                    _append_jsonl(facts_file, row)
        return len(done) + len(tasks)

    def _score_answer(self, records, bench, answer_file, dump_path, dump_only, judge, nproc):
        template = (PROMPT_DIR / 'answer_verification.txt').read_text(encoding='utf-8')
        done = _jsonl_done_keys(answer_file, ('question_id',))
        tasks = []
        for record in records:
            if record.get('status') != 'ok':
                continue
            qid = str(record.get('question_id'))
            if qid in done:
                continue
            be = bench.get(qid)
            parsed_record = record.get('parsed') or {}
            has_answer_tag = bool(parsed_record.get('has_answer_tag'))
            if has_answer_tag and parsed_record.get('answer'):
                model_answer = parsed_record.get('answer', '')
            else:
                model_answer = record.get('model_raw', '')
            if be is not None and not be.is_answerable:
                gold_answer = 'Unanswerable'
            else:
                gold_answer = be.answer_text if be else ''
            user_text = template.format(
                question=be.question if be else '',
                gold_answer=gold_answer,
                model_answer=model_answer,
            )
            messages = [{'role': 'user', 'content': user_text}]
            _append_message_dump(dump_path, {
                'stage': 'judge_answer',
                'question_id': qid,
                'messages': messages,
            })
            tasks.append({
                'qid': qid,
                'gold_answer': gold_answer,
                'model_answer': model_answer,
                'has_answer_tag': has_answer_tag,
                'user_text': user_text,
            })

        def _judge_one_answer(qid, gold_answer, model_answer, has_answer_tag, user_text):
            if dump_only:
                return {
                    'question_id': qid,
                    'gold_answer': gold_answer,
                    'model_answer': model_answer,
                    'has_answer_tag': has_answer_tag,
                    'consistent': None,
                    'reason': 'message_dumped',
                    'judge_raw': '',
                    'judge_usage': {},
                    'status': 'message_dumped',
                }

            try:
                response = judge.generate(user_text)
                parsed = _parse_judge_json(str(response)) or {}
                consistent = parsed.get('consistent')
                error = None
            except Exception as err:
                response = f'JUDGE_ERROR: {err}'
                parsed = {}
                consistent = None
                error = str(err)

            status = 'ok' if isinstance(consistent, bool) else 'judge_parse_error'
            if error is not None:
                status = 'judge_error'
            return {
                'question_id': qid,
                'gold_answer': gold_answer,
                'model_answer': model_answer,
                'has_answer_tag': has_answer_tag,
                'consistent': bool(consistent) if isinstance(consistent, bool) else None,
                'reason': parsed.get('reason') or parsed.get('reasoning') or error or '',
                'judge_raw': str(response),
                'judge_usage': {},
                'status': status,
            }

        if tasks:
            rows = track_progress_rich(_judge_one_answer, tasks, nproc=nproc)
            for row in rows:
                _append_jsonl(answer_file, row)
        return len(done) + len(tasks)

    def evaluate(self, eval_file, **judge_kwargs):
        dump_path = os.environ.get('DOCSCOPE_DUMP_MESSAGES', '')
        dump_only = bool(dump_path) or bool(judge_kwargs.pop('dump_messages', False))
        nproc = judge_kwargs.pop('nproc', 4)
        model_name = judge_kwargs.get('model', self.DEFAULT_JUDGE)

        records = self._records_from_eval(eval_file)
        bench = self._benchmark_index()
        pages_file = get_intermediate_file_path(eval_file, '_pages', 'jsonl')
        bbox_file = get_intermediate_file_path(eval_file, f'_{model_name}_bbox', 'jsonl')
        facts_file = get_intermediate_file_path(eval_file, f'_{model_name}_facts', 'jsonl')
        answer_file = get_intermediate_file_path(eval_file, f'_{model_name}_answer', 'jsonl')
        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        imgs_dir = Path(self.img_root) / 'eval_imgs' / Path(eval_file).stem

        summary = self._score_pages(records, bench, pages_file)
        judge = None if dump_only else build_judge(max_tokens=2048, **judge_kwargs)
        summary['bbox|dumped_messages'] = self._score_bbox(
            records, bench, bbox_file, imgs_dir, dump_path, dump_only, judge, model_name, nproc)
        summary['facts|dumped_messages'] = self._score_facts(
            records, bench, facts_file, dump_path, dump_only, judge, nproc)
        summary['answer|dumped_messages'] = self._score_answer(
            records, bench, answer_file, dump_path, dump_only, judge, nproc)
        summary['status'] = 'message_dumped' if dump_only else 'ok'
        dump(summary, score_file)
        return summary
