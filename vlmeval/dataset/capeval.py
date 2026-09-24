import json
import os.path as osp
import re
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.utils import DEBUG_MESSAGE, build_judge
from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load, modelscope_flag_set
from vlmeval.utils import track_progress_rich

HF_REPO = 'LiuzhipengUCAS/CAPEval'
EXPECTED_N_IMAGES = 300
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff'}
PROMPT_VERSION = 'official-single-pass-v1'
EVALUATOR_VERSION = 'capeval-cp-v1'
CAPTION_PROMPT = 'Analyze the image in a comprehensive and detailed manner.'

SINGLE_PASS_SYSTEM = """You are an expert image-caption judge for a research benchmark.

Behavior:
- Apply a moderately strict standard: reward clear, accurate coverage; penalize contradictions.
- When the caption is genuinely ambiguous about a checklist point, prefer "not_mentioned" over guessing "yes".
- Use only the caption text and the checklist metadata provided; do not invent scene facts from prior knowledge.
- Output must be one valid JSON object exactly in the form requested by the user—no markdown code fences, no text before or after."""  # noqa: E501

CHECKLIST_KEYS = [
    'instance_checklist', 'attribute_checklist', 'relation_checklist', 'image_checklist',
    'text_checklist', 'human_checklist', 'ui_checklist', 'world_knowledge_checklist',
]
CATEGORY_NAMES = {
    'SO': 'Scene & Object',
    'PA': 'People & Activity',
    'TI': 'Text & Interface',
    'DK': 'Design & Knowledge',
}
_YES_TOKENS = frozenset({'yes', 'y', 'correct', 'correctly_mentioned', 'b'})
_NO_TOKENS = frozenset({'no', 'n', 'incorrect', 'incorrectly_mentioned', 'c'})
_NM_TOKENS = frozenset({
    'not_mentioned', 'not mentioned', 'not-mentioned', 'notmentioned',
    'nm', 'a', 'unknown', 'n/a', 'na',
})


def _data_root():
    return Path(LMUDataRoot()) / 'CAPEval'


def _count_images(image_dir):
    if not image_dir.is_dir():
        return 0
    return sum(
        1 for p in image_dir.iterdir()
        if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in IMAGE_SUFFIXES
    )


def locate_capeval_files(root):
    checklist_cands = [root / 'checklist.jsonl', root / 'data' / 'checklist.jsonl']
    image_cands = [root / 'image', root / 'images', root / 'data' / 'image', root / 'data' / 'images']
    checklist = next((p for p in checklist_cands if p.is_file()), None)
    image_dir = next((p for p in image_cands if p.is_dir()), None)
    if checklist is None or image_dir is None or _count_images(image_dir) < EXPECTED_N_IMAGES:
        return None
    return checklist, image_dir


def _snapshot_download(local_dir):
    repo = HF_REPO
    if modelscope_flag_set():
        from modelscope import dataset_snapshot_download
        return dataset_snapshot_download(dataset_id=repo, local_dir=str(local_dir))
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            'CAPEval auto-download needs huggingface_hub (already in VLMEvalKit '
            f'requirements.txt). Place checklist.jsonl + image/ under {local_dir}.'
        ) from e
    return snapshot_download(repo_id=repo, repo_type='dataset', local_dir=str(local_dir))


def ensure_capeval_data():
    root = _data_root()
    found = locate_capeval_files(root)
    if found is not None:
        return found
    root.mkdir(parents=True, exist_ok=True)
    _snapshot_download(root)
    found = locate_capeval_files(root)
    if found is None:
        raise FileNotFoundError(
            f'Failed to prepare CAPEval under {root}. '
            f'Expected checklist.jsonl and {EXPECTED_N_IMAGES} images ({HF_REPO}).'
        )
    return found


def _safe_model_name(model_name):
    return str(model_name).replace('/', '_').replace(':', '_')


def load_jsonl(path, checklist_rows_only=False):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.lstrip('\ufeff').strip()
    if not text:
        return []
    dec = json.JSONDecoder()
    rows, idx, n = [], 0, len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = dec.raw_decode(text, idx)
        idx = end
        if not isinstance(obj, dict):
            continue
        if checklist_rows_only and not any(obj.get(k) for k in CHECKLIST_KEYS):
            continue
        rows.append(obj)
    return rows


def build_checklist_items_with_index(entry):
    items = []
    idx = 0
    for ck in CHECKLIST_KEYS:
        for item in entry.get(ck, []) or []:
            if not isinstance(item, dict):
                continue
            q = str(item.get('Question', '')).strip()
            if not q:
                continue
            tags = item.get('Tags', '') or ''
            items.append({
                'item_index': idx,
                'checklist_type': ck,
                'tags': tags,
                'tag': tags,
                'question': q,
            })
            idx += 1
    return items


def domain_from_img_path(img_path):
    stem = Path(str(img_path)).stem
    prefix = []
    for ch in stem:
        if ch.isalpha():
            prefix.append(ch)
        else:
            break
    return ''.join(prefix).upper() if prefix else 'unknown'


def canonicalize_verdict(raw):
    if raw is None:
        return None
    ver = str(raw).strip().lower()
    if not ver:
        return None
    ver = ver.strip('\'"').rstrip('.,;:').strip()
    ver = re.sub(r'\s+', ' ', ver)
    compact = ver.replace(' ', '_').replace('-', '_')
    if ver in _YES_TOKENS or compact in _YES_TOKENS:
        return 'yes'
    if ver in _NO_TOKENS or compact in _NO_TOKENS:
        return 'no'
    nm_compact = {t.replace(' ', '_').replace('-', '_') for t in _NM_TOKENS}
    if ver in _NM_TOKENS or compact in nm_compact:
        return 'not_mentioned'
    return None


def parse_single_pass_json(raw, n_items):
    text = raw or ''
    for sep in ('</think>', '</thinking>'):
        if sep in text:
            text = text.split(sep, 1)[-1]
    text = text.strip()
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        text = m.group(0)
    decode_err = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        decode_err = f'json_decode: {e}'
        data = None
    if data is None:
        by_idx, by_reason = {}, {}
        item_pat = re.compile(
            r'\{[^{}]*?"item_index"\s*:\s*(\d+)[^{}]*?"verdict"\s*:\s*"([^"]*)"'
            r'[^{}]*?"reasoning"\s*:\s*"([^"]*)"[^{}]*?\}',
            re.S,
        )
        for mm in item_pat.finditer(text):
            ver = canonicalize_verdict(mm.group(2))
            if ver is None:
                continue
            ii = int(mm.group(1))
            by_idx[ii] = ver
            by_reason[ii] = mm.group(3).strip()
        if by_idx:
            return {
                'verdicts': [by_idx.get(i, 'not_mentioned') for i in range(n_items)],
                'reasonings': [by_reason.get(i, '') for i in range(n_items)],
            }, None
        return None, decode_err or 'json_decode_unknown'
    if not isinstance(data, dict):
        return None, 'root_not_object'
    verdicts = data.get('gt_verdicts')
    if not isinstance(verdicts, list):
        return None, 'missing_gt_verdicts'
    by_idx, by_reason = {}, {}
    for v in verdicts:
        if not isinstance(v, dict) or v.get('item_index') is None:
            continue
        try:
            ii = int(v['item_index'])
        except (TypeError, ValueError):
            continue
        ver = canonicalize_verdict(v.get('verdict', ''))
        if ver is None:
            continue
        by_idx[ii] = ver
        by_reason[ii] = str(v.get('reasoning', '') or '').strip()
    return {
        'verdicts': [by_idx.get(i, 'not_mentioned') for i in range(n_items)],
        'reasonings': [by_reason.get(i, '') for i in range(n_items)],
    }, None


def normalize_verdict_list(verdicts):
    y = n = nm = 0
    for v in verdicts:
        if v == 'yes':
            y += 1
        elif v == 'no':
            n += 1
        else:
            nm += 1
    return y, n, nm


def _cp_percent(yes, no, total):
    c = 100.0 * (yes + no) / total if total else 0.0
    mentioned = yes + no
    p = 100.0 * yes / mentioned if mentioned else 0.0
    return c, p


def _empty_bucket():
    return {'yes1': 0, 'no1': 0, 'not_mentioned1': 0, 'total1': 0, 'n_images': 0, 'n_error': 0}


def _summary_block(bucket):
    sy, sn, st = bucket['yes1'], bucket['no1'], bucket['total1']
    c, p = _cp_percent(sy, sn, st)
    out = {
        'C': round(c, 4), 'P': round(p, 4),
        'yes1': sy, 'no1': sn, 'not_mentioned1': bucket['not_mentioned1'],
        'total1': st, 'n_images': bucket['n_images'],
    }
    if bucket.get('n_error'):
        out['n_error'] = int(bucket['n_error'])
    return out


def aggregate_records(records):
    g = {**_empty_bucket(), 'by_category': defaultdict(_empty_bucket)}
    for row in records:
        if row.get('status') != 'ok':
            g['n_error'] += 1
            continue
        y1 = int(row.get('yes1', 0) or 0)
        n1 = int(row.get('no1', 0) or 0)
        nm1 = int(row.get('not_mentioned1', 0) or 0)
        t1 = int(row.get('total1', 0) or 0)
        g['yes1'] += y1
        g['no1'] += n1
        g['not_mentioned1'] += nm1
        g['total1'] += t1
        g['n_images'] += 1
        cat = CATEGORY_NAMES.get((row.get('domain') or '').upper(), row.get('domain') or 'unknown')
        bc = g['by_category'][cat]
        bc['yes1'] += y1
        bc['no1'] += n1
        bc['not_mentioned1'] += nm1
        bc['total1'] += t1
        bc['n_images'] += 1
    order = list(CATEGORY_NAMES.values())
    cats = sorted(g['by_category'].keys(), key=lambda x: (order.index(x) if x in order else 99, x))
    return {
        'summary': _summary_block(g),
        'per_category': {name: _summary_block(g['by_category'][name]) for name in cats},
    }


def build_single_pass_user_prompt(caption, checklist_items):
    lines = [
        'Evaluate the CAPTION against each checklist item below.',
        '',
        'For each item_index, based ONLY on the caption, assign exactly one verdict:',
        '  - "yes": the caption correctly covers the factual content that this checklist item is asking about—'
        'the caption supports the proposition in the question without contradiction.',
        '  - "no": the caption engages with the same topic as the question but contradicts it or is clearly inconsistent.',  # noqa: E501
        '  - "not_mentioned": the caption does not address this checklist point (or is too vague to decide).',
        'If the question is phrased negatively (e.g. whether something is absent), '
        '"yes" means the caption is consistent with that negative claim—not that you are answering the word "yes" to English grammar.',  # noqa: E501
        '',
        'For every verdict you MUST include a short "reasoning" field (one concise sentence) explaining',
        'the evidence from the caption (for calibration and auditing).',
        '',
        'CAPTION:',
        caption,
        '',
        'CHECKLIST (metadata is for context; the Question text is primary):',
        '  - item_index: stable id you must echo in gt_verdicts.',
        '  - tag: fine-grained label for downstream analysis (e.g. color, spatial); do not replace the Question.',
        '  - type: which checklist channel this question belongs to (e.g. attribute vs relation); use the Question as ground truth.',  # noqa: E501
        '',
    ]
    for it in checklist_items:
        lines.append(
            f"  item_index={it['item_index']}  tag={it.get('tags', '')!r}  "
            f"type={it['checklist_type']}  Q: {it['question']}"
        )
    lines.extend([
        '',
        'Return ONLY a JSON object (no markdown) with exactly this structure:',
        '{"gt_verdicts":['
        '{"item_index":<int>,"verdict":"yes"|"no"|"not_mentioned","reasoning":"<one short sentence>"},'
        '...]}',
        'Rules:',
        '- Every item_index from the checklist must appear exactly once in gt_verdicts.',
        '- Every gt_verdicts entry must include non-empty reasoning (at least a few words).',
    ])
    return '\n'.join(lines)


def captions_from_eval_file(data):
    captions = {}
    for _, row in data.iterrows():
        pred = row.get('prediction', '')
        if pred is None or (isinstance(pred, float) and str(pred) == 'nan'):
            continue
        key = row.get('image_id') or row.get('img_path')
        if key is None or (isinstance(key, float) and str(key) == 'nan'):
            img = row.get('image_path', row.get('image', ''))
            key = Path(str(img)).name if img is not None else None
        if not key:
            continue
        captions[str(Path(str(key)).name)] = str(pred)
    return captions


def prepare_units(captions, checklist_path):
    cl_by_img = {}
    for r in load_jsonl(checklist_path, checklist_rows_only=True):
        p = str(r.get('img_path') or '').strip()
        if not p:
            continue
        cl_by_img[p] = r
        cl_by_img[Path(p).name] = r
    units, skipped = [], 0
    for cap_key, caption in captions.items():
        name = Path(str(cap_key)).name
        cl_row = cl_by_img.get(name) or cl_by_img.get(str(cap_key))
        if cl_row is None or caption is None:
            skipped += 1
            continue
        img_path = str(cl_row.get('img_path') or name).strip()
        items = build_checklist_items_with_index(cl_row)
        if not img_path or not items:
            skipped += 1
            continue
        units.append({
            'image_id': img_path,
            'domain': domain_from_img_path(img_path),
            'caption': caption,
            'checklist_items': items,
        })
    if not units:
        raise RuntimeError(
            'No caption keys matched CAPEval checklist img_path '
            f'(skipped={skipped}). Expected keys like SO001.jpg.'
        )
    return units


def is_ok_record(rec):
    return isinstance(rec, dict) and rec.get('status') == 'ok'


def load_cached_answers(tmp_file, storage_file):
    ans = {}
    if osp.exists(storage_file):
        records = load(storage_file)
        if isinstance(records, list):
            for rec in records:
                if isinstance(rec, dict) and rec.get('image_id'):
                    ans[rec['image_id']] = rec
        elif isinstance(records, dict):
            ans.update(records)
    if osp.exists(tmp_file):
        loaded = load(tmp_file)
        if isinstance(loaded, dict):
            ans.update(loaded)
    return ans


def pending_units(units, ans):
    return [u for u in units if not is_ok_record(ans.get(u['image_id']))]


def CAPEval_atomeval(model, prompt, n_items):
    raw = model.generate(prompt)
    parsed, err = parse_single_pass_json(raw, n_items)
    if parsed is None:
        return dict(status='error', error=err, raw_response=raw or '',
                    yes1=0, no1=0, not_mentioned1=0, total1=0)
    y, n, nm = normalize_verdict_list(parsed['verdicts'])
    reasons = parsed.get('reasonings') or [''] * n_items
    while len(reasons) < n_items:
        reasons.append('')
    return dict(
        status='ok',
        raw_response=raw or '',
        yes1=y, no1=n, not_mentioned1=nm, total1=n_items,
        gt_verdicts=[
            {'item_index': j, 'verdict': parsed['verdicts'][j], 'reasoning': reasons[j]}
            for j in range(n_items)
        ],
    )


def assemble_records(units, ans):
    records = []
    for u in units:
        rec = dict(ans.get(u['image_id']) or {})
        rec['image_id'] = u['image_id']
        rec['domain'] = u['domain']
        rec['caption'] = u['caption']
        rec['checklist_items'] = u['checklist_items']
        if not is_ok_record(rec) and 'status' not in rec:
            rec.update(status='error', error='missing_judge_result',
                       yes1=0, no1=0, not_mentioned1=0, total1=0)
        records.append(rec)
    return records


def records_to_score_df(records, *, judge_model=None, prompt_version=None, evaluator_version=None):
    metrics = aggregate_records(records)
    summary = metrics['summary']
    n_error = int(summary.get('n_error', 0) or 0)
    n_ok = int(summary.get('n_images', 0) or 0)
    row = {
        'C': summary.get('C'),
        'P': summary.get('P'),
        'n_images': n_ok,
        'n_error': n_error,
        'eval_status': 'partial' if n_error else 'ok',
    }
    if judge_model is not None:
        row['judge_model'] = judge_model
    if prompt_version is not None:
        row['prompt_version'] = prompt_version
    if evaluator_version is not None:
        row['evaluator_version'] = evaluator_version
    inv = {v: k for k, v in CATEGORY_NAMES.items()}
    for cat, blk in (metrics.get('per_category') or {}).items():
        short = inv.get(cat, cat)
        row[f'C_{short}'] = blk.get('C')
        row[f'P_{short}'] = blk.get('P')
    return pd.DataFrame([row])


class CAPEval(ImageBaseDataset):
    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    DEFAULT_JUDGE = 'qwen-72b'
    force_use_dataset_prompt = True
    DATASET_URL = {'CAPEval': f'https://huggingface.co/datasets/{HF_REPO}'}
    DATASET_MD5 = {}

    def __init__(self, dataset='CAPEval', **kwargs):
        super().__init__(dataset=dataset, skip_noimg=False)

    @classmethod
    def supported_datasets(cls):
        return ['CAPEval']

    def load_data(self, dataset='CAPEval'):
        checklist_path, image_root = ensure_capeval_data()
        self.checklist_path = str(checklist_path)
        self.image_root = str(image_root)
        rows = []
        for i, obj in enumerate(load_jsonl(str(checklist_path), checklist_rows_only=True)):
            img_path = str(obj.get('img_path') or '').strip()
            if not img_path:
                continue
            stem = Path(img_path).stem
            prefix = ''.join(ch for ch in stem if ch.isalpha()) or 'unknown'
            rows.append({
                'index': i,
                'image_id': img_path,
                'image_path': str(image_root / Path(img_path).name),
                'question': CAPTION_PROMPT,
                'category': prefix.upper(),
            })
        if len(rows) < EXPECTED_N_IMAGES:
            warnings.warn(
                f'CAPEval checklist has {len(rows)} image rows; expected {EXPECTED_N_IMAGES}.'
            )
        return pd.DataFrame(rows)

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        for m in msgs:
            if m.get('type') == 'image' and not osp.isfile(str(m.get('value', ''))):
                raise FileNotFoundError(f'CAPEval image not found: {m.get("value")}')
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        nproc = judge_kwargs.pop('nproc', 4)
        judge_kwargs.pop('use_vllm', None)
        judge_kwargs.pop('use_verifier', None)
        judge_model = judge_kwargs.pop('model', None) or self.DEFAULT_JUDGE
        cache_tag = f'{_safe_model_name(judge_model)}_{PROMPT_VERSION}_{EVALUATOR_VERSION}'
        storage = get_intermediate_file_path(eval_file, f'_{cache_tag}_judge', 'json')
        tmp_file = get_intermediate_file_path(eval_file, f'_{cache_tag}_judge_tmp', 'pkl')
        score_file = get_intermediate_file_path(eval_file, f'_{cache_tag}_score', 'csv')

        captions = captions_from_eval_file(load(eval_file))
        if not captions:
            raise RuntimeError(f'No captions found in {eval_file}.')
        checklist_path = getattr(self, 'checklist_path', None)
        if not checklist_path or not osp.isfile(checklist_path):
            checklist_path, _ = ensure_capeval_data()
            checklist_path = str(checklist_path)
        units = prepare_units(captions, checklist_path)
        ans = load_cached_answers(tmp_file, storage)
        todo = pending_units(units, ans)
        if todo:
            model = build_judge(
                model=judge_model, temperature=0.0,
                system_prompt=SINGLE_PASS_SYSTEM, max_tokens=8192, **judge_kwargs,
            )
            assert model.working(), (
                'CAPEval evaluation requires a working judge API '
                f'(default --judge {self.DEFAULT_JUDGE}).\n' + DEBUG_MESSAGE
            )
            tups = [
                (model, build_single_pass_user_prompt(u['caption'], u['checklist_items']),
                 len(u['checklist_items']))
                for u in todo
            ]
            track_progress_rich(
                CAPEval_atomeval, tups, nproc=nproc, chunksize=nproc,
                keys=[u['image_id'] for u in todo], save=tmp_file,
            )
            ans = load_cached_answers(tmp_file, storage)

        records = assemble_records(units, ans)
        for rec in records:
            rec['judge_model'] = judge_model
            rec['prompt_version'] = PROMPT_VERSION
            rec['evaluator_version'] = EVALUATOR_VERSION
        dump(records, storage)
        ret = records_to_score_df(
            records, judge_model=judge_model,
            prompt_version=PROMPT_VERSION, evaluator_version=EVALUATOR_VERSION,
        )
        n_error = int(ret.iloc[0]['n_error'])
        n_ok = int(ret.iloc[0]['n_images'])
        if n_error:
            warnings.warn(
                f'CAPEval partial evaluation: {n_error} image(s) failed the judge; '
                f'C/P are aggregated over {n_ok} successful image(s) only. See {score_file}.'
            )
        dump(ret, score_file)
        return ret
