"""Checklist schema, verdict parsing, and C / P aggregation (official CAPEval protocol)."""

import json
import re
from collections import defaultdict
from pathlib import Path

# Bump this (and judge cache suffix) if C/P aggregation or verdict parsing changes.
EVALUATOR_VERSION = 'capeval-cp-v1'

CHECKLIST_KEYS = [
    'instance_checklist',
    'attribute_checklist',
    'relation_checklist',
    'image_checklist',
    'text_checklist',
    'human_checklist',
    'ui_checklist',
    'world_knowledge_checklist',
]

CATEGORY_NAMES = {
    'SO': 'Scene & Object',
    'PA': 'People & Activity',
    'TI': 'Text & Interface',
    'DK': 'Design & Knowledge',
}


def flatten_items(entry):
    items = []
    for ck in CHECKLIST_KEYS:
        for item in entry.get(ck, []) or []:
            if not isinstance(item, dict):
                continue
            q = str(item.get('Question', '')).strip()
            if q:
                items.append({
                    'checklist_type': ck,
                    'tags': item.get('Tags', ''),
                    'question': q,
                })
    return items


def build_checklist_items_with_index(entry):
    items = []
    for idx, it in enumerate(flatten_items(entry)):
        items.append({
            'item_index': idx,
            'checklist_type': it['checklist_type'],
            'tags': it.get('tags', '') or '',
            'tag': it.get('tags', '') or '',
            'question': it['question'],
        })
    return items


def split_raw_tag_field(tags):
    if not tags or not str(tags).strip():
        return []
    parts = re.split(r'[,，;；|]+', str(tags).strip())
    return [p.strip() for p in parts if p.strip()]


def domain_from_img_path(img_path):
    stem = Path(str(img_path)).stem
    prefix = []
    for ch in stem:
        if ch.isalpha():
            prefix.append(ch)
        else:
            break
    return ''.join(prefix).upper() if prefix else 'unknown'


def load_jsonl(path, checklist_rows_only=False):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.lstrip('\ufeff').strip()
    if not text:
        return []
    dec = json.JSONDecoder()
    rows = []
    idx = 0
    n = len(text)
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


def _strip_think(text):
    for sep in ('</think>', '</thinking>'):
        if sep in text:
            text = text.split(sep, 1)[-1]
    return text.strip()


# Exact tokens only (after light cleanup). Do not substring-match "yes" inside a sentence.
_YES_TOKENS = frozenset({'yes', 'y', 'correct', 'correctly_mentioned', 'b'})
_NO_TOKENS = frozenset({'no', 'n', 'incorrect', 'incorrectly_mentioned', 'c'})
_NM_TOKENS = frozenset({
    'not_mentioned', 'not mentioned', 'not-mentioned', 'notmentioned',
    'nm', 'a', 'unknown', 'n/a', 'na',
})


def canonicalize_verdict(raw):
    """Map a judge verdict token to ``yes`` / ``no`` / ``not_mentioned``.

    Normalizes case, surrounding quotes, a trailing period, and
    ``not mentioned`` / ``not_mentioned``. Returns ``None`` for free-form
    text such as ``I think this should be yes because...`` so callers must
    not treat that as ``yes``. Missing / invalid items are later filled as
    ``not_mentioned`` (official CAPEval behavior).
    """
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


def validate_unit(image_id, caption, checklist_items):
    """Raise ``ValueError`` if a scoring unit is missing required fields."""
    if image_id is None or str(image_id).strip() in ('', 'nan', 'None'):
        raise ValueError('image_id is required')
    if caption is None:
        raise ValueError(f'caption is required for image_id={image_id!r}')
    if not checklist_items:
        raise ValueError(f'checklist is empty for image_id={image_id!r}')
    return True


def parse_single_pass_json(raw, n_items):
    text = _strip_think(raw or '')
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
        if not isinstance(v, dict):
            continue
        idx = v.get('item_index')
        if idx is None:
            continue
        try:
            ii = int(idx)
        except (TypeError, ValueError):
            continue
        ver = canonicalize_verdict(v.get('verdict', ''))
        if ver is None:
            # Invalid token: do not guess "yes" from prose. Official fill is not_mentioned.
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
    """Official CAPEval: C = 100*(yes+no)/total, P = 100*yes/(yes+no).

    When ``yes + no == 0`` (nothing mentioned), P is 0.0 — same as the
    official scorer, which guards the divide-by-zero.
    """
    c = 100.0 * (yes + no) / total if total else 0.0
    mentioned = yes + no
    p = 100.0 * yes / mentioned if mentioned else 0.0
    return c, p


def _empty_bucket():
    return {
        'yes1': 0, 'no1': 0, 'not_mentioned1': 0, 'total1': 0,
        'n_images': 0, 'n_error': 0,
    }


def _summary_block(bucket):
    sy, sn, st = bucket['yes1'], bucket['no1'], bucket['total1']
    c, p = _cp_percent(sy, sn, st)
    out = {
        'C': round(c, 4),
        'P': round(p, 4),
        'yes1': sy,
        'no1': sn,
        'not_mentioned1': bucket['not_mentioned1'],
        'total1': st,
        'n_images': bucket['n_images'],
    }
    if bucket.get('n_error'):
        out['n_error'] = int(bucket['n_error'])
    return out


def aggregate_records(records):
    g = {
        **_empty_bucket(),
        'by_category': defaultdict(_empty_bucket),
    }
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
