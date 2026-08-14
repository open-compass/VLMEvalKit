"""CAPEval evaluate via VLMEvalKit ``build_judge``, official checklist protocol."""

import os.path as osp
from pathlib import Path

import pandas as pd

from vlmeval.smp import load
from .schema import (CATEGORY_NAMES, aggregate_records, build_checklist_items_with_index,
                     domain_from_img_path, load_jsonl, normalize_verdict_list,
                     parse_single_pass_json, validate_unit)


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


def load_checklist_by_image(checklist_path):
    cl_by_img = {}
    for r in load_jsonl(checklist_path, checklist_rows_only=True):
        p = str(r.get('img_path') or '').strip()
        if not p:
            continue
        cl_by_img[p] = r
        cl_by_img[Path(p).name] = r
    return cl_by_img


def prepare_units(captions, checklist_path):
    cl_by_img = load_checklist_by_image(checklist_path)
    units = []
    skipped = 0
    for cap_key, caption in captions.items():
        name = Path(str(cap_key)).name
        cl_row = cl_by_img.get(name) or cl_by_img.get(str(cap_key))
        if cl_row is None:
            skipped += 1
            continue
        img_path = str(cl_row.get('img_path') or name).strip()
        items = build_checklist_items_with_index(cl_row)
        try:
            validate_unit(img_path, caption, items)
        except ValueError:
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
    """Merge incremental pkl + final judge json. Tmp keys win over storage."""
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
    """Skip successful cache hits; retry missing or ``status != ok`` records."""
    return [u for u in units if not is_ok_record(ans.get(u['image_id']))]


def CAPEval_atomeval(model, prompt, n_items):
    """One image: official user prompt → judge.generate → parse verdicts."""
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
        yes1=y,
        no1=n,
        not_mentioned1=nm,
        total1=n_items,
        gt_verdicts=[
            {
                'item_index': j,
                'verdict': parsed['verdicts'][j],
                'reasoning': reasons[j],
            }
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
