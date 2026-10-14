"""MRareBench — a multimodal rare-disease benchmark over PMC case reports.

Two tracks, one class each. Every track ships several **dataset names** that share
one TSV, one ``evaluate()`` and one judge, and differ only in the *input condition*
the model is shown; the name selects a row of that class's ``CONDITIONS`` register.

T1 Diagnosis (``MRareBenchDiagnosis``) — ranked differential list.
  A from-scratch redesign that intentionally *departs* from the original
  single-primary-diagnosis logic in ``mmrarebench.py`` (the 3-dim weighted cascade
  ``S_T1`` + token F1, judged by Qwen3-VL-235B). Inspired by RareBench Table 4 /
  Task-4, the model outputs a **ranked list of the 10 most likely diagnoses**
  (most -> least likely), scored with:

  1. Deterministic Recall / Position metrics (the HARD primary metrics, no judge):
     Recall@{1,3,5,10}, MRR, MR (mean rank of the first hit, RareBench style),
     plus hit_rate and parse_fail_rate.
  2. Optional rare-candidate ratio@10, enabled only when a rare judge is
     explicitly provided. This measures rare-disease hypothesis tendency, not
     correctness.
  3. A complementary gpt-5.4-mini judge that rates 5 binary (YES/NO) dimensions and
     averages them into a soft score (cross-validates + adds clinical plausibility
     signal beyond keyword matching).

  Conditions: FD (findings text + images) / baseline (findings redacted, images) /
  TO (findings redacted, no image). FD->baseline isolates textual leakage;
  baseline->TO isolates visual necessity.

T2 Evidence Verification (``MRareBenchEvidenceVerif``) — backward evidence check.
  The diagnosis is GIVEN; the model must describe the visual evidence supporting
  it, scored by a per-item binary rubric with deterministic (judge-free) recall
  metrics alongside. Conditions: gold / TO (no image) / NoDx (label withheld).

Design rationale is documented in
``step2_evaluation_20260625/T1_DIAGNOSIS_V2_DESIGN.md`` and
``T4_EVIDENCE_VERIFICATION_DESIGN.md``.
"""
import json
import os
import os.path as osp
import re
import string
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..smp.file import LMUDataRoot, dump, get_intermediate_file_path, load
from ..smp.vlm import read_ok
from ..utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import build_judge

# --- configuration (tunable constants) ---------------------------------------
BENCH_ROOT = os.environ.get('MRAREBENCH_ROOT', 'MRareBench')
TRACK = 'diagnosis'
TSV_NAME = 'diagnosis_opened.tsv'
TOPK = 10                       # number of ranked diagnoses requested from the model
RECALL_KS = (1, 3, 5, 10)       # k values reported for Recall@k
MATCH_TOKEN_F1_THRESHOLD = 0.6  # token-set F1 above which a candidate counts as a hit
RARE_LABELS = {'rare', 'nonrare', 'generic', 'unknown'}


def parse_json_list(val):
    """Parse a TSV cell into a list (JSON list, or ';'-separated fallback)."""
    if isinstance(val, list):
        return val
    if pd.isna(val) or val == '':
        return []
    try:
        parsed = json.loads(val)
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except (json.JSONDecodeError, TypeError):
        return [s.strip() for s in str(val).split(';') if s.strip()]


def normalize_answer(answer: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles, fix spaces."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def token_set_f1(a: str, b: str) -> float:
    """Token-set F1 between two normalized strings."""
    ta = set(normalize_answer(a).split())
    tb = set(normalize_answer(b).split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    precision = inter / len(ta)
    recall = inter / len(tb)
    return 2 * precision * recall / (precision + recall)


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {'true', 'yes', 'y', '1'}:
            return True
        if val in {'false', 'no', 'n', '0'}:
            return False
    return None


def _extract_json_object(text):
    text = str(text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _canonical_rare_record(record, candidate, raw='', parse_failed=False):
    record = record if isinstance(record, dict) else {}
    label = str(record.get('label', 'unknown')).strip().lower().replace('-', '')
    if label not in RARE_LABELS:
        label = 'unknown'

    is_rare = _coerce_bool(record.get('is_rare_disease'))
    if is_rare is None:
        is_rare = label == 'rare'
    if label in {'nonrare', 'generic', 'unknown'}:
        is_rare = False

    is_specific = _coerce_bool(record.get('is_specific_disease'))
    if is_specific is None:
        is_specific = label in {'rare', 'nonrare'}
    if label == 'generic':
        is_specific = False

    orpha = record.get('orpha_code_if_known', None)
    if isinstance(orpha, str) and orpha.strip().lower() in {'', 'none', 'null', 'unknown'}:
        orpha = None

    cand = str(record.get('candidate') or candidate or '').strip()
    return {
        'candidate': cand,
        'normalized_candidate': normalize_answer(cand),
        'label': label,
        'is_rare_disease': bool(is_rare),
        'is_specific_disease': bool(is_specific),
        'orpha_code_if_known': orpha,
        'parse_failed': bool(parse_failed),
        'raw': str(raw),
    }


def _parse_rare_candidate_judge_response(response, candidate):
    parsed = _extract_json_object(response)
    if parsed is None:
        return _canonical_rare_record(
            {'candidate': candidate, 'label': 'unknown'},
            candidate,
            raw=response,
            parse_failed=True,
        )
    return _canonical_rare_record(parsed, candidate, raw=response, parse_failed=False)


def _load_rare_candidate_cache(cache_path):
    cache = {}
    if not cache_path or not osp.exists(cache_path):
        return cache
    with open(cache_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            candidate = item.get('candidate', '')
            rec = _canonical_rare_record(
                item,
                candidate,
                raw=item.get('raw', ''),
                parse_failed=item.get('parse_failed', False),
            )
            key = rec.get('normalized_candidate') or normalize_answer(candidate)
            if key:
                cache[key] = rec
    return cache


def _append_rare_candidate_cache(cache_path, records):
    if not cache_path or not records:
        return
    os.makedirs(osp.dirname(cache_path) or '.', exist_ok=True)
    with open(cache_path, 'a', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _build_rare_candidate_prompt(candidate):
    return (
        'You are a strict rare-disease ontology assistant. Classify ONE diagnosis '
        'candidate from a ranked differential diagnosis list.\n\n'
        f'Candidate diagnosis: {candidate}\n\n'
        'Return whether this candidate is a specific rare disease entity. '
        'Do not judge whether it is correct for a patient. Do not use the case '
        'context. Classify broad or non-specific terms such as "infection", '
        '"tumor", "autoimmune disease", "genetic disorder", or "syndrome" as '
        'generic unless they name a specific disease entity.\n\n'
        'Reply with valid JSON only, exactly using this schema:\n'
        '{\n'
        '  "candidate": "...",\n'
        '  "label": "rare | nonrare | generic | unknown",\n'
        '  "is_rare_disease": true,\n'
        '  "is_specific_disease": true,\n'
        '  "orpha_code_if_known": "ORPHA:xxxx or null"\n'
        '}\n'
        'Use label "unknown" if you cannot judge reliably.'
    )


def apply_rare_candidate_judge(data, ranked_lists, rare_judge_kwargs, cache_path=None):
    """Append model-judged rare-candidate metrics to ``data``.

    The judge is called once per unique normalized diagnosis candidate, then the
    cached candidate labels are mapped back to each row's top-10 list.
    """
    rare_judge_kwargs = dict(rare_judge_kwargs or {})
    rare_judge_model = rare_judge_kwargs.get('model')
    if not rare_judge_model or rare_judge_model == 'exact_matching':
        return data, {}

    nproc = int(rare_judge_kwargs.pop('nproc', 4))
    batch_size = int(
        rare_judge_kwargs.pop(
            'batch_size',
            os.environ.get('T1V2_RARE_BATCH_SIZE', 100),
        )
    )
    cache = _load_rare_candidate_cache(cache_path)

    unique_candidates = {}
    for ranked in ranked_lists:
        for cand in ranked[:TOPK]:
            key = normalize_answer(cand)
            if key and key not in unique_candidates:
                unique_candidates[key] = str(cand).strip()

    missing = [(key, cand) for key, cand in unique_candidates.items() if key not in cache]
    if missing:
        judge_model = build_judge(**rare_judge_kwargs)
        if not judge_model.working():
            warnings.warn('Rare-candidate judge API unavailable; rare metrics were not added.')
            return data, {}

        def _rare_call(model, candidate):
            try:
                raw = model.generate(_build_rare_candidate_prompt(candidate))
                return _parse_rare_candidate_judge_response(raw, candidate)
            except Exception as err:
                return _canonical_rare_record(
                    {'candidate': candidate, 'label': 'unknown'},
                    candidate,
                    raw=f'RARE_JUDGE_ERROR: {err}',
                    parse_failed=True,
                )

        if batch_size <= 0:
            batch_size = len(missing)
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            end = start + len(batch)
            print(
                f'[Rare Judge] candidates {start + 1}-{end}/{len(missing)} '
                f'(nproc={nproc}, cache={cache_path or ""})',
                flush=True,
            )
            tups = [dict(model=judge_model, candidate=cand) for _, cand in batch]
            out = track_progress_rich(_rare_call, tups, nproc=nproc,
                                      chunksize=nproc,
                                      description='[Rare Judge] Diagnosis v2')
            new_records = []
            for (key, _), rec in zip(batch, out):
                rec = _canonical_rare_record(rec, rec.get('candidate', ''),
                                             rec.get('raw', ''),
                                             rec.get('parse_failed', False))
                cache[key] = rec
                new_records.append(rec)
            _append_rare_candidate_cache(cache_path, new_records)
            print(
                f'[Rare Judge] cached {len(new_records)} candidates; '
                f'done {end}/{len(missing)}',
                flush=True,
            )

    flags_rows, count_rows, ratio_rows = [], [], []
    unknown_rows, generic_rows, raw_rows = [], [], []
    used_parse_fail = []
    for ranked in ranked_lists:
        flags, raw_records = [], []
        unknown_count, generic_count = 0, 0
        for cand in ranked[:TOPK]:
            key = normalize_answer(cand)
            rec = cache.get(key)
            if rec is None:
                rec = _canonical_rare_record({'candidate': cand, 'label': 'unknown'}, cand,
                                             raw='CACHE_MISS', parse_failed=True)
            flags.append(1 if rec.get('is_rare_disease') else 0)
            if rec.get('label') == 'unknown':
                unknown_count += 1
            if rec.get('label') == 'generic':
                generic_count += 1
            used_parse_fail.append(1 if rec.get('parse_failed') else 0)
            raw_records.append(rec)
        rare_count = int(sum(flags))
        flags_rows.append(json.dumps(flags, ensure_ascii=False))
        count_rows.append(rare_count)
        ratio_rows.append(float(rare_count / TOPK))
        unknown_rows.append(int(unknown_count))
        generic_rows.append(int(generic_count))
        raw_rows.append(json.dumps(raw_records, ensure_ascii=False))

    data['rare_candidate_flags@10'] = flags_rows
    data['rare_candidate_count@10'] = count_rows
    data['rare_candidate_ratio@10'] = ratio_rows
    data['rare_candidate_unknown_count@10'] = unknown_rows
    data['rare_candidate_generic_count@10'] = generic_rows
    data['rare_candidate_judge_raw'] = raw_rows

    summary = {
        'rare_candidate_count@10_mean': float(np.mean(count_rows)) if count_rows else 0.0,
        'rare_candidate_ratio@10': float(np.mean(ratio_rows)) if ratio_rows else 0.0,
        'rare_candidate_unknown_count@10_mean': float(np.mean(unknown_rows)) if unknown_rows else 0.0,
        'rare_candidate_generic_count@10_mean': float(np.mean(generic_rows)) if generic_rows else 0.0,
        'rare_candidate_judge_parse_fail_rate': (
            float(np.mean(used_parse_fail)) if used_parse_fail else 0.0
        ),
        'rare_candidate_unique_total': int(len(unique_candidates)),
        'rare_candidate_cache_path': str(cache_path or ''),
    }
    return data, summary


def is_hit(candidate: str, gold_list) -> bool:
    """A candidate disease counts as a hit against the gold set when, after SQuAD
    normalization, it (a) exactly equals a gold/alias, (b) is a bidirectional
    substring of one, or (c) has token-set F1 >= MATCH_TOKEN_F1_THRESHOLD."""
    nc = normalize_answer(candidate)
    if not nc:
        return False
    for g in gold_list:
        ng = normalize_answer(g)
        if not ng:
            continue
        if nc == ng:
            return True
        if ng in nc or nc in ng:
            return True
        if token_set_f1(candidate, g) >= MATCH_TOKEN_F1_THRESHOLD:
            return True
    return False


def parse_ranked_list(prediction, topk=TOPK):
    """Parse a model response into an ordered list of diagnosis strings.

    Primary: numbered lines like ``1. xxx`` / ``2) yyy``. Fallback: split on
    newlines/semicolons. Returns up to ``topk`` items (rank 1 first)."""
    text = str(prediction)
    # Reasoning models (e.g. UniMedVL) wrap chain-of-thought in <think>...</think>
    # and emit the final differential only afterwards; strip it so the numbered
    # reasoning paths inside are not mistaken for the ranked diagnosis list.
    had_think = '<think>' in text.lower()
    text = re.sub(r'(?is)<think>.*?</think>', '', text).strip()
    items = []
    for ln in text.splitlines():
        m = re.match(r'^\s*(\d+)[\.\)]\s*(.+?)\s*$', ln)
        if m:
            items.append(m.group(2).strip())
    # Fallback for a single-line "1. a 2. b 3. c" differential (numbered items
    # inline, no newlines); only fires when per-line parsing under-segments,
    # so clean multi-line lists are left untouched. Gated on had_think so no
    # clean (non-reasoning) model's existing parse can ever change.
    if had_think and len(items) < 2:
        inline = [s.strip(' -*\t\n') for s in re.split(r'(?:^|\s)\d{1,2}[\.\)]\s+', text) if s.strip()]
        if len(inline) > len(items):
            items = inline
    if not items:
        items = [s.strip(' -*\t') for s in re.split(r'[\n;]+', text) if s.strip()]
    # drop empties, dedup preserving order
    seen = set()
    cleaned = []
    for it in items:
        if it and it.lower() not in seen:
            seen.add(it.lower())
            cleaned.append(it)
    return cleaned[:topk]


# All three T1 conditions read this one file; they differ only in what the model
# is shown. Local-first: place the finalized TSV at LMUData/MRareBench/diagnosis/.
# HF fallback retained for portability.
T1_TSV_URL = (
    'https://huggingface.co/datasets/junzhin/MRareBench/resolve/main/'
    'diagnosis/diagnosis_opened.tsv'
)


class MRareBenchDiagnosis(ImageBaseDataset):
    """T1 Diagnosis (v2): ranked top-10 differential, Recall/Position + multi-dim judge.

    One class serves all three T1 input conditions. ``build_dataset()`` passes the
    requested name through to ``self.dataset_name``, which selects a row of
    ``CONDITIONS``; the TSV, the metrics and the judge are shared.
    """

    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    # ----- input-condition register (visual-necessity ablation) ----------------
    #   FD (Findings-Disclosed) : context + findings text + images
    #   LC (baseline)           : context (findings redacted) + images
    #   TO (Text-Only)          : context (findings redacted), NO image
    # FD->LC isolates textual leakage; LC->TO isolates visual necessity.
    #
    # `basis` is stored verbatim per condition instead of being derived from
    # `attach_images`: re-wording one condition's instruction must never be able
    # to silently re-word another's. Do not collapse it into a conditional.
    CONDITIONS = {
        'MRareBench_Diagnosis': dict(
            inject_findings=False, attach_images=True,
            basis='the clinical presentation and the medical images'),
        'MRareBench_Diagnosis_FD': dict(
            inject_findings=True, attach_images=True,
            basis='the clinical presentation and the medical images'),
        'MRareBench_Diagnosis_TO': dict(
            inject_findings=False, attach_images=False,
            basis='the clinical presentation'),
    }

    # Baseline first, so a caller taking supported_datasets()[0] gets the
    # reference condition rather than an arbitrary variant.
    DATASET_URL = {name: T1_TSV_URL for name in CONDITIONS}
    DATASET_MD5 = {}

    @property
    def condition(self):
        """The registered input condition for this instance's dataset name."""
        cond = self.CONDITIONS.get(self.dataset_name)
        assert cond is not None, (
            f'{self.dataset_name!r} is not a registered MRareBench T1 input '
            f'condition; expected one of {list(self.CONDITIONS)}'
        )
        return cond

    # ----- data loading (local-first) -----------------------------------------
    def load_data(self, dataset):
        local_path = osp.join(LMUDataRoot(), BENCH_ROOT, TRACK, TSV_NAME)
        if osp.exists(local_path):
            data = pd.read_csv(local_path, sep='\t', encoding='utf-8')
            if 'index' not in data.columns:
                data['index'] = range(len(data))
            data['index'] = [str(x) for x in data['index']]
            self.data_path = local_path
            return data
        url = self.DATASET_URL.get(dataset, None)
        if url:
            return self.prepare_tsv(url, None)
        raise FileNotFoundError(
            f'MRareBench v2 TSV not found at {local_path} and no usable DATASET_URL.'
        )

    # ----- image handling (base64 first, file path fallback) -------------------
    def dump_image(self, line):
        from ..smp import decode_base64_to_image_file

        image_b64_list = parse_json_list(line.get('image', '[]'))
        if image_b64_list and any(image_b64_list):
            os.makedirs(self.img_root, exist_ok=True)
            tgt_paths = []
            for i, b64_data in enumerate(image_b64_list):
                if not b64_data:
                    continue
                idx = str(line.get('index', i))
                tgt_path = osp.join(self.img_root, f"{idx}--{i + 1}.jpg")
                if not read_ok(tgt_path):
                    try:
                        decode_base64_to_image_file(b64_data, tgt_path)
                    except Exception as e:
                        warnings.warn(f'Failed to decode base64 image: {e}')
                        continue
                tgt_paths.append(tgt_path)
            if tgt_paths:
                return tgt_paths

        image_list = parse_json_list(line.get('image_path', '[]'))
        if not image_list:
            return []
        track_dir = osp.join(LMUDataRoot(), BENCH_ROOT, TRACK)
        abs_paths = []
        for rel_path in image_list:
            abs_paths.append(rel_path if osp.isabs(rel_path) else osp.join(track_dir, rel_path))
        for p in abs_paths:
            if not read_ok(p):
                warnings.warn(f'Image file not found: {p}')
        return abs_paths

    @staticmethod
    def _render_findings(line):
        """Parse the 'image findings' JSON array and
        return a markdown block of the findings, or '' if missing/empty."""
        raw = line.get('image findings', '')
        if not (pd.notna(raw) and str(raw).strip()):
            return ''
        try:
            entries = json.loads(str(raw))
        except (json.JSONDecodeError, TypeError):
            return ''
        if not isinstance(entries, list):
            return ''
        blocks, seen = [], set()
        for e in entries:
            if not isinstance(e, dict):
                continue
            # each entry's 'markdown' already carries its own '### Title' header;
            # fall back to title + body if markdown is absent.
            md = str(e.get('markdown', '') or '').strip()
            if not md:
                title = str(e.get('title', '') or '').strip()
                body = str(e.get('body', '') or '').strip()
                md = (f'### {title}\n{body}' if title else body).strip()
            if md and md not in seen:
                seen.add(md)
                blocks.append(md)
        return '\n\n'.join(blocks)

    # ----- prompt: ask for a ranked top-10 differential ------------------------
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        cond = self.condition

        tgt_path = self.dump_image(line) if cond['attach_images'] else []
        question = str(line.get('question', '')) if pd.notna(line.get('question', '')) else ''
        context = str(line.get('context', '')) if pd.notna(line.get('context', '')) else ''

        if cond['inject_findings']:
            # FD: re-inject the imaging-findings text the baseline redacts.
            findings = self._render_findings(line)
            if findings:
                context = (context + '\n\n' + findings).strip() if context else findings

        # figure_mapping -> per-image label (kept from v1 to preserve image grounding)
        basename_to_label = {}
        if cond['attach_images']:
            fig_map_raw = line.get('figure_mapping', '')
            if pd.notna(fig_map_raw) and str(fig_map_raw).strip():
                try:
                    fm = json.loads(str(fig_map_raw))
                    if isinstance(fm, dict):
                        for label, rel_path in fm.items():
                            basename_to_label[osp.basename(rel_path)] = label
                except (json.JSONDecodeError, TypeError):
                    pass

        instruction = (
            f'You are an expert physician. Based on {cond["basis"]}, list the {TOPK} '
            f'most likely diagnoses, ranked from MOST likely (1) to LEAST likely '
            f'({TOPK}). '
            f'Output ONLY a numbered list, one standard clinical disease name per line '
            f'(e.g. "1. Disease name"). Do not include explanations, reasoning, or extra text.'
        )

        prompt_parts = []
        if context:
            prompt_parts.append(context)
        if question:
            prompt_parts.append(f'Question: {question}')
        prompt_parts.append(instruction)
        prompt = '\n\n'.join(prompt_parts)

        msgs = []
        for p in tgt_path:
            label = basename_to_label.get(osp.basename(p), '')
            if label:
                msgs.append(dict(type='text', value=f'{label}:'))
            msgs.append(dict(type='image', value=p))
        msgs.append(dict(type='text', value=prompt))
        return msgs

    # ----- evaluation ----------------------------------------------------------
    def evaluate(self, eval_file, **judge_kwargs):
        judge_kwargs = dict(judge_kwargs)
        rare_judge_model = (
            judge_kwargs.pop('rare_judge_model', None)
            or judge_kwargs.pop('rare_model', None)
        )
        rare_judge_cache = judge_kwargs.pop('rare_judge_cache', None)
        rare_judge_nproc = judge_kwargs.pop('rare_judge_nproc', None)
        rare_judge_api_base = judge_kwargs.pop('rare_judge_api_base', None)
        rare_judge_key = judge_kwargs.pop('rare_judge_key', None)

        data = load(eval_file)
        assert 'prediction' in data.columns, 'Missing prediction column'
        data['prediction'] = [str(x) for x in data['prediction']]
        meta = self.data

        # 1) parse ranked lists + gold sets ------------------------------------
        ranked_lists, gold_sets = [], []
        first_hit_ranks, parse_fail = [], []
        for _, row in data.iterrows():
            idx = str(row['index'])
            meta_row = meta[meta['index'].astype(str) == idx]
            meta_row = meta_row.iloc[0] if len(meta_row) else row

            gold = [str(meta_row.get('answer', '')).strip()]
            for a in parse_json_list(meta_row.get('answer_aliases', '')):
                if str(a).strip():
                    gold.append(str(a).strip())
            gold = [g for g in gold if g]
            gold_sets.append(gold)

            ranked = parse_ranked_list(row['prediction'], TOPK)
            ranked_lists.append(ranked)
            parse_fail.append(1 if len(ranked) == 0 else 0)

            fhr = None
            for r, cand in enumerate(ranked, start=1):
                if is_hit(cand, gold):
                    fhr = r
                    break
            first_hit_ranks.append(fhr)

        # 2) deterministic Recall / Position metrics (HARD primary) ------------
        n = len(data)
        hits = [r is not None for r in first_hit_ranks]
        recall_at = {k: float(np.mean([1.0 if (r is not None and r <= k) else 0.0
                                       for r in first_hit_ranks])) for k in RECALL_KS}
        mrr = float(np.mean([1.0 / r if r is not None else 0.0 for r in first_hit_ranks]))
        hit_ranks = [r for r in first_hit_ranks if r is not None]
        # RareBench-style mean rank of the first hit (lower is better)
        mr = float(np.mean(hit_ranks)) if hit_ranks else 0.0

        data['ranked_list'] = [json.dumps(rl, ensure_ascii=False) for rl in ranked_lists]
        data['first_hit_rank'] = [r if r is not None else -1 for r in first_hit_ranks]
        data['hit'] = [int(h) for h in hits]

        results = {
            'n_total': n,
            'n_hit': int(sum(hits)),
            'hit_rate': float(np.mean(hits)) if n else 0.0,
            'parse_fail_rate': float(np.mean(parse_fail)) if n else 0.0,
            'MRR': mrr,
            'MR': mr,
        }
        for k in RECALL_KS:
            results[f'recall@{k}'] = recall_at[k]

        # 3) optional gpt-5.4-mini multi-binary-dim judge (soft, complementary) --
        standard_judge_kwargs = dict(judge_kwargs)
        judge_name = standard_judge_kwargs.get('model', None)
        if judge_name and judge_name != 'exact_matching':
            judge_model = build_judge(**standard_judge_kwargs)
            if judge_model.working():
                nproc = standard_judge_kwargs.pop('nproc', 4)
                prompts = [self._build_judge_prompt(rl, gs)
                           for rl, gs in zip(ranked_lists, gold_sets)]

                def _judge_call(model, prompt):
                    try:
                        raw = model.generate(prompt)
                        return dict(dims=self._parse_judge_dims(raw), raw=str(raw))
                    except Exception as err:
                        return dict(dims=[0, 0, 0, 0, 0], raw=f'JUDGE_ERROR: {err}')

                tups = [dict(model=judge_model, prompt=p) for p in prompts]
                out = track_progress_rich(_judge_call, tups, nproc=nproc, chunksize=nproc,
                                          description='[Judge] Diagnosis v2')
                dim_mat = np.array([o['dims'] if isinstance(o, dict) else [0] * 5 for o in out],
                                   dtype=float)
                # TYPE_MATCH is ordinal 0-3; others are binary 0/1. Normalize
                # TYPE_MATCH to [0,1] ONLY for the summary judge_score mean; the
                # per-row column keeps the raw 0-3 value (full granularity).
                dim_names = ['presence', 'top1', 'typematch', 'specificity', 'coherence']
                norm_mat = dim_mat.copy()
                norm_mat[:, 2] = norm_mat[:, 2] / 3.0
                per_item = norm_mat.mean(axis=1)
                data['judge_score'] = per_item.tolist()
                # persist every dimension per-row (no condensing to one number)
                for j, dn in enumerate(dim_names):
                    data[f'judge_{dn}'] = dim_mat[:, j].tolist()
                data['judge_raw'] = [o.get('raw', '') if isinstance(o, dict) else ''
                                     for o in out]
                results['judge_avg_score'] = float(per_item.mean()) if n else 0.0
                for j, dn in enumerate(dim_names):
                    results[f'judge_dim_{dn}'] = float(dim_mat[:, j].mean()) if n else 0.0
            else:
                warnings.warn('Judge API unavailable; reporting deterministic metrics only.')
        else:
            warnings.warn('No --judge specified; reporting deterministic Recall/Position only.')

        # 4) optional rare-disease hypothesis-space judge ----------------------
        if rare_judge_model:
            if rare_judge_cache is None:
                rare_judge_cache = get_intermediate_file_path(
                    eval_file, '_rare_candidate_cache', 'jsonl')
            rare_judge_kwargs = dict(judge_kwargs)
            rare_judge_kwargs['model'] = rare_judge_model
            if rare_judge_nproc is not None:
                rare_judge_kwargs['nproc'] = rare_judge_nproc
            if rare_judge_api_base:
                rare_judge_kwargs['api_base'] = rare_judge_api_base
            if rare_judge_key:
                rare_judge_kwargs['key'] = rare_judge_key
            data, rare_summary = apply_rare_candidate_judge(
                data, ranked_lists, rare_judge_kwargs, cache_path=rare_judge_cache)
            results.update(rare_summary)

        # 5) persist -----------------------------------------------------------
        score_file = get_intermediate_file_path(eval_file, '_detailed_metrics', 'json')
        dump(results, score_file)
        per_row_file = get_intermediate_file_path(eval_file, '_per_row', 'xlsx')
        keep_cols = [c for c in ['index', 'answer', 'prediction', 'ranked_list',
                                 'first_hit_rank', 'hit',
                                 'judge_presence', 'judge_top1', 'judge_typematch',
                                 'judge_specificity', 'judge_coherence',
                                 'judge_score', 'judge_raw',
                                 'rare_candidate_flags@10',
                                 'rare_candidate_count@10',
                                 'rare_candidate_ratio@10',
                                 'rare_candidate_unknown_count@10',
                                 'rare_candidate_generic_count@10',
                                 'rare_candidate_judge_raw'] if c in data.columns]
        dump(data[keep_cols], per_row_file)
        return results

    # ----- judge prompt: 4 binary dims + 1 ordinal TYPE_MATCH (0-3) -------------
    def _build_judge_prompt(self, ranked_list, gold_list):
        gold = gold_list[0] if gold_list else ''
        aliases = '; '.join(gold_list[1:]) if len(gold_list) > 1 else '(none)'
        numbered = '\n'.join(f'{i}. {d}' for i, d in enumerate(ranked_list, start=1)) or '(empty)'
        return (
            'You are an extremely strict attending-physician evaluator. '
            'For YES/NO items, default to NO and answer YES only when a criterion '
            'is completely and explicitly satisfied.\n\n'
            f'Correct diagnosis (gold): {gold}\n'
            f'Accepted synonyms/abbreviations: {aliases}\n\n'
            f'The model produced this ranked differential (1 = most likely):\n{numbered}\n\n'
            'Score each of the following, one answer per line, in order:\n'
            '1. PRESENCE: Does the list contain the correct diagnosis or an '
            'accepted synonym anywhere? (YES/NO)\n'
            '2. TOP1: Is the #1 (most likely) item the correct diagnosis or an '
            'accepted synonym? (YES/NO)\n'
            '3. TYPE_MATCH: Judge the disease-type proximity between the BEST item '
            'in the list and the gold diagnosis, on a 0-3 scale:\n'
            '   3 = the gold itself, an accepted synonym, or an explicit subtype of it;\n'
            '   2 = a different subtype of the same disease, or its direct parent '
            'entity (highly related disease type);\n'
            '   1 = the same disease family or same underlying mechanism '
            '(related but not specific);\n'
            '   0 = unrelated.\n'
            '4. SPECIFICITY: Are the listed items specific disease entities rather '
            'than vague terms like "infection", "tumor", or "autoimmune disease"? (YES/NO)\n'
            '5. COHERENCE: Is the differential clinically plausible for this '
            'presentation and ordered reasonably (more likely diseases ranked '
            'higher)? (YES/NO)\n\n'
            'Reply with ONLY five lines, one answer per line, in this exact order: '
            'YES/NO, YES/NO, 0-3 integer, YES/NO, YES/NO.'
        )

    def _parse_judge_dims(self, response):
        """Return [presence, top1, typematch, specificity, coherence].

        presence/top1/specificity/coherence are binary 0/1; typematch is an
        ordinal integer 0-3. Parses line-by-line to keep position alignment."""
        lines = [ln.strip() for ln in str(response).strip().splitlines() if ln.strip()]

        def _binary(s):
            m = re.search(r'\b(yes|no)\b', s.lower())
            return 1 if (m and m.group(1) == 'yes') else 0

        def _ordinal(s):
            # drop a leading line-number prefix like "3." / "3)" so it is not
            # mistaken for the score value
            s = re.sub(r'^\s*\d+\s*[\.\)]\s*', '', s)
            m = re.search(r'[0-3]', s)
            return int(m.group(0)) if m else 0

        # take the first 5 informative lines
        vals = lines[:5]
        while len(vals) < 5:
            vals.append('')
        return [_binary(vals[0]), _binary(vals[1]), _ordinal(vals[2]),
                _binary(vals[3]), _binary(vals[4])]


# =============================================================================
# T2 — Evidence Verification
#
# Task: the final diagnosis is GIVEN; the model must identify/verify the visual
# evidence in the figure(s) that supports it (evidence attribution, NOT
# re-diagnosis). T2 is the evidence-grounding companion to T1 — see the METRIC
# VALIDITY ADDENDUM 10.6 (T1-T2 paired grounding) for the headline research use.
#
#   (open, SECONDARY) : free-text per-image evidence description, scored by a
#                       per-item BINARY LLM rubric over `acceptable_evidence_points`
#                       (required-point recall + optional coverage + faithfulness).
#                       Binary-only by design (the ADDENDUM shows ordinal judge
#                       dims are not reproducible; binary dims are).
#
# Every metric is reported separately (no composite score). Per-row outputs carry
# all join identifiers (row_key, pmcid, doc_id, task_type, t1_* keys) so the T1-T2
# paired quadrant analysis can be computed offline with NO model/judge re-run.
# =============================================================================
T2_TRACK = 'evidence_verification'
# Single source of truth (immutable, 608 evaluable rows = 609 pipeline - 1 unrecoverable MCQ).
# 2026-07-05: bumped ver1 314 → ver3 609 delivered (base 372 + expansion 237, post slot-repair).
#             Evaluable headline count is 608: 1 MCQ (t4ev_seed:190) is unrecoverable and dropped,
#             so this FULL609 file physically holds 608 rows. Report T2 as 608 everywhere.
T2_FULL_TSV_NAME = 'evidence_verification_opened_FULL609.tsv'
# All three T2 conditions read this one file; they differ only in what is shown.
T2_TSV_URL = (
    'https://huggingface.co/datasets/junzhin/MRareBench/resolve/main/'
    'evidence_verification/evidence_verification_opened_FULL609.tsv'
)

# per-row identifier columns carried into every T2 per-row output (Part D enabler)
T2_ID_COLS = (
    'index', 'row_key', 'pmcid', 'doc_id', 'case_id', 'task_type',
    'final_diagnosis', 'orpha_code', 'overlap_with_T1', 't1_index',
    't1_ver2_paired_subset', 't1_ver2_match_key_pmcid', 't1_ver2_matched_index',
    'supporting_image_ids', 'decisive_image_ids',
)

T2_META_COLS = (
    'evidence_modality', 'image_dependency', 'diagnostic_relevance',
    'cross_image_relation', 'source_support', 'text_only_solvability',
    'mcq_quality',
)


# ----- deterministic open-ended metrics (no judge, reproducible) --------------
# These score the open-ended evidence description without any API call, so they
# can be recomputed any number of times. They are paraphrase-robust (semantic
# similarity, not string F1). Backend: a cached sentence-transformer by default
# (set T2_EMBED_MODEL to override; 'tfidf' to force the offline TF-IDF fallback).
_SIM_BACKEND = {}
DET_TAUS = (0.3, 0.35, 0.4, 0.5, 0.6)  # robustness bands (T2 design §9) + headline tau
# 0.50 chosen by judge calibration (40 items / 224 required points, gpt-5.4-mini):
# at tau=0.50 the MiniLM recall matches the strict judge recall (gap ~0.05), curing
# the low-tau optimism. S-PubMedBert-MS-MARCO was rejected — its cosine space is
# anisotropic (all pairs 0.82-0.92, non-discriminative). Must stay in DET_TAUS.
DET_HEADLINE_TAU = 0.50

T2_HIER_LEVEL = {
    'modality': 1,
    'macro_pattern': 1,
    'micro_pattern': 1,
    'other': 1,
    'cross_image_integration': 2,
    'diagnostic_attribution': 3,
}
T2_HIER_WEIGHT = {
    'modality': 0.5,
    'macro_pattern': 1.0,
    'micro_pattern': 1.2,
    'other': 1.0,
    # Core T2 skill: relate visible evidence across images/modalities.
    'cross_image_integration': 2.0,
    # Diagnosis is given in T2, so attribution is a light consistency check.
    'diagnostic_attribution': 0.75,
}
T2_VISUAL_CATS = {'modality', 'macro_pattern', 'micro_pattern', 'other'}
T2_CROSS_CATS = {'cross_image_integration'}


def _all_hit(vals):
    vals = list(vals)
    return 1.0 if (not vals or all(int(v) == 1 for v in vals)) else 0.0


def _hier_required_score(req_vec, req_cats):
    """Hard hierarchical T2 score from judge point hits.

    Later-level credit depends on earlier-level visual evidence:
      L1 visible findings -> L2 cross-image integration -> L3 diagnosis attribution.
    This prevents diagnosis-conditioned guesses from receiving full high-level credit
    when the model did not establish the prerequisite visual evidence.
    """
    hits = [int(v) for v in req_vec]
    cats = [str(c) for c in req_cats]
    if not hits or len(hits) != len(cats):
        return None

    weights = [T2_HIER_WEIGHT.get(c, 1.0) for c in cats]
    visual_hits = [h for h, c in zip(hits, cats) if c in T2_VISUAL_CATS]
    cross_hits = [h for h, c in zip(hits, cats) if c in T2_CROSS_CATS]
    visual_gate = _all_hit(visual_hits)
    cross_gate = _all_hit(cross_hits)

    raw_by_level = {1: [], 2: [], 3: []}
    gated_by_level = {1: [], 2: [], 3: []}
    gated_weighted = []
    for h, c, w in zip(hits, cats, weights):
        lv = min(T2_HIER_LEVEL.get(c, 1), 3)
        if lv <= 1:
            gate = 1.0
        elif lv == 2:
            gate = visual_gate
        else:
            gate = visual_gate * cross_gate
        gh = float(h) * gate
        gated_weighted.append(gh * w)
        raw_by_level[lv].append(float(h))
        gated_by_level[lv].append(gh)

    def _mean(vals):
        return float(np.mean(vals)) if vals else None

    return {
        'hierarchical_required_recall': float(sum(gated_weighted) / sum(weights)),
        'hier_visual_gate': visual_gate,
        'hier_cross_gate': cross_gate,
        'hier_level1_visual_recall': _mean(raw_by_level[1]),
        'hier_level2_cross_recall_raw': _mean(raw_by_level[2]),
        'hier_level2_cross_recall_gated': _mean(gated_by_level[2]),
        'hier_level3_diag_recall_raw': _mean(raw_by_level[3]),
        'hier_level3_diag_recall_gated': _mean(gated_by_level[3]),
    }


def _get_sim_backend():
    """Return (sim_fn, name). sim_fn(list_a, list_b) -> np.ndarray [a x b] cosine."""
    if 'fn' in _SIM_BACKEND:
        return _SIM_BACKEND['fn'], _SIM_BACKEND['name']
    model_name = os.environ.get('T2_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2').strip()
    if model_name and model_name.lower() not in ('none', 'tfidf', 'off'):
        try:
            # the default model is expected to be cached; force offline load so a
            # broken mirror cannot break eval. A user-supplied model may download.
            if not os.environ.get('T2_EMBED_MODEL'):
                os.environ.setdefault('HF_HUB_OFFLINE', '1')
                os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
            from sentence_transformers import SentenceTransformer
            mdl = SentenceTransformer(model_name, device='cpu')

            def fn(a, b):
                a, b = list(a), list(b)
                if not a or not b:
                    return np.zeros((len(a), len(b)))
                ea = mdl.encode(a, normalize_embeddings=True, show_progress_bar=False)
                eb = mdl.encode(b, normalize_embeddings=True, show_progress_bar=False)
                return np.asarray(ea) @ np.asarray(eb).T
            _SIM_BACKEND.update(fn=fn, name=f'sbert:{model_name}')
            return fn, _SIM_BACKEND['name']
        except Exception as e:
            warnings.warn(f'T2 embedding backend unavailable ({str(e)[:80]}); using TF-IDF.')
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def fn(a, b):
        a, b = list(a), list(b)
        if not a or not b:
            return np.zeros((len(a), len(b)))
        corpus = [str(s) for s in a + b]
        if not any(s.strip() for s in corpus):
            return np.zeros((len(a), len(b)))
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit(corpus)
        return cosine_similarity(vec.transform([str(s) for s in a]),
                                 vec.transform([str(s) for s in b]))
    _SIM_BACKEND.update(fn=fn, name='tfidf')
    return fn, _SIM_BACKEND['name']


_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')


def _split_sentences(text):
    out = []
    for p in _SENT_SPLIT_RE.split(str(text)):
        p = re.sub(r'^[\s\-\*#>•\d\.\)]+', '', p).strip()
        if len(p) >= 3:
            out.append(p)
    return out


def _image_token(point):
    """Number from a leading '[Image N]' label, else None."""
    m = re.match(r'\s*\[\s*image\s*(\d+)', str(point), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _det_open_metrics(answer, required, optional, rationale, taus=DET_TAUS):
    """Deterministic, reproducible open-ended metrics (no judge)."""
    sim_fn, backend = _get_sim_backend()
    sents = _split_sentences(answer)
    res = {'det_backend': backend, 'det_n_sentences': len(sents)}

    def maxsims(points):
        if not points or not sents:
            return []
        return [float(x) for x in np.asarray(sim_fn(points, sents)).max(axis=1)]

    rq, op = maxsims(required), maxsims(optional)
    res['det_required_maxsims'] = rq
    res['det_optional_maxsims'] = op
    res['det_required_meanmaxsim'] = float(np.mean(rq)) if rq else None
    res['det_optional_meanmaxsim'] = float(np.mean(op)) if op else None
    for t in taus:
        if rq:
            res[f'det_required_recall@{t}'] = float(np.mean([1.0 if s >= t else 0.0 for s in rq]))
        if op:
            res[f'det_optional_recall@{t}'] = float(np.mean([1.0 if s >= t else 0.0 for s in op]))

    # grounded-sentence ratio = deterministic faithfulness/precision proxy:
    # fraction of the model's sentences that are semantically close to SOME gold
    # evidence. Low ratio => content unrelated to the gold (padding/fabrication).
    gold = list(required) + list(optional) + _split_sentences(rationale)
    if sents and gold:
        sent_max = [float(x) for x in np.asarray(sim_fn(sents, gold)).max(axis=1)]
        res['det_sent_groundedness'] = sent_max
        res['det_mean_sent_groundedness'] = float(np.mean(sent_max))
        for t in taus:
            res[f'det_grounded_ratio@{t}'] = float(np.mean([1.0 if s >= t else 0.0 for s in sent_max]))

    # image-grounded required recall: a required point tied to 'Image k' counts
    # only if its content matches a sentence that actually mentions Image k
    # (tests correct attribution; less leaked by the context's image list).
    img_hits = []
    for p in required:
        k = _image_token(p)
        if k is None:
            continue
        cand = [s for s in sents if re.search(rf'\bimage\s*{k}\b', s, re.IGNORECASE)]
        img_hits.append(float(np.asarray(sim_fn([p], cand)).max()) if cand else 0.0)
    res['det_image_grounded_maxsims'] = img_hits
    if img_hits:
        res['det_image_grounded_meanmaxsim'] = float(np.mean(img_hits))
        for t in taus:
            res[f'det_image_grounded_recall@{t}'] = float(
                np.mean([1.0 if s >= t else 0.0 for s in img_hits]))
    return res


# ----- deterministic image-attribution / grounding metrics (no judge) ---------
# These test WHICH images the model tied its evidence to, purely by parsing the
# 'Image k' tokens the open-ended prompt forces the model to emit, then comparing
# that set to the gold decisive/supporting image ids. No API: it is string/set
# math over the already-generated answer. IMPORTANT (construct-validity caveat):
# these presume the 'Image k:' text label reliably binds to the k-th attached
# image (interleaved order is preserved by the API wrappers, INTERLEAVE=True).
# Only meaningful where the item has DISTRACTOR images (n_images > |decisive|);
# for 2-image / all-decisive items there is nothing to "select", so the selection
# metrics are aggregated over the distractor subset only (n reported separately).
def _parse_image_id_set(raw):
    """Extract a set of image indices from a decisive/supporting-ids cell.
    Robust to '[1, 3]', '1;3', 'Image 1, Image 3'."""
    if raw is None:
        return set()
    s = str(raw).strip()
    if not s or s.lower() in ('nan', 'none', 'null', '[]'):
        return set()
    return set(int(x) for x in re.findall(r'\d+', s))


def _referenced_image_ids(text):
    """Image numbers the model explicitly referenced anywhere ('Image 3', 'image #3')."""
    return set(int(m) for m in re.findall(r'\bimage\s*#?\s*(\d+)', str(text), re.IGNORECASE))


def _labeled_image_ids(text):
    """Image numbers the model used as an explicit per-image LABEL at segment start
    ('Image 3:' / '**Image 3**:' / 'Image 3 -'), i.e. format-compliant labelling."""
    return set(int(m) for m in re.findall(
        r'(?im)^\s*[\*\-#>\s]*image\s*#?\s*(\d+)\s*\*{0,2}\s*[:\-\)]', str(text)))


def _det_image_attribution_metrics(prediction, n_images, decisive_ids, supporting_ids):
    """Deterministic image-grounding metrics. Returns a dict; keys are omitted when
    undefined (no gold decisive set, or no images), so callers can aggregate only
    what is present."""
    M = _referenced_image_ids(prediction)
    D = set(decisive_ids)
    S = set(supporting_ids) | D            # gold "is-evidence" set (decisive ∪ supporting)
    valid = set(range(1, int(n_images) + 1)) if n_images and int(n_images) > 0 else set()
    res = {}
    res['det_n_ref_images'] = len(M)
    if valid:
        M_valid = M & valid
        M_oor = M - valid                  # cited a NONEXISTENT image = fabrication
        res['det_image_coverage'] = len(M_valid) / len(valid)          # breadth of engagement
        # out-of-range = cited a NONEXISTENT image: gold-free fabrication signal.
        res['det_image_ref_out_of_range_rate'] = (len(M_oor) / len(M)) if M else 0.0
        res['det_image_format_compliance'] = len(_labeled_image_ids(prediction) & valid) / len(valid)
        # headline hallucinated-attribution rate: of everything the model cited,
        # how much is NOT a gold-supported image (off-target real image OR nonexistent).
        # Requires a gold evidence set S; undefined (omitted) when the item has no
        # decisive/supporting annotation, so we never punish citing unlabeled evidence.
        if M and S:
            offtarget = (M_valid - S) | M_oor
            res['det_image_attribution_hallucination_rate'] = len(offtarget) / len(M)
    if D:
        res['det_image_select_recall'] = len(M & D) / len(D)           # found the right images?
        M_valid = (M & valid) if valid else M
        if M_valid:
            res['det_image_select_precision'] = len(M_valid & D) / len(M_valid)
            res['det_image_attribution_offtarget_rate'] = len(M_valid - S) / len(M_valid)
    return res


class MRareBenchEvidenceVerif(ImageBaseDataset):
    """T2 (open-ended): free-text per-image evidence description, scored by a
    per-item BINARY rubric over that row's own `acceptable_evidence_points`.
    The judge runs ONLY when --judge is set, so the fleet can be inferred first
    (cheap) and judged later (the costed step).

    One class serves all three T2 input conditions; ``self.dataset_name`` selects
    a row of ``CONDITIONS``. The TSV, the rubric and the metrics are shared.
    """

    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    # ----- input-condition register -------------------------------------------
    #   MRareBench_EvidenceVerif      : images attached, diagnosis given  <- gold
    #   MRareBench_EvidenceVerif_TO   : NO image,        diagnosis given
    #   MRareBench_EvidenceVerif_NoDx : images attached, diagnosis withheld
    #
    # Each variant flips exactly one input against gold, so each contrast is
    # attributable. Both are opt-in probes (a few reference models), not part of
    # the default fleet sweep.
    #
    # _TO withholds the image and yields two signals:
    #   * required_recall(image) - required_recall(no-image) = genuine visual
    #     contribution (high no-image recall = reciting findings from the
    #     diagnosis prior rather than seeing them);
    #   * groundedness / fabrication = whether the answer invents visual evidence
    #     when the image channel is absent.
    #
    # _NoDx withholds the label and isolates its information gain:
    #   Delta_dx(level) = required_recall(dx given) - required_recall(withheld).
    # Level-1 (visible-finding) Delta_dx is the key signal: small = the model
    # reads the visible evidence without the label; large = the backward score
    # rode on the label.
    #
    # DELIBERATE, DO NOT "UNIFY": the instruction is byte-identical in all three
    # conditions, including _TO, where it still says "The images are labelled
    # Image 1, ...". That mismatch IS the fabrication probe -- a faithful model
    # cannot describe visible evidence that is absent. Hence no per-condition
    # instruction is registered here.
    CONDITIONS = {
        'MRareBench_EvidenceVerif': dict(attach_images=True, withhold_diagnosis=False),
        'MRareBench_EvidenceVerif_TO': dict(attach_images=False, withhold_diagnosis=False),
        'MRareBench_EvidenceVerif_NoDx': dict(attach_images=True, withhold_diagnosis=True),
    }

    DATASET_URL = {name: T2_TSV_URL for name in CONDITIONS}
    DATASET_MD5 = {}

    # The gold name lives ONLY in the leading 'Established diagnosis: X.' context
    # sentence (verified across all 608 rows: it appears nowhere else in context or
    # question_open), so stripping that sentence fully withholds it. The residual
    # phrase 'the established diagnosis' names no disease and is identical in both
    # conditions, so it cancels in Delta_dx.
    _DX_CTX_PREFIX = re.compile(r'^\s*Established diagnosis:\s*[^.]*\.\s*', re.I)

    @property
    def condition(self):
        """The registered input condition for this instance's dataset name."""
        cond = self.CONDITIONS.get(self.dataset_name)
        assert cond is not None, (
            f'{self.dataset_name!r} is not a registered MRareBench T2 input '
            f'condition; expected one of {list(self.CONDITIONS)}'
        )
        return cond

    # ----- data loading (local-first, T2 track) -------------------------------
    def load_data(self, dataset):
        # Single source of truth: the immutable FULL609 file (608 evaluable rows).
        # No fallback: any missing-file condition should surface loudly, not
        # silently switch to a stale copy. Row-limited smokes use MMRARE_T2_LIMIT
        # (in-memory) instead of touching the file.
        local_path = osp.join(LMUDataRoot(), BENCH_ROOT, T2_TRACK, T2_FULL_TSV_NAME)
        if osp.exists(local_path):
            data = pd.read_csv(local_path, sep='\t', encoding='utf-8')
            if 'index' not in data.columns:
                data['index'] = range(len(data))
            data['index'] = [str(x) for x in data['index']]
            limit = os.environ.get('MMRARE_T2_LIMIT', '').strip()
            if limit.isdigit() and int(limit) > 0:
                data = data.head(int(limit)).reset_index(drop=True)
            self.data_path = local_path
            return data
        url = self.DATASET_URL.get(dataset, None)
        if url:
            return self.prepare_tsv(url, None)
        raise FileNotFoundError(
            f'MRareBench T2 TSV not found at {local_path} and no usable DATASET_URL.'
        )

    # ----- image handling (base64 first, file-path fallback on the T2 track) ---
    def dump_image(self, line):
        from ..smp import decode_base64_to_image_file

        image_b64_list = parse_json_list(line.get('image', '[]'))
        if image_b64_list and any(image_b64_list):
            os.makedirs(self.img_root, exist_ok=True)
            tgt_paths = []
            for i, b64_data in enumerate(image_b64_list):
                if not b64_data:
                    continue
                idx = str(line.get('index', i))
                tgt_path = osp.join(self.img_root, f"{idx}--{i + 1}.jpg")
                if not read_ok(tgt_path):
                    try:
                        decode_base64_to_image_file(b64_data, tgt_path)
                    except Exception as e:
                        warnings.warn(f'Failed to decode base64 image: {e}')
                        continue
                tgt_paths.append(tgt_path)
            if tgt_paths:
                return tgt_paths

        image_list = parse_json_list(line.get('image_path', '[]'))
        if not image_list:
            return []
        track_dir = osp.join(LMUDataRoot(), BENCH_ROOT, T2_TRACK)
        abs_paths = []
        for rel_path in image_list:
            abs_paths.append(rel_path if osp.isabs(rel_path) else osp.join(track_dir, rel_path))
        for p in abs_paths:
            if not read_ok(p):
                warnings.warn(f'Image file not found: {p}')
        return abs_paths

    # ----- shared prompt helpers ----------------------------------------------
    def _image_messages(self, line):
        """Attach each image with a standardized 'Image k:' label matching the
        context 'Relevant images' line and the evidence points' [Image k] tokens.
        Labels come from image_slot_map.public_slot (in slot order); a positional
        fallback ('Image k') guarantees a label even if the slot map is absent.
        (The old figure_mapping parse read the wrong column and attached no label.)"""
        tgt_path = self.dump_image(line)
        slot_labels = []
        raw = line.get('image_slot_map', '')
        if pd.notna(raw) and str(raw).strip():
            try:
                sm = json.loads(str(raw))
                if isinstance(sm, list):
                    slot_labels = [str(s.get('public_slot', '')).strip()
                                   for s in sm if isinstance(s, dict)]
            except (json.JSONDecodeError, TypeError):
                slot_labels = []
        msgs = []
        for i, p in enumerate(tgt_path):
            label = slot_labels[i] if i < len(slot_labels) and slot_labels[i] else f'Image {i + 1}'
            msgs.append(dict(type='text', value=f'{label}:'))
            msgs.append(dict(type='image', value=p))
        return msgs

    def _meta_row(self, idx):
        meta = self.data
        hit = meta[meta['index'].astype(str) == str(idx)]
        return hit.iloc[0] if len(hit) else None

    @staticmethod
    def _identifiers(meta_row):
        out = {}
        for c in T2_ID_COLS + T2_META_COLS:
            if c == 'index':
                continue
            try:
                v = meta_row.get(c, '')
            except Exception:
                v = ''
            out[c] = '' if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
        return out

    @staticmethod
    def _category_bucket(category):
        """Stable 5-bucket evidence layer mapping for subgroup analysis."""
        c = re.sub(r'[^a-z0-9]+', '_', str(category or '').strip().lower()).strip('_')
        if c in {'abstract_modality', 'modality'}:
            return 'modality'
        if c in {'macroscopic_pattern', 'region_or_field', 'macro_pattern'}:
            return 'macro_pattern'
        if c in {'microstructural_pattern', 'micro_pattern'}:
            return 'micro_pattern'
        if c == 'cross_image_integration':
            return 'cross_image_integration'
        if c == 'diagnostic_attribution':
            return 'diagnostic_attribution'
        return 'other'

    # ----- prompt: describe the visual evidence supporting the given diagnosis --
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        cond = self.condition

        if cond['withhold_diagnosis']:
            # NoDx: the ONLY change vs gold. Copy first -- never mutate self.data.
            line = line.copy()
            ctx = str(line.get('context', '')) if pd.notna(line.get('context', '')) else ''
            line['context'] = self._DX_CTX_PREFIX.sub('', ctx).strip()

        context = str(line.get('context', '')) if pd.notna(line.get('context', '')) else ''
        question = str(line.get('question_open', '')) if pd.notna(line.get('question_open', '')) else ''
        if not question:
            question = str(line.get('question', '')) if pd.notna(line.get('question', '')) else ''
        instruction = (
            'You are an expert physician. The diagnosis is already established. '
            'The images are labelled Image 1, Image 2, ... For EACH image, refer to it '
            'explicitly by its label (e.g. "Image 1:") and state the main visible '
            'finding/pattern, then explain how the findings across the images fit '
            'together to support the established diagnosis. Be specific and describe '
            'only what is visible.'
        )
        prompt_parts = []
        if context:
            prompt_parts.append(context)
        if question:
            prompt_parts.append(f'Question: {question}')
        prompt_parts.append(instruction)
        prompt = '\n\n'.join(prompt_parts)

        msgs = self._image_messages(line) if cond['attach_images'] else []
        msgs.append(dict(type='text', value=prompt))
        return msgs

    # ----- rubric points -------------------------------------------------------
    @staticmethod
    def _split_points(meta_row):
        """Return (required, optional) lists of '[Image N] point' strings."""
        pts = parse_json_list(meta_row.get('acceptable_evidence_points', ''))
        required, optional = [], []
        for p in pts:
            if not isinstance(p, dict):
                continue
            text = str(p.get('point', '') or '').strip()
            if not text:
                continue
            img = str(p.get('image', '') or '').strip()
            entry = f'[{img}] {text}' if img else text
            if str(p.get('credit', '') or '').strip().lower() == 'required':
                required.append(entry)
            else:
                optional.append(entry)
        return required, optional

    @classmethod
    def _split_points_with_categories(cls, meta_row):
        """Return required/optional point strings plus evidence-dimension buckets."""
        pts = parse_json_list(meta_row.get('acceptable_evidence_points', ''))
        required, optional, req_cats, opt_cats = [], [], [], []
        for p in pts:
            if not isinstance(p, dict):
                continue
            text = str(p.get('point', '') or '').strip()
            if not text:
                continue
            img = str(p.get('image', '') or '').strip()
            entry = f'[{img}] {text}' if img else text
            bucket = cls._category_bucket(p.get('category', ''))
            if str(p.get('credit', '') or '').strip().lower() == 'required':
                required.append(entry)
                req_cats.append(bucket)
            else:
                optional.append(entry)
                opt_cats.append(bucket)
        return required, optional, req_cats, opt_cats

    def _build_rubric_prompt(self, answer, diagnosis, rationale, required, optional):
        def fmt(points):
            return '\n'.join(f'{i}. {p}' for i, p in enumerate(points, 1)) or '(none)'
        n_req, n_opt = len(required), len(optional)
        total = 2 * n_req + n_opt
        return (
            'You are an extremely strict attending-physician evaluator of a model\'s '
            'description of the VISIBLE evidence in medical images. For every YES/NO '
            'judgment default to NO; answer YES only when the criterion is clearly met '
            '(not vaguely, not by merely repeating the question or the diagnosis name).\n\n'
            f'Established diagnosis (given to the model; NOT what you are scoring): {diagnosis}\n\n'
            f'Reference rationale (ground truth of what the images show):\n{rationale}\n\n'
            f'MODEL ANSWER TO EVALUATE:\n{answer}\n\n'
            f'REQUIRED evidence points ({n_req}):\n{fmt(required)}\n\n'
            f'OPTIONAL evidence points ({n_opt}):\n{fmt(optional)}\n\n'
            'Output ONLY YES/NO, one per line, in EXACTLY this order:\n'
            f'- Lines 1..{n_req} = COVERAGE of each REQUIRED point (in order): does the '
            'answer explicitly and correctly state this point? (YES/NO)\n'
            f'- Next {n_opt} line(s) = COVERAGE of each OPTIONAL point (in order): same '
            'question. (YES/NO)\n'
            f'- Final {n_req} line(s) = CONTRADICTION of each REQUIRED point (in order): does '
            'the answer state something that DIRECTLY CONTRADICTS this point (claims it is '
            'absent, or claims the opposite finding)? Answer YES only for a clear '
            'contradiction, otherwise NO.\n\n'
            f'Reply with EXACTLY {total} lines, each only YES or NO, no numbering, no other text.'
        )

    @staticmethod
    def _parse_binary(response, n):
        vals = []
        for ln in str(response).splitlines():
            m = re.search(r'\b(yes|no)\b', ln.lower())
            if m:
                vals.append(1 if m.group(1) == 'yes' else 0)
        vals = vals[:n]
        while len(vals) < n:
            vals.append(0)
        return vals

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'prediction' in data.columns, 'Missing prediction column'
        data['prediction'] = [str(x) for x in data['prediction']]

        recs, req_lists, opt_lists, req_cat_lists, prompts = [], [], [], [], []
        det_cat_sims = {}
        # deterministic image-attribution accumulators (no API)
        det_scalar_accum = {}       # meaningful for all multi-image items
        det_distractor_accum = {}   # selection metrics: distractor subset only
        det_hier_vals = []
        _SELECT_KEYS = {'det_image_select_recall', 'det_image_select_precision',
                        'det_image_attribution_offtarget_rate'}
        for _, row in tqdm(data.iterrows(), total=len(data), desc='[DetMetrics] T2 evidence (local, no API)'):
            idx = str(row['index'])
            meta_row = self._meta_row(idx)
            rec = {'index': idx, 'prediction': row['prediction']}
            if meta_row is None:
                recs.append(rec)
                req_lists.append([])
                opt_lists.append([])
                req_cat_lists.append([])
                prompts.append(None)
                continue
            rec.update(self._identifiers(meta_row))
            required, optional, req_cats, opt_cats = self._split_points_with_categories(meta_row)
            diagnosis = str(meta_row.get('final_diagnosis', '') or '')
            rationale = str(meta_row.get('gold_rationale', '')
                            or meta_row.get('gold_answer_open', '') or '')
            rec['t2_required_points'] = json.dumps(required, ensure_ascii=False)
            rec['t2_optional_points'] = json.dumps(optional, ensure_ascii=False)
            rec['t2_required_categories'] = json.dumps(req_cats, ensure_ascii=False)
            rec['t2_optional_categories'] = json.dumps(opt_cats, ensure_ascii=False)

            # ----- deterministic metrics (ALWAYS, no judge/API) ----------------
            det = _det_open_metrics(row['prediction'], required, optional, rationale)
            for cat, sim in zip(req_cats, det.get('det_required_maxsims') or []):
                det_cat_sims.setdefault(cat, []).append(float(sim))
            for k, v in det.items():
                rec[k] = json.dumps([round(float(x), 4) for x in v]) if isinstance(v, list) else v

            # ----- deterministic image-attribution / grounding (no judge/API) ---
            try:
                imgs = parse_json_list(meta_row.get('image', '[]')) \
                    or parse_json_list(meta_row.get('image_path', '[]'))
                n_images = len([x for x in imgs if x])
            except Exception:
                n_images = 0
            decisive = _parse_image_id_set(meta_row.get('decisive_image_ids', ''))
            supporting = _parse_image_id_set(meta_row.get('supporting_image_ids', ''))
            attr = _det_image_attribution_metrics(row['prediction'], n_images, decisive, supporting)
            has_distractors = bool(decisive) and n_images > len(decisive)
            rec['det_n_images'] = n_images
            rec['det_n_decisive'] = len(decisive)
            rec['det_has_distractors'] = int(has_distractors)
            for k, v in attr.items():
                rec[k] = json.dumps(v) if isinstance(v, list) else v
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if k in _SELECT_KEYS:
                        if has_distractors:
                            det_distractor_accum.setdefault(k, []).append(float(v))
                    elif k != 'det_n_ref_images':
                        det_scalar_accum.setdefault(k, []).append(float(v))
            # deterministic hierarchical required recall (tau-thresholded mirror of
            # the judge hierarchy: L1 visible -> L2 cross-image -> L3 attribution).
            rq_sims = det.get('det_required_maxsims') or []
            if rq_sims and len(rq_sims) == len(req_cats):
                req_hits = [1 if s >= DET_HEADLINE_TAU else 0 for s in rq_sims]
                hs = _hier_required_score(req_hits, req_cats)
                if hs is not None:
                    rec['det_hierarchical_required_recall'] = hs['hierarchical_required_recall']
                    det_hier_vals.append(hs['hierarchical_required_recall'])

            recs.append(rec)
            req_lists.append(required)
            opt_lists.append(optional)
            req_cat_lists.append(req_cats)
            prompts.append(self._build_rubric_prompt(
                row['prediction'], diagnosis, rationale, required, optional))

        results = {'n_total': int(len(data))}

        # ---- aggregate deterministic metrics (always available) --------------
        def _det_agg(key):
            vals = [r[key] for r in recs if isinstance(r.get(key), (int, float))]
            return float(np.mean(vals)) if vals else None
        if recs:
            results['det_backend'] = next((r.get('det_backend') for r in recs
                                           if r.get('det_backend')), None)
        # threshold-independent means + thresholded recall across ALL robustness
        # bands (T2 design §9: report recall@{0.3,0.35,0.4,0.5,0.6}, not one tuned tau).
        det_agg_keys = ['det_required_meanmaxsim', 'det_optional_meanmaxsim',
                        'det_mean_sent_groundedness', 'det_image_grounded_meanmaxsim']
        for t in DET_TAUS:
            det_agg_keys += [f'det_required_recall@{t}', f'det_optional_recall@{t}',
                             f'det_grounded_ratio@{t}', f'det_image_grounded_recall@{t}']
        for key in det_agg_keys:
            v = _det_agg(key)
            if v is not None:
                results[key] = v
        for cat, vals in sorted(det_cat_sims.items()):
            if not vals:
                continue
            results[f'det_required_{cat}_n_points'] = int(len(vals))
            results[f'det_required_{cat}_meanmaxsim'] = float(np.mean(vals))
            for t in DET_TAUS:
                results[f'det_required_{cat}_recall@{t}'] = float(
                    np.mean([1.0 if s >= t else 0.0 for s in vals]))

        # ---- deterministic image-attribution / grounding aggregates (no API) --
        # coverage / format-compliance / hallucination: all multi-image items;
        # selection recall/precision/off-target: distractor subset only (n_images>|decisive|).
        for k, vals in det_scalar_accum.items():
            if vals:
                results[k] = float(np.mean(vals))
        results['n_image_attribution_scored'] = len(det_scalar_accum.get('det_image_coverage', []))
        for k, vals in det_distractor_accum.items():
            if vals:
                results[k] = float(np.mean(vals))
        results['n_image_select_scored_distractor_subset'] = len(
            det_distractor_accum.get('det_image_select_recall', []))
        if det_hier_vals:
            results['det_hierarchical_required_recall'] = float(np.mean(det_hier_vals))
            results['det_hierarchical_required_recall_drop'] = (
                results.get(f'det_required_recall@{DET_HEADLINE_TAU}', 0.0)
                - results['det_hierarchical_required_recall'])

        # ---- judge rubric (only when --judge is set; the costed step) ---------
        judge_name = judge_kwargs.get('model', None)
        if judge_name and judge_name != 'exact_matching':
            judge_model = build_judge(**judge_kwargs)
            if judge_model.working():
                nproc = dict(judge_kwargs).pop('nproc', 4)

                def _judge_call(model, prompt, n):
                    if prompt is None:
                        return dict(vec=[0] * n, raw='JUDGE_SKIP: no meta row')
                    try:
                        raw = model.generate(prompt)
                        return dict(vec=self._parse_binary(raw, n), raw=str(raw))
                    except Exception as err:
                        return dict(vec=[0] * n, raw=f'JUDGE_ERROR: {err}')

                tups = [dict(model=judge_model, prompt=p,
                             n=2 * len(req_lists[i]) + len(opt_lists[i]))
                        for i, p in enumerate(prompts)]
                # Resumable judge cache: a killed/stalled judge pass must not force
                # re-judging all rows from scratch. Only rows with a genuinely
                # successful cached verdict are skipped; JUDGE_ERROR entries (a
                # transient infra failure, never a verdict) are always retried.
                judge_keys = [recs[i]['index'] for i in range(len(tups))]

                def _judge_cache_valid(entry, expected_n):
                    if not isinstance(entry, dict):
                        return False
                    if str(entry.get('raw', '')).startswith('JUDGE_ERROR'):
                        return False
                    vec = entry.get('vec')
                    return isinstance(vec, list) and len(vec) == expected_n

                ckpt_path = get_intermediate_file_path(eval_file, f'_judge_{judge_name}_tmp')
                cached = load(ckpt_path) if osp.exists(ckpt_path) else {}
                todo = [i for i in range(len(tups))
                        if not _judge_cache_valid(cached.get(judge_keys[i]), tups[i]['n'])]

                if todo:
                    track_progress_rich(
                        _judge_call, [tups[i] for i in todo], nproc=nproc, chunksize=nproc,
                        save=ckpt_path, keys=[judge_keys[i] for i in todo],
                        description='[Judge] T2 evidence')
                    cached = load(ckpt_path) if osp.exists(ckpt_path) else cached

                out = [cached.get(judge_keys[i],
                                  dict(vec=[0] * tups[i]['n'], raw='JUDGE_ERROR: missing after resume'))
                       for i in range(len(tups))]

                req_recalls, opt_recalls, contra_rates = [], [], []
                hier_recalls = []
                hier_aux = {
                    'hier_visual_gate_pass_rate': [],
                    'hier_cross_gate_pass_rate': [],
                    'hier_level1_visual_recall': [],
                    'hier_level2_cross_recall_raw': [],
                    'hier_level2_cross_recall_gated': [],
                    'hier_level3_diag_recall_raw': [],
                    'hier_level3_diag_recall_gated': [],
                }
                judge_cat_hits = {}
                for i, o in enumerate(out):
                    n_req, n_opt = len(req_lists[i]), len(opt_lists[i])
                    vec = o['vec'] if isinstance(o, dict) else [0] * (2 * n_req + n_opt)
                    req_vec = vec[:n_req]
                    opt_vec = vec[n_req:n_req + n_opt]
                    con_vec = vec[n_req + n_opt:2 * n_req + n_opt]
                    rr = float(np.mean(req_vec)) if n_req else None
                    orr = float(np.mean(opt_vec)) if n_opt else None
                    cr = float(np.mean(con_vec)) if n_req else None
                    recs[i]['t2_required_vec'] = json.dumps(req_vec)
                    recs[i]['t2_optional_vec'] = json.dumps(opt_vec)
                    recs[i]['t2_contradiction_vec'] = json.dumps(con_vec)
                    recs[i]['t2_required_recall'] = rr if rr is not None else ''
                    recs[i]['t2_optional_recall'] = orr if orr is not None else ''
                    recs[i]['t2_contradiction_rate'] = cr if cr is not None else ''
                    recs[i]['t2_judge_raw'] = o.get('raw', '') if isinstance(o, dict) else ''
                    hs = _hier_required_score(req_vec, req_cat_lists[i])
                    if hs is not None:
                        recs[i]['t2_hierarchical_required_recall'] = hs['hierarchical_required_recall']
                        recs[i]['t2_hier_visual_gate'] = hs['hier_visual_gate']
                        recs[i]['t2_hier_cross_gate'] = hs['hier_cross_gate']
                        recs[i]['t2_hier_level1_visual_recall'] = hs['hier_level1_visual_recall']
                        recs[i]['t2_hier_level2_cross_recall_raw'] = hs['hier_level2_cross_recall_raw']
                        recs[i]['t2_hier_level2_cross_recall_gated'] = hs['hier_level2_cross_recall_gated']
                        recs[i]['t2_hier_level3_diag_recall_raw'] = hs['hier_level3_diag_recall_raw']
                        recs[i]['t2_hier_level3_diag_recall_gated'] = hs['hier_level3_diag_recall_gated']
                        hier_recalls.append(hs['hierarchical_required_recall'])
                        hier_aux['hier_visual_gate_pass_rate'].append(hs['hier_visual_gate'])
                        hier_aux['hier_cross_gate_pass_rate'].append(hs['hier_cross_gate'])
                        for hk in ('hier_level1_visual_recall',
                                   'hier_level2_cross_recall_raw',
                                   'hier_level2_cross_recall_gated',
                                   'hier_level3_diag_recall_raw',
                                   'hier_level3_diag_recall_gated'):
                            if hs[hk] is not None:
                                hier_aux[hk].append(hs[hk])
                    for cat, hit in zip(req_cat_lists[i], req_vec):
                        judge_cat_hits.setdefault(cat, []).append(int(hit))
                    if rr is not None:
                        req_recalls.append(rr)
                    if orr is not None:
                        opt_recalls.append(orr)
                    if cr is not None:
                        contra_rates.append(cr)

                results['required_recall'] = float(np.mean(req_recalls)) if req_recalls else 0.0
                results['full_required_coverage'] = (
                    float(np.mean([1.0 if r >= 1.0 else 0.0 for r in req_recalls]))
                    if req_recalls else 0.0)
                results['optional_recall'] = float(np.mean(opt_recalls)) if opt_recalls else 0.0
                results['required_contradiction_rate'] = (
                    float(np.mean(contra_rates)) if contra_rates else 0.0)
                results['n_judged_required'] = len(req_recalls)
                if hier_recalls:
                    results['hierarchical_required_recall'] = float(np.mean(hier_recalls))
                    results['hierarchical_required_recall_drop'] = (
                        results['required_recall'] - results['hierarchical_required_recall'])
                    results['full_hierarchical_required_coverage'] = float(
                        np.mean([1.0 if r >= 1.0 else 0.0 for r in hier_recalls]))
                    for hk, vals in hier_aux.items():
                        if vals:
                            results[hk] = float(np.mean(vals))
                # per-task_type required_recall
                by_tt = {}
                for i in range(len(recs)):
                    if recs[i].get('t2_required_recall', '') == '':
                        continue
                    tt = str(recs[i].get('task_type', '')) or '(blank)'
                    by_tt.setdefault(tt, []).append(float(recs[i]['t2_required_recall']))
                results['required_recall_by_task_type'] = {
                    tt: dict(n=len(v), required_recall=float(np.mean(v)))
                    for tt, v in by_tt.items()}
                results['required_recall_by_evidence_category'] = {
                    cat: dict(n=len(v), required_recall=float(np.mean(v)))
                    for cat, v in sorted(judge_cat_hits.items()) if v}
                # cross-check: judge required_recall vs deterministic recall (free sanity)
                diffs = []
                dkey = f'det_required_recall@{DET_HEADLINE_TAU}'
                for i in range(len(recs)):
                    jr = recs[i].get('t2_required_recall', '')
                    dr = recs[i].get(dkey, None)
                    if jr != '' and isinstance(dr, (int, float)):
                        diffs.append(abs(float(jr) - float(dr)))
                if diffs:
                    results['judge_vs_det_required_recall_mae'] = float(np.mean(diffs))
            else:
                warnings.warn('Judge API unavailable; deterministic metrics only (no rubric scores).')
        else:
            warnings.warn('No --judge specified; deterministic metrics computed, rubric skipped. '
                          'Re-run evaluate with --judge to add the LLM rubric metrics.')

        out_df = pd.DataFrame(recs)
        score_file = get_intermediate_file_path(eval_file, '_detailed_metrics', 'json')
        dump(results, score_file)
        per_row_file = get_intermediate_file_path(eval_file, '_per_row', 'xlsx')
        dump(out_df, per_row_file)
        return results
