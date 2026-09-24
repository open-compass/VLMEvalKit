import base64
import hashlib
import json
import os
import os.path as osp
import re
import sqlite3
import subprocess
import sys
import tempfile

import pandas as pd
from huggingface_hub import snapshot_download

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

from vlmeval.smp import (LMUDataRoot, decode_base64_to_image_file, dump,
                         get_intermediate_file_path, get_logger, load, read_ok, toliststr)
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils.judge_util import build_judge

logger = get_logger(__name__)

# Reasoning is routed by the actual output contract, not by broad benchmark
# category. In particular, closed bitstring and ranking tasks explicitly forbid
# explanations even though some of them belong to reasoning-heavy categories.
REASONING_QIDS = {
    'constraintfollow_001',
    'euler-stratification_001',
    'gaia-dr3_001',
    'pineapple-avocado-hm_001',
    'poet2-progen2_001',
    'spatial-dreamer-ladder_001',
    'spatial-ssrl-fig1_001',
    'spatial_ssrl_modify_001',
}

SCORER_VERSION = 'rules_v5_final_answer_only'
# Rule-only fixes do not invalidate deterministic LLM-judge responses. Keep the
# judge cache namespace stable until the judge prompt or transport changes.
JUDGE_CACHE_VERSION = 'final_answer_v1'

# Only tasks whose question and reference define a closed, mechanical contract
# are routed to Python. Every unlisted task, including unreviewed json_match
# rows, falls back to the GPT-5.4-mini judge.
RULE_SCORER_BY_QID = {
    'depth-anything3_001': 'structured_json',
    'environment3_001': 'structured_json',
    'np_a2_02_001': 'structured_json',
    'ProtFlowArticle_001': 'structured_json',
    'np_a3_01_001': 'structured_json',
    'ppo_001': 'structured_json',
    'spa3r_001': 'structured_json',
    'chartassistant_001': 'structured_json',
    'mmifengine_002': 'structured_json',
    'np_a4_01_001': 'structured_json',
    'np_a4_02_001': 'structured_json',
    'saprot_001': 'structured_json',
    'np_a5_04_001': 'structured_json',
    'np_a5_05_001': 'structured_json',
    'np_a5_06_001': 'structured_json',
    'np_a5_07_001': 'structured_json',
    'gspo_001': 'structured_json',
    'hptransfer_001': 'structured_json',
    'np_b2_04_001': 'structured_json',
    'np_b2_05_001': 'structured_json',
    'spatial-ssrl_001': 'structured_json',
    'np_c2_02_001': 'structured_json',
    'schubert-pipe-dreams_001': 'structured_json',
    'sle-glyco_002': 'structured_json',
    'rlhf-formula-compare_001': 'structured_json',
    'cambrian-llavaov-mammoth_001': 'structured_json',
    'cambrian-llavaov-mammoth_002': 'structured_json',
    'np_d2_01_001': 'structured_json',
    'np_d2_02_001': 'structured_json',
    'np_d3_01_001': 'structured_json',
    'np_d3_02_001': 'structured_json',
    'sle-glyco-igg_001': 'structured_json',
    'gspo_002': 'structured_json',
    'itwa-spin_001': 'structured_json',
    'np_e1_01_001': 'structured_json',
    'mmifengine_003': 'structured_json',
    'np_f2_01_001': 'structured_json',
    'data_sample_001': 'structured_json',
    'np_a1_01_001': 'ranking',
    'np_a1_02_001': 'ranking',
    'np_a1_03_001': 'ranking',
    'np_a1_04_001': 'ranking',
    'np_a1_05_001': 'ranking',
    'np_c1_04_001': 'bitstring',
    'np_c1_05_001': 'bitstring',
    'np_c1_06_001': 'bitstring',
    'np_c1_07_001': 'bitstring',
    'np_c1_08_001': 'bitstring',
    'np_c1_09_001': 'bitstring',
    'p11b_001': 'closed_scalar',
    'sle-glyco_001': 'sle_choice_groups',
    'np_c3_03_001': 'unordered_exact_dicts',
    'np_d3_01new_001': 'indexed_exact_dicts',
    'twiffnew_001': 'predicted_exact_dicts',
    'np_a2_02_002': 'subfigure_choice',
    'np_a3_01_002': 'keyed_list_items',
    'np_a4_02_001_dup2': 'keyed_list_items',
}

# These tasks explicitly make list order part of the answer contract. Lists in
# other audited structured tasks are treated as unordered collections.
ORDERED_JSON_LIST_QIDS = {
    'depth-anything3_001',
    'np_a2_02_001',
    'np_a4_01_001',
    'np_a4_02_001',
    'np_d2_01_001',
    'np_d2_02_001',
    'np_d3_01_001',
    'np_e1_01_001',
    'np_f2_01_001',
}

STRUCTURED_SCALAR_ALIASES_BY_QID = {
    'gspo_002': {'violet': 'purple'},
}

KEYED_LIST_ITEM_RULE_CONFIG = {
    'np_a3_01_002': {'format_score': 0.10, 'item_score': 0.10},
    'np_a4_02_001_dup2': {'format_score': 0.09, 'item_score': 0.07},
}

# ── Normalization helpers (ported from SciDocBench eval.py) ──────────────────


def normalize_location(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'\btab\.\s*', 'table ', s)
    s = re.sub(r'\bfig\.\s*', 'figure ', s)
    s = re.sub(r'(\d+(?:\.\d+)*)\.\s', r'\1 ', s)
    s = s.rstrip('.')
    return s.strip()


def normalize_equation(s: str) -> str:
    m = re.search(r'\d+', s.strip())
    return f"Eq. ({m.group()})" if m else s.strip().lower()


def normalize_roles(s: str) -> str:
    parts = [r.strip() for r in s.split(",") if r.strip()]
    return ",".join(sorted(parts))


_SUP_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def normalize_number(s: str):
    s = s.strip().replace(",", "").translate(_SUP_MAP)
    s = re.sub(r'\s*[×x]\s*10\^?\{?(-?\d+)\}?', lambda m: f'e{m.group(1)}', s)
    try:
        return f"{float(s):g}"
    except ValueError:
        return None


# ── JSON parsing helpers ─────────────────────────────────────────────────────


def _repair_json_escapes(s: str) -> str:
    return re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', s)


def _extract_json_block(s: str) -> str:
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return s.strip()


def _safe_json_loads(s: str):
    s = _extract_json_block(s)
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(_repair_json_escapes(s))
        except Exception:
            return None


def _parse_json_like(value):
    if not isinstance(value, str):
        return value
    parsed = _safe_json_loads(value)
    if parsed is not None:
        return parsed
    # A few legacy references contain a trailing comma before the closing list.
    repaired = re.sub(r',\s*([}\]])', r'\1', value.strip())
    try:
        return json.loads(_repair_json_escapes(repaired))
    except Exception:
        return value


def _extract_final_answer(text, expect_think_end=False):
    """Remove model reasoning and return only the answer after ``</think>``.

    Qwen chat templates commonly place ``<think>`` in the generation prompt,
    so decoded completions contain the closing tag but not the opening tag. If
    other rows from the same inference run contain that closing tag, a row
    without one is an unfinished reasoning trace rather than a final answer.
    """
    text = str(text or '')
    closing_tags = list(re.finditer(r'</think\s*>', text, flags=re.IGNORECASE))
    if closing_tags:
        return text[closing_tags[-1].end():].strip()

    if re.search(r'<think\b[^>]*>', text, flags=re.IGNORECASE):
        return ''
    if expect_think_end:
        return ''
    return text.strip()


def _extract_structured_prediction(text):
    """Return the last complete JSON value, preferring the final answer block."""
    fenced = re.findall(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    for candidate in reversed(fenced):
        parsed = _parse_json_like(candidate)
        if not isinstance(parsed, str):
            return parsed

    parsed = _parse_json_like(text)
    if not isinstance(parsed, str):
        return parsed

    decoder = json.JSONDecoder()
    candidates = []
    for start, char in enumerate(text):
        if char not in '[{':
            continue
        for candidate in (text[start:], _repair_json_escapes(text[start:])):
            try:
                value, consumed = decoder.raw_decode(candidate)
            except Exception:
                continue
            candidates.append((start + consumed, -start, value))
            break
    return max(candidates, default=(None, None, None))[2]


def _pair_qid(item):
    pair_qid = str(item.get('pair_qid', '') or '')
    if pair_qid:
        return pair_qid
    return str(item.get('qid', '') or '').split('__', 1)[0]


def _parse_segments(raw):
    if isinstance(raw, list):
        return raw
    if raw is None:
        return []
    if isinstance(raw, float) and pd.isna(raw):
        return []
    if not isinstance(raw, str):
        return []

    raw = raw.strip()
    if not raw:
        return []
    try:
        segments = json.loads(raw)
    except Exception:
        return []
    return segments if isinstance(segments, list) else []


def _segment_image_lookup(paths):
    lookup = {}
    for path in toliststr(paths):
        basename = osp.basename(path)
        stem = osp.splitext(basename)[0]
        for key in (path, basename, stem):
            lookup.setdefault(key, path)
    return lookup


def _resolve_segment_image(image_name, lookup):
    image_name = str(image_name)
    basename = osp.basename(image_name)
    stem = osp.splitext(basename)[0]
    for key in (image_name, basename, stem):
        if key in lookup:
            return lookup[key]
    return image_name


def _json_safe_value(value):
    if value is None:
        return ''

    try:
        if pd.isna(value):
            return ''
    except (TypeError, ValueError):
        pass

    if hasattr(value, 'item'):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass

    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    return value


# ── Evaluation methods ───────────────────────────────────────────────────────


def _scalar_matches(key, pred_val, ans_val, scalar_aliases=None):
    if pred_val is None or ans_val is None:
        return pred_val is None and ans_val is None
    p, a = str(pred_val).strip(), str(ans_val).strip()
    suffix = key.rsplit('.', 1)[-1].lower() if '.' in key else key.lower()
    if suffix == 'location':
        return normalize_location(p) == normalize_location(a)
    if suffix in {'models', 'tasks', 'choice'} or key.startswith('['):
        return normalize_roles(p.upper()) == normalize_roles(a.upper())
    if '\\' in key:
        return normalize_equation(p) == normalize_equation(a)
    pn, an = normalize_number(p), normalize_number(a)
    if pn is not None and an is not None:
        return pn == an
    p = re.sub(r'\s+', ' ', p).casefold()
    a = re.sub(r'\s+', ' ', a).casefold()
    if scalar_aliases:
        p = scalar_aliases.get(p, p)
        a = scalar_aliases.get(a, a)
    return p == a


def _max_unordered_list_score(
        pred, answer, key, exact_dict_keys=False, scalar_aliases=None):
    if not pred or not answer:
        return 0.0
    scores = [
        [_structured_similarity(
            p,
            a,
            f'{key}[]',
            ordered_lists=False,
            exact_dict_keys=exact_dict_keys,
            scalar_aliases=scalar_aliases,
        ) for p in pred]
        for a in answer
    ]
    if len(pred) <= 12:
        states = {0: 0.0}
        for row in scores:
            updated = dict(states)
            for mask, total in states.items():
                for index, score in enumerate(row):
                    if mask & (1 << index):
                        continue
                    new_mask = mask | (1 << index)
                    updated[new_mask] = max(updated.get(new_mask, 0.0), total + score)
            states = updated
        return max(states.values(), default=0.0)

    # Large lists in SciDocBench contain scalar labels. A deterministic greedy
    # fallback avoids exponential matching while still granting item credit.
    candidates = sorted(
        (score, ai, pi)
        for ai, row in enumerate(scores)
        for pi, score in enumerate(row)
        if score > 0
    )
    used_answer, used_pred, total = set(), set(), 0.0
    for score, ai, pi in reversed(candidates):
        if ai in used_answer or pi in used_pred:
            continue
        used_answer.add(ai)
        used_pred.add(pi)
        total += score
    return total


def _structured_similarity(
        pred, answer, key='', ordered_lists=False, exact_dict_keys=False,
        scalar_aliases=None):
    if isinstance(answer, dict):
        if not isinstance(pred, dict):
            return 0.0
        if not answer:
            return 1.0 if not pred else 0.0
        matched = sum(
            0.0 if field not in pred else _structured_similarity(
                pred[field], expected, f'{key}.{field}' if key else str(field),
                ordered_lists=ordered_lists,
                exact_dict_keys=exact_dict_keys,
                scalar_aliases=scalar_aliases,
            )
            for field, expected in answer.items()
        )
        denominator = max(len(answer), len(pred)) if exact_dict_keys else len(answer)
        return matched / denominator

    if isinstance(answer, list):
        if not isinstance(pred, list):
            return 0.0
        denominator = max(len(answer), len(pred))
        if denominator == 0:
            return 1.0
        if ordered_lists:
            matched = sum(
                _structured_similarity(
                    p,
                    a,
                    f'{key}[{index}]',
                    ordered_lists=True,
                    exact_dict_keys=exact_dict_keys,
                    scalar_aliases=scalar_aliases,
                )
                for index, (p, a) in enumerate(zip(pred, answer))
            )
        else:
            matched = _max_unordered_list_score(
                pred,
                answer,
                key,
                exact_dict_keys=exact_dict_keys,
                scalar_aliases=scalar_aliases,
            )
        return matched / denominator

    return float(_scalar_matches(key, pred, answer, scalar_aliases))


def eval_json_match(
        prediction: str, answer, ordered_lists=False, exact_dict_keys=True,
        scalar_aliases=None) -> tuple:
    pred = _extract_structured_prediction(prediction)
    if pred is None:
        return 0.0, 'Failed to parse prediction as JSON'
    answer = _parse_json_like(answer)
    if not isinstance(answer, (dict, list)):
        return 0.0, f'Unsupported JSON answer type: {type(answer).__name__}'
    if not isinstance(pred, type(answer)):
        return 0.0, f'Prediction type {type(pred).__name__} does not match reference'
    score = _structured_similarity(
        pred,
        answer,
        ordered_lists=ordered_lists,
        exact_dict_keys=exact_dict_keys,
        scalar_aliases=scalar_aliases,
    )
    return score, (
        f'structured rule score={score:.4f}, ordered_lists={ordered_lists}, '
        f'exact_dict_keys={exact_dict_keys}'
    )


def _walk_scalars(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_scalars(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_scalars(child)
    elif value is not None:
        yield str(value)


def _normalized_closed_value(value):
    value = str(value).strip().strip('`"\'')
    value = value.replace('&', 'and')
    value = re.sub(r'[^a-zA-Z0-9]+', ' ', value).strip().lower()
    return value


def _extract_choice_set(prediction):
    structured = _extract_structured_prediction(prediction)
    values = []
    if isinstance(structured, dict):
        for key, value in structured.items():
            if str(key).lower() == 'choice':
                values.append(str(value))
    matches = re.findall(r'["\']?Choice["\']?\s*[:=]\s*["\']([^"\'\n}]+)',
                         prediction, re.IGNORECASE)
    values.extend(matches)
    if not values:
        return None
    return {letter.upper() for letter in re.findall(r'[A-J]', values[-1], re.IGNORECASE)}


def _rule_bitstring(prediction, answer):
    reference_strings = [
        value for value in _walk_scalars(_parse_json_like(answer))
        if re.fullmatch(r'[01]+', value.strip())
    ]
    if not reference_strings:
        return 0.0, 'bitstring rule: reference bitstring not found'
    reference = reference_strings[-1].strip()

    candidates = [
        value.strip() for value in _walk_scalars(
            _extract_structured_prediction(prediction))
        if re.fullmatch(r'[01]+', value.strip())
    ]
    candidates.extend(re.findall(rf'(?<![01])[01]{{{len(reference)}}}(?![01])', prediction))
    candidates = [value for value in candidates if len(value) == len(reference)]
    if not candidates:
        return 0.0, f'bitstring rule: no {len(reference)}-bit final answer'
    candidate = candidates[-1]
    matched = sum(left == right for left, right in zip(reference, candidate))
    return matched / len(reference), f'bitstring rule: {matched}/{len(reference)} bits matched'


def _rule_ranking(prediction, answer):
    reference = str(answer).strip().strip('"\'')
    expected = [part.strip().upper() for part in reference.split('>') if part.strip()]
    rankings = re.findall(r'(?<![A-Z])([A-Z](?:\s*>\s*[A-Z]){2,})(?![A-Z])',
                          prediction.upper())
    if not rankings:
        return 0.0, 'ranking rule: no ranking sequence found'
    candidate = [part.strip() for part in rankings[-1].split('>')]
    if len(candidate) != len(expected):
        return 0.0, f'ranking rule: expected {len(expected)} positions, got {len(candidate)}'
    matched = sum(left == right for left, right in zip(expected, candidate))
    return matched / len(expected), f'ranking rule: {matched}/{len(expected)} positions matched'


def _rule_closed_scalar(prediction, answer):
    def select_value(value):
        if isinstance(value, dict):
            lowered = {str(key).lower(): child for key, child in value.items()}
            for key in ('answer', 'category', 'final_answer', 'choice'):
                if key in lowered:
                    return lowered[key]
        values = list(_walk_scalars(value))
        return values[-1] if values else value

    reference_value = select_value(_parse_json_like(answer))
    reference = _normalized_closed_value(reference_value)
    structured = _extract_structured_prediction(prediction)
    if structured is not None:
        candidate_value = select_value(structured)
        candidate = _normalized_closed_value(candidate_value)
        score = float(candidate == reference)
        reference_token = str(reference_value).strip().strip('`"\'')
        if not score and re.fullmatch(r'[A-Z][A-Z0-9-]{1,15}', reference_token):
            candidate_tokens = {
                token.upper()
                for token in re.findall(r'[A-Za-z0-9-]+', str(candidate_value))
            }
            score = float(reference_token.upper() in candidate_tokens)
        return score, f'closed scalar rule: expected={reference!r}, got={candidate!r}'

    tail = prediction.strip().splitlines()[-1] if prediction.strip() else ''
    candidate = _normalized_closed_value(tail)
    score = float(
        candidate == reference
        or re.search(rf'\b{re.escape(reference)}\b', candidate) is not None
    )
    return score, f'closed scalar rule: expected={reference!r}, final_line={candidate!r}'


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _rule_unordered_exact_dicts(prediction, answer):
    expected = _parse_json_like(answer)
    candidate = _extract_structured_prediction(prediction)
    if not isinstance(expected, list) or not isinstance(candidate, list):
        return 0.0, 'unordered dict rule: prediction/reference is not a JSON list'
    remaining = [_canonical_json(value) for value in candidate if isinstance(value, dict)]
    matched = 0
    for value in expected:
        encoded = _canonical_json(value)
        if encoded in remaining:
            matched += 1
            remaining.remove(encoded)
    score = matched / len(expected) if expected else float(not candidate)
    return score, f'unordered dict rule: {matched}/{len(expected)} exact dictionaries matched'


def _rule_indexed_exact_dicts(prediction, answer):
    expected = _parse_json_like(answer)
    candidate = _extract_structured_prediction(prediction)
    if not isinstance(expected, list) or not isinstance(candidate, list):
        return 0.0, 'indexed dict rule: prediction/reference is not a JSON list'
    if len(candidate) != len(expected):
        return 0.0, f'indexed dict rule: expected {len(expected)} rows, got {len(candidate)}'
    if not all(isinstance(value, dict) for value in candidate):
        return 0.0, 'indexed dict rule: every row must be a dictionary'
    expected_keys = [set(value) for value in expected]
    if any(set(value) != keys for value, keys in zip(candidate, expected_keys)):
        return 0.0, 'indexed dict rule: one or more rows have incorrect fields'
    matched = sum(
        _canonical_json(predicted) == _canonical_json(reference)
        for predicted, reference in zip(candidate, expected)
    )
    score = 0.04 + 0.12 * matched
    return min(score, 1.0), f'indexed dict rule: format=0.04, exact_rows={matched}/{len(expected)}'


def _rule_predicted_exact_dicts(prediction, answer):
    expected = _parse_json_like(answer)
    candidate = _extract_structured_prediction(prediction)
    if not isinstance(expected, list) or not isinstance(candidate, list) or not candidate:
        return 0.0, 'predicted dict rule: expected a non-empty JSON list'
    remaining = [_canonical_json(value) for value in expected]
    score, correct, hallucinated = 0.0, 0, 0
    for value in candidate:
        if not isinstance(value, dict) or set(value) != {'sample', 'title'}:
            return 0.0, 'predicted dict rule: every item must contain sample and title only'
        encoded = _canonical_json(value)
        if encoded in remaining:
            score += 0.33
            correct += 1
            remaining.remove(encoded)
        else:
            score -= 0.2
            hallucinated += 1
    score = max(0.0, min(1.0, score))
    return score, f'predicted dict rule: correct={correct}, hallucinated={hallucinated}'


def _rule_sle_choice_groups(prediction):
    choices = _extract_choice_set(prediction)
    if choices is None:
        return 0.0, 'SLE choice rule: Choice field not found'
    components = [
        0.3 if {'A', 'C'} <= choices and not {'B', 'D'} & choices else 0.0,
        0.1 if 'E' in choices else 0.0,
        0.3 if {'F', 'G'} <= choices and 'H' not in choices else 0.0,
        0.3 if 'J' in choices and 'I' not in choices else 0.0,
    ]
    return sum(components), f'SLE choice rule: choices={sorted(choices)}, groups={components}'


def _rule_subfigure_choice(prediction, answer):
    expected = _parse_json_like(answer)
    candidate = _extract_structured_prediction(prediction)
    if not isinstance(expected, dict) or not isinstance(candidate, dict):
        return 0.0, 'subfigure choice rule: final JSON object not found'
    subfigure = normalize_location(str(candidate.get('Subfigure', '')))
    expected_subfigure = normalize_location(str(expected.get('Subfigure', '')))
    if subfigure != expected_subfigure:
        return 0.0, f'subfigure choice rule: expected {expected_subfigure!r}, got {subfigure!r}'
    choices = {letter.upper() for letter in re.findall(r'[A-D]', str(candidate.get('Choice', '')))}
    choice_score = 0.5 if choices == {'C', 'D'} else 0.25 if choices in ({'C'}, {'D'}) else 0.0
    return 0.5 + choice_score, f'subfigure choice rule: choices={sorted(choices)}'


def _rule_keyed_list_items(prediction, answer, config):
    expected = _parse_json_like(answer)
    candidate = _extract_structured_prediction(prediction)
    if not isinstance(expected, dict) or not isinstance(candidate, dict):
        return 0.0, 'keyed list rule: prediction/reference is not a JSON object'
    if not all(isinstance(value, list) for value in candidate.values()):
        return 0.0, 'keyed list rule: every predicted value must be a list'

    item_total = sum(
        _structured_similarity(
            candidate[key],
            expected_value,
            key=str(key),
            ordered_lists=False,
            exact_dict_keys=True,
        ) if key in candidate else 0.0
        for key, expected_value in expected.items()
    )
    score = config['format_score'] + config['item_score'] * item_total
    score = max(0.0, min(1.0, score))
    return score, (
        f'keyed list rule: format={config["format_score"]:.2f}, '
        f'item_similarity={item_total:.4f}/{len(expected)}'
    )


def eval_rule(rule_name, prediction, answer, qid=None):
    if rule_name == 'bitstring':
        return _rule_bitstring(prediction, answer)
    if rule_name == 'ranking':
        return _rule_ranking(prediction, answer)
    if rule_name == 'closed_scalar':
        return _rule_closed_scalar(prediction, answer)
    if rule_name == 'sle_choice_groups':
        return _rule_sle_choice_groups(prediction)
    if rule_name == 'unordered_exact_dicts':
        return _rule_unordered_exact_dicts(prediction, answer)
    if rule_name == 'indexed_exact_dicts':
        return _rule_indexed_exact_dicts(prediction, answer)
    if rule_name == 'predicted_exact_dicts':
        return _rule_predicted_exact_dicts(prediction, answer)
    if rule_name == 'subfigure_choice':
        return _rule_subfigure_choice(prediction, answer)
    if rule_name == 'keyed_list_items':
        return _rule_keyed_list_items(
            prediction, answer, KEYED_LIST_ITEM_RULE_CONFIG[qid])
    raise ValueError(f'Unknown SciDocBench rule scorer: {rule_name}')


def eval_exec_match(prediction: str, answer: dict) -> tuple:
    code = prediction
    m = re.search(r'```(?:python)?\s*\n(.*?)\n```', prediction, re.DOTALL)
    if m:
        code = m.group(1)

    input_path = answer.get("input_path", "")
    reference_script = answer["reference_script"]
    embedded_input_path = None
    embedded_input = answer.get("input_image_base64")
    if embedded_input:
        suffix = osp.splitext(str(answer.get("input_path", "")))[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as image_file:
            image_file.write(base64.b64decode(embedded_input, validate=True))
            embedded_input_path = image_file.name
        input_path = embedded_input_path
    if not input_path:
        return 0.0, "Missing exec input image"

    def run_script(script, out_path):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as cf:
            cf.write(script)
            script_file = cf.name
        try:
            result = subprocess.run(
                [sys.executable, script_file, input_path, out_path],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                lines = result.stderr.strip().splitlines()
                return lines[-1] if lines else "no stderr"
            return None
        finally:
            os.unlink(script_file)

    ref_out = tempfile.mktemp(suffix=".png")
    pred_out = tempfile.mktemp(suffix=".png")
    try:
        err = run_script(reference_script, ref_out)
        if err:
            return 0.0, f"Reference script failed: {err}"
        if not os.path.exists(ref_out):
            return 0.0, "Reference script produced no output file"

        err = run_script(code, pred_out)
        if err:
            return 0.0, f"Execution failed: {err}"
        if not os.path.exists(pred_out):
            return 0.0, "Code ran but produced no output file"

        from PIL import Image, ImageChops
        ref = Image.open(ref_out)
        pred_img = Image.open(pred_out)

        if ref.size != pred_img.size:
            return 0.0, f"Size mismatch: expected {ref.size}, got {pred_img.size}"
        if ref.mode != pred_img.mode:
            pred_img = pred_img.convert(ref.mode)

        diff = ImageChops.difference(ref, pred_img)
        if diff.getbbox() is None:
            return 1.0, "Pixel-perfect match"

        histogram = diff.histogram()
        band_count = len(ref.getbands())
        squared_error = sum(
            value ** 2 * count
            for band in range(band_count)
            for value, count in enumerate(
                histogram[band * 256:(band + 1) * 256]
            )
        )
        mse = squared_error / (ref.size[0] * ref.size[1] * band_count)
        if mse <= 10:
            return 0.5, f"Near-match (MSE={mse:.2f})"
        return 0.0, f"Pixel mismatch (MSE={mse:.2f})"
    except subprocess.TimeoutExpired:
        return 0.0, "Execution timed out (>30s)"
    except Exception as e:
        return 0.0, f"Eval error: {e}"
    finally:
        for p in (ref_out, pred_out, embedded_input_path):
            if not p:
                continue
            try:
                os.unlink(p)
            except OSError:
                pass


def _parse_judge_response(raw: str) -> tuple:
    def result_tuple(result):
        score = float(result.get('score', result.get('reasoning_score', 0)))
        if not 0.0 <= score <= 1.0:
            raise ValueError(f'Judge score outside [0, 1]: {score}')
        return score, result.get('eval_note', result.get('reasoning_note', ''))

    try:
        result = json.loads(raw)
        return result_tuple(result)
    except Exception:
        pass
    try:
        result = json.loads(_repair_json_escapes(raw))
        return result_tuple(result)
    except Exception:
        pass
    m = re.search(r'"(?:score|reasoning_score)"\s*:\s*([0-9.]+)', raw)
    if m:
        score = float(m.group(1))
        if 0.0 <= score <= 1.0:
            return score, f'[score extracted via regex] {raw}'
    return 0.0, f'Failed to parse valid judge response: {raw}'


# ── Judge prompt templates ───────────────────────────────────────────────────

SCIDOC_JUDGE_POLICY = """\
You are the deterministic grader for SciDocBench. Follow the supplied scoring
rubric mechanically; the reference answer and rubric are authoritative.

Grading policy:
- Grade the model's final answer. If analysis is followed by a final JSON,
  table, code block, or explicit conclusion, use that final answer rather than
  abandoned intermediate guesses.
- For JSON, lists, tables, LaTeX, and Mermaid, ignore whitespace, key order,
  code fences, and harmless formatting differences unless the rubric explicitly
  scores them. Compare required facts and relationships semantically.
- Do not require byte-for-byte equality unless the rubric explicitly says
  "exact". Award every partial-credit component independently and apply all
  stated exclusions, gates, deductions, and score caps.
- Do not invent requirements or use outside knowledge. Calculate the rubric
  components internally, then return one final score in [0.0, 1.0].
"""

SCIDOC_JUDGE_PROMPT = """\
Score the prediction by factual coverage and accuracy. Treat the reference as a
set of required facts. Give proportional credit for correct required facts,
reduce credit for contradictions or hallucinated additions, and do not score
style or formatting unless the question requires it.

Question:
{prompt}

Reference answer:
{answer}

Model prediction:
{prediction}

Respond with a JSON object only, no extra text:
{{"score": <float between 0.0 and 1.0>, "eval_note": "<brief reason>"}}"""

SCIDOC_REASONING_CHECK_PROMPT = """\
You are auditing a model's reasoning on a scientific-paper task. You do NOT
have access to the paper itself, so you cannot verify whether cited numbers or
references truly exist. Assess only what can be judged from the text alone.

Question the model was given:
{prompt}

Model's full output (answer + reasoning):
{prediction}

Rate the reasoning on two independent axes, each in [0.0, 1.0]:

1. internal_consistency — Does the chain of reasoning actually arrive at the
   final answer the model gives? Penalize contradictions between the stated
   reasoning and the stated answer, or logical leaps that skip steps that would
   change the conclusion. A short reasoning that cleanly justifies a short
   answer is fine; consistency is about coherence, not length.

2. no_hallucination — Does the reasoning avoid fabricating specifics? Penalize
   invented table/figure/equation numbers, made-up citation keys, suspiciously
   precise numeric claims that look retrofitted to the answer, or appeals to
   content the question doesn't suggest exists. Vague but honest reasoning
   ("the table shows X is higher") is preferable to confident-sounding
   fabrications ("Table 7 row 3 reports 42.1%").

Respond with a JSON object only, no extra text:
{{"internal_consistency": <float>, "no_hallucination": <float>, \
"reasoning_note": "<brief explanation, no newlines>"}}"""


_judge_cache_path = None
_judge_cache_namespace = JUDGE_CACHE_VERSION


def _prepare_judge_template(judge_prompt):
    template = judge_prompt if judge_prompt else SCIDOC_JUDGE_PROMPT
    template = re.sub(
        r'<\s*1\.0\s+or\s+0\.0\s*>',
        '<float between 0.0 and 1.0>',
        template,
        flags=re.IGNORECASE,
    )
    return f'{SCIDOC_JUDGE_POLICY}\n\nTask-specific rubric:\n{template}'


def _cached_judge_generate(judge_model, message):
    if not _judge_cache_path:
        return judge_model.generate(message, temperature=0)
    cache_key = hashlib.sha256(
        f'{_judge_cache_namespace}\0{message}'.encode('utf-8')).hexdigest()

    def lookup():
        with sqlite3.connect(_judge_cache_path, timeout=60) as connection:
            connection.execute(
                'CREATE TABLE IF NOT EXISTS judge_cache '
                '(cache_key TEXT PRIMARY KEY, response TEXT NOT NULL)')
            row = connection.execute(
                'SELECT response FROM judge_cache WHERE cache_key = ?',
                (cache_key,),
            ).fetchone()
            return row[0] if row is not None else None

    try:
        cached = lookup()
        if cached is not None:
            return cached
    except sqlite3.Error as error:
        logger.warning('SciDocBench judge cache read failed: %s', error)

    lock_dir = f'{_judge_cache_path}.locks'
    if fcntl is None:
        response = judge_model.generate(message, temperature=0)
        return response
    os.makedirs(lock_dir, exist_ok=True)
    with open(osp.join(lock_dir, f'{cache_key}.lock'), 'a', encoding='utf-8') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            try:
                cached = lookup()
                if cached is not None:
                    return cached
            except sqlite3.Error as error:
                logger.warning('SciDocBench judge cache recheck failed: %s', error)

            response = judge_model.generate(message, temperature=0)
            try:
                with sqlite3.connect(_judge_cache_path, timeout=60) as connection:
                    connection.execute(
                        'CREATE TABLE IF NOT EXISTS judge_cache '
                        '(cache_key TEXT PRIMARY KEY, response TEXT NOT NULL)')
                    connection.execute(
                        'INSERT OR REPLACE INTO judge_cache(cache_key, response) VALUES (?, ?)',
                        (cache_key, response),
                    )
            except sqlite3.Error as error:
                logger.warning('SciDocBench judge cache write failed: %s', error)
            return response
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def eval_judge(judge_model, prediction, answer, prompt, judge_prompt=None):
    template = _prepare_judge_template(judge_prompt)
    message = template.format(prompt=prompt, answer=answer, prediction=prediction)
    raw = _cached_judge_generate(judge_model, message)
    return _parse_judge_response(raw)


def _parse_reasoning_response(raw: str) -> tuple:
    """Parse the two-axis reasoning judge output; average the axes."""
    def _extract(obj):
        ic = obj.get("internal_consistency")
        nh = obj.get("no_hallucination")
        if ic is None or nh is None:
            return None
        score = (float(ic) + float(nh)) / 2.0
        note = obj.get("reasoning_note", "")
        return score, f"ic={float(ic):.2f}, nh={float(nh):.2f}; {note}"

    try:
        result = _extract(json.loads(raw))
        if result is not None:
            return result
    except Exception:
        pass
    try:
        result = _extract(json.loads(_repair_json_escapes(raw)))
        if result is not None:
            return result
    except Exception:
        pass
    ic_m = re.search(r'"internal_consistency"\s*:\s*([0-9.]+)', raw)
    nh_m = re.search(r'"no_hallucination"\s*:\s*([0-9.]+)', raw)
    if ic_m and nh_m:
        ic, nh = float(ic_m.group(1)), float(nh_m.group(1))
        return (ic + nh) / 2.0, f"[regex] ic={ic:.2f}, nh={nh:.2f}"
    return 0.0, f"Failed to parse reasoning response: {raw}"


def eval_reasoning(judge_model, prediction, question):
    message = SCIDOC_REASONING_CHECK_PROMPT.format(
        prompt=question, prediction=prediction)
    raw = _cached_judge_generate(judge_model, message)
    return _parse_reasoning_response(raw)


# ── Parallel evaluation helper ───────────────────────────────────────────────

_judge_model = None


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(f'{name} must be one of 1/0, true/false, yes/no, or on/off.')


def _configure_judge_transport(judge_kwargs):
    """Use only transport options supported by upstream VLMEvalKit."""
    judge_kwargs.setdefault('temperature', 0.0)
    logger.info('SciDocBench judge temperature: %s', judge_kwargs['temperature'])
    return judge_kwargs


def _configure_content_cache(judge_kwargs, model_name):
    global _judge_cache_path, _judge_cache_namespace

    value = os.environ.get('SCIDOC_JUDGE_CONTENT_CACHE', 'auto').strip()
    if value.lower() in {'', '0', 'false', 'off', 'none'}:
        _judge_cache_path = None
    else:
        if value.lower() == 'auto':
            value = osp.abspath(osp.join('.cache', 'scidocbench_judge_cache.sqlite3'))
        _judge_cache_path = osp.expanduser(value)
        os.makedirs(osp.dirname(_judge_cache_path) or '.', exist_ok=True)
    _judge_cache_namespace = json.dumps({
        'scorer_version': JUDGE_CACHE_VERSION,
        'model': model_name,
        'api_base': judge_kwargs.get('api_base', judge_kwargs.get('base_url')),
        'temperature': judge_kwargs.get('temperature', 0),
    }, sort_keys=True)
    logger.info('SciDocBench content-addressed judge cache: %s',
                _judge_cache_path or 'disabled')


def _resolve_eval_method(item):
    qid = _pair_qid(item)
    rule_name = RULE_SCORER_BY_QID.get(qid)
    if rule_name:
        return f'rule:{rule_name}'
    source_method = str(item.get('eval_method', 'judge') or 'judge')
    return 'exec_match' if source_method == 'exec_match' else 'judge'


def _parse_field(raw, fallback):
    """Parse a JSON-serialized field from TSV, with fallback."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return fallback
    return raw if isinstance(raw, dict) else fallback


def _eval_one_item(item_json):
    """Evaluate a single sample. Called by track_progress_rich.

    Returns (answer_score, reasoning_score, note). Formal evaluation only uses
    the final answer and leaves ``reasoning_score`` unset. The reasoning trace
    can be audited separately by setting ``SCIDOC_EVAL_REASONING_DIAGNOSTIC=1``;
    that diagnostic never changes the answer score.
    """
    item = json.loads(item_json)
    prediction = str(item.get('prediction', ''))
    final_answer = _extract_final_answer(
        prediction,
        expect_think_end=bool(item.get('_expect_think_end', False)),
    )
    eval_method = _resolve_eval_method(item)
    qid = _pair_qid(item)
    has_reasoning_contract = (
        bool(item.get('_enable_reasoning_diagnostic', False))
        and qid in REASONING_QIDS
    )

    # Formal benchmark policy: after the inference backend has exhausted its
    # configured retries, keep the sample in the denominator and score the
    # failed prediction as zero.  Do this before any answer/reasoning judge
    # calls so persistent API failures cannot consume judge requests or be
    # mistaken for a substantive response.
    if 'Failed to obtain answer via API' in prediction:
        reasoning_score = 0.0 if has_reasoning_contract else None
        return (
            0.0,
            reasoning_score,
            'API failure after configured retries; scored as zero.',
        )

    answer = _parse_field(item.get('answer', '{}'), item.get('answer', ''))

    judge_prompt = item.get('judge_prompt', '')
    if isinstance(judge_prompt, float) or not judge_prompt:
        judge_prompt = None

    question = str(item.get('question', ''))

    try:
        if not final_answer:
            answer_score, note = 0.0, 'No final answer after model reasoning.'
        elif eval_method.startswith('rule:'):
            rule_name = eval_method.split(':', 1)[1]
            if rule_name == 'structured_json':
                answer_score, note = eval_json_match(
                    final_answer,
                    answer,
                    ordered_lists=qid in ORDERED_JSON_LIST_QIDS,
                    exact_dict_keys=True,
                    scalar_aliases=STRUCTURED_SCALAR_ALIASES_BY_QID.get(qid),
                )
            else:
                answer_score, note = eval_rule(
                    rule_name, final_answer, answer, qid=qid)
        elif eval_method == 'judge':
            answer_str = (json.dumps(answer, ensure_ascii=False)
                          if isinstance(answer, (dict, list)) else str(answer))
            answer_score, note = eval_judge(
                _judge_model, final_answer, answer_str, question, judge_prompt)
        elif eval_method == 'exec_match':
            answer_score, note = eval_exec_match(final_answer, answer)
        else:
            answer_score, note = 0.0, f"Unknown eval_method: {eval_method}"

        reasoning_score = None
        if has_reasoning_contract and _judge_model is not None:
            reasoning_score, reason_note = eval_reasoning(
                _judge_model, prediction, question)
            note = (f"answer={answer_score:.2f}, reasoning={reasoning_score:.2f}; "
                    f"{note}; reasoning: {reason_note}")
    except Exception as e:
        return 0.0, None, f"Eval error: {e}"

    return answer_score, reasoning_score, note


# ── Dataset class ────────────────────────────────────────────────────────────


class SciDocBench(ImageBaseDataset):

    TYPE = 'VQA'
    DEFAULT_JUDGE_MODEL = 'gpt-5.4-mini'
    HF_REPO_ID = 'HenryExcellent/SciDocBench'
    HF_REVISION = '2f7fc6f5dd37707a89b14202af2ac6878f678206'

    DATASET_URL = {
        'SciDocBench': (
            'https://huggingface.co/datasets/HenryExcellent/SciDocBench/'
            f'resolve/{HF_REVISION}/SciDocBench.tsv'
        ),
    }
    DATASET_MD5 = {
        'SciDocBench': '2507953151fa2cc0dbbe363ab65bc870',
    }

    def __init__(self, dataset='SciDocBench', skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)
        self._ensure_document_images()

    def _missing_document_images(self):
        missing = []
        for value in self.data['image_path']:
            for raw_path in toliststr(value):
                if read_ok(raw_path) or read_ok(osp.join(self.img_root, raw_path)):
                    continue
                missing.append(raw_path)
        return missing

    def _ensure_document_images(self):
        missing = self._missing_document_images()
        if not missing:
            return

        logger.info(
            'Downloading SciDocBench document images from %s (%d missing references).',
            self.HF_REPO_ID,
            len(missing),
        )
        snapshot_download(
            repo_id=self.HF_REPO_ID,
            repo_type='dataset',
            revision=self.HF_REVISION,
            local_dir=LMUDataRoot(),
            allow_patterns=['images/SciDocBench/**'],
        )
        missing = self._missing_document_images()
        if missing:
            raise FileNotFoundError(
                f'Failed to materialize {len(missing)} SciDocBench image references; '
                f'first missing path: {missing[0]!r}'
            )

    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        if 'image' in line and isinstance(line['image'], list):
            tgt_path = []
            if 'image_path' in line:
                image_path = (line['image_path'] if isinstance(line['image_path'], list)
                              else [line['image_path']])
            else:
                image_path = [f"{line['index']}_{i}.jpg" for i in range(len(line['image']))]
            for img, im_name in zip(line['image'], image_path):
                path = osp.join(self.img_root, im_name)
                os.makedirs(osp.dirname(path), exist_ok=True)
                if not read_ok(path):
                    decode_base64_to_image_file(img, path)
                tgt_path.append(path)
        elif 'image' in line and isinstance(line['image'], str):
            tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
            if not read_ok(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = []
            for raw_path in toliststr(line['image_path']):
                if read_ok(raw_path):
                    tgt_path.append(raw_path)
                    continue
                resolved_path = osp.join(self.img_root, raw_path)
                if not read_ok(resolved_path):
                    raise FileNotFoundError(
                        f"Could not resolve SciDocBench image {raw_path!r} "
                        f"directly or below {self.img_root!r}"
                    )
                tgt_path.append(resolved_path)
        return tgt_path

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        # The portable release stores paths relative to ``self.img_root``.
        tgt_path = self.dump_image(line)

        segments = _parse_segments(line.get('segments', ''))
        if segments:
            lookup = _segment_image_lookup(tgt_path)
            msgs = []
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                if 'image' in segment:
                    msgs.append(dict(
                        type='image',
                        value=_resolve_segment_image(segment['image'], lookup)))
                elif 'text' in segment and str(segment['text']).strip():
                    msgs.append(dict(type='text', value=str(segment['text'])))
            if msgs:
                return msgs

        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        global _judge_model

        nproc = judge_kwargs.pop('nproc', 4)
        judge_kwargs = _configure_judge_transport(judge_kwargs)
        model_name = judge_kwargs.get('model', cls.DEFAULT_JUDGE_MODEL)
        _configure_content_cache(judge_kwargs, model_name)

        storage = get_intermediate_file_path(
            eval_file, f'_{model_name}_{SCORER_VERSION}')
        tmp_file = get_intermediate_file_path(
            eval_file, f'_{model_name}_{SCORER_VERSION}', 'pkl')

        if osp.exists(storage):
            logger.info(f'Scoring file {storage} already exists, will reuse.')
        else:
            data = load(eval_file)

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            enable_reasoning_diagnostic = _env_flag(
                'SCIDOC_EVAL_REASONING_DIAGNOSTIC', default=False)
            # Qwen thinking templates inject the opening tag into the prompt,
            # so it is absent from decoded predictions. Detect that convention
            # once per inference artifact and treat rows without ``</think>``
            # as truncated reasoning with no scorable final answer.
            expect_think_end = any(
                re.search(
                    r'</think\s*>',
                    str(line.get('prediction', '')),
                    flags=re.IGNORECASE,
                )
                for line in lines
            )
            indices = [str(line['index']) for line in lines]
            route_counts = {}
            for line in lines:
                route = _resolve_eval_method(line)
                route_counts[route] = route_counts.get(route, 0) + 1
            logger.info('SciDocBench effective scoring routes: %s', route_counts)

            # Serialize each row to JSON for the worker function
            tups = []
            for line in lines:
                item = {}
                for col in data.columns:
                    item[col] = _json_safe_value(line[col])
                item['_expect_think_end'] = expect_think_end
                item['_enable_reasoning_diagnostic'] = enable_reasoning_diagnostic
                tups.append(json.dumps(item, ensure_ascii=False))

            # Load checkpoint and skip already-evaluated items
            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
                logger.info(f'Loaded {len(ans)} cached results from {tmp_file}')

            remaining_tups = [x for x, i in zip(tups, indices) if i not in ans]
            remaining_indices = [i for i in indices if i not in ans]

            if len(remaining_indices):
                _judge_model = build_judge(max_tokens=8192, **judge_kwargs)
                new_results = track_progress_rich(
                    _eval_one_item,
                    remaining_tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=remaining_indices,
                    save=tmp_file,
                )
                for k, v in zip(remaining_indices, new_results):
                    ans[k] = v

            # Build result rows in original order. Tolerate legacy 2-tuple
            # cached entries from older runs.
            results = []
            for line in lines:
                sid = str(line['index'])
                cached = ans.get(sid)
                if cached is None:
                    answer_score, reasoning_score, note = 0.0, None, 'Not evaluated'
                elif len(cached) == 3:
                    answer_score, reasoning_score, note = cached
                else:
                    answer_score, note = cached
                    reasoning_score = None
                results.append({
                    'index': sid,
                    'qid': line.get('qid', ''),
                    'pair_qid': line.get('pair_qid', line.get('qid', '')),
                    'partition': line.get('partition', ''),
                    'mode': line.get('mode', ''),
                    'language': line.get('language', ''),
                    'category': line.get('category', ''),
                    'eval_method': _resolve_eval_method(line),
                    'score': answer_score,
                    'reasoning_score': reasoning_score,
                    'eval_note': note,
                })

            result_df = pd.DataFrame(results)
            dump(result_df, storage)

        # Load from storage and aggregate
        result_df = load(storage)

        def _mean_pct(series):
            vals = series.dropna()
            return round(vals.mean() * 100, 2) if len(vals) else float('nan')

        def _major_category(category):
            match = re.match(r'^[A-G]', str(category))
            return match.group(0) if match else None

        summary_rows = []
        summary_rows.append({
            'Category': 'Overall (answer)',
            'Num': len(result_df),
            'Score': _mean_pct(result_df['score']),
        })
        if 'reasoning_score' in result_df.columns:
            reasoning_subset = result_df[
                result_df['pair_qid'].isin(REASONING_QIDS)
                & result_df['reasoning_score'].notna()
            ]
            if len(reasoning_subset):
                summary_rows.append({
                    'Category': 'Reasoning (question whitelist)',
                    'Num': len(reasoning_subset),
                    'Score': _mean_pct(reasoning_subset['reasoning_score']),
                })

        preferred_partitions = [
            'en_all_first',
            'en_interleave',
            'zh_all_first',
            'zh_interleave',
        ]
        available_partitions = [
            str(value)
            for value in result_df.get('partition', pd.Series(dtype=str)).dropna().unique()
            if str(value)
        ]
        partition_order = [
            value for value in preferred_partitions if value in available_partitions
        ] + sorted(
            value for value in available_partitions if value not in preferred_partitions
        )
        for partition in partition_order:
            subset = result_df[result_df['partition'] == partition]
            summary_rows.append({
                'Category': f'partition:{partition}',
                'Num': len(subset),
                'Score': _mean_pct(subset['score']),
            })

        for column in ('mode', 'language'):
            if column not in result_df:
                continue
            for value in sorted(
                str(item) for item in result_df[column].dropna().unique() if str(item)
            ):
                subset = result_df[result_df[column] == value]
                summary_rows.append({
                    'Category': f'{column}:{value}',
                    'Num': len(subset),
                    'Score': _mean_pct(subset['score']),
                })

        for method in sorted(result_df['eval_method'].unique()):
            subset = result_df[result_df['eval_method'] == method]
            summary_rows.append({
                'Category': f'method:{method}',
                'Num': len(subset),
                'Score': _mean_pct(subset['score']),
            })

        major_categories = result_df['category'].map(_major_category)
        for major in sorted(x for x in major_categories.dropna().unique()):
            subset = result_df[major_categories == major]
            summary_rows.append({
                'Category': major,
                'Num': len(subset),
                'Score': _mean_pct(subset['score']),
            })

        for cat in sorted(result_df['category'].unique()):
            subset = result_df[result_df['category'] == cat]
            summary_rows.append({
                'Category': cat,
                'Num': len(subset),
                'Score': _mean_pct(subset['score']),
            })

        summary = pd.DataFrame(summary_rows)
        score_file = get_intermediate_file_path(
            eval_file, f'_acc_{SCORER_VERSION}', 'csv')
        dump(summary, score_file)
        logger.info(f'SciDocBench evaluation finished. Results saved to {score_file}')
        logger.info(f'\n{summary.to_string(index=False)}')
        return summary

    @classmethod
    def report_primary_metric(cls, metrics: dict | None) -> dict:
        if not isinstance(metrics, dict) or not metrics:
            return {}

        key = 'Category=Overall (answer)|Score'
        if key in metrics:
            return {'Overall Answer Score': metrics[key]}
        return super().report_primary_metric(metrics)
