import json
import os
import os.path as osp
import re
import subprocess
import tempfile
from collections import Counter

import pandas as pd

from vlmeval.smp import (decode_base64_to_image_file, dump, get_intermediate_file_path, get_logger,
                         load, read_ok, toliststr)
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils.judge_util import build_judge

logger = get_logger(__name__)

# Categories that genuinely require a reasoning chain (derivation, data
# cross-checking, cross-doc synthesis, code writing, alignment). Everything else
# is fact lookup / extraction where the answer alone is what matters — the
# ``reasoning`` field is neither requested in the prompt (stripped at prepare
# time; see scidocbench_prepare.py) nor evaluated here.
REASONING_CATEGORIES = {"C1", "C2", "C3", "D2", "D3", "E2", "F1", "F2", "G1"}

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


def _json_loads_relaxed(s: str):
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(_repair_json_escapes(s))
        except Exception:
            return None


def _extract_json_block(s: str) -> str:
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return s.strip()


def _iter_json_object_spans(s: str):
    """Yield balanced JSON object spans while respecting quoted strings."""
    pos = 0
    while pos < len(s):
        start = s.find('{', pos)
        if start < 0:
            break

        stack = []
        in_str = False
        escaped = False
        end = None

        for idx in range(start, len(s)):
            ch = s[idx]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == '\\':
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
            elif ch in '{[':
                stack.append(ch)
            elif ch in '}]':
                if not stack:
                    break
                top = stack.pop()
                if (top == '{' and ch != '}') or (top == '[' and ch != ']'):
                    break
                if not stack:
                    end = idx + 1
                    break

        if end is None:
            pos = start + 1
            continue

        yield start, end
        pos = end


def _only_json_separators(s: str, spans) -> bool:
    separators = ' \t\r\n,`[]{}'
    pos = 0
    for start, end in spans:
        if s[pos:start].strip(separators):
            return False
        pos = end
    return not s[pos:].strip(separators)


def _safe_json_loads(s: str):
    s = _extract_json_block(s)

    obj = _json_loads_relaxed(s)
    if obj is not None:
        return obj

    spans = []
    objs = []
    for start, end in _iter_json_object_spans(s):
        obj = _json_loads_relaxed(s[start:end])
        if isinstance(obj, dict):
            spans.append((start, end))
            objs.append(obj)

    if not objs:
        return None
    if len(objs) == 1:
        return objs[0]

    if _only_json_separators(s, spans):
        merged = {}
        for obj in objs:
            merged.update(obj)
        return merged

    return objs[-1]


def _strip_thinking_for_answer(prediction: str) -> str:
    """Remove leaked thinking tags before answer-only scoring."""
    text = str(prediction).strip()
    if '</think>' in text:
        return text.rsplit('</think>', 1)[1].strip()
    return re.sub(r'(?is)<think>.*?</think>\s*', '', text).strip()


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


def _add_segment_image_aliases(lookup, raw_paths, resolved_paths):
    for raw_path, resolved_path in zip(toliststr(raw_paths), toliststr(resolved_paths)):
        basename = osp.basename(raw_path)
        stem = osp.splitext(basename)[0]
        for key in (raw_path, basename, stem):
            lookup.setdefault(key, resolved_path)
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


def _merge_list_of_dicts(obj):
    if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
        merged = {}
        for item in obj:
            merged.update(item)
        return merged
    return obj


def _coerce_json_list(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ('answer_list', 'answers', 'answer', 'items', 'metrics'):
            value = obj.get(key)
            if isinstance(value, list):
                return value
        if len(obj) == 1:
            value = next(iter(obj.values()))
            if isinstance(value, list):
                return value
    return None


def _normalize_list_item(value):
    if isinstance(value, str):
        text = re.sub(r'\s+', ' ', value.strip())
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value).strip()

    number = normalize_number(text)
    if number is not None:
        return f'num:{number}'
    return text.casefold()


def _lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _eval_json_list_match(pred_list, ans_list):
    pred_norm = [_normalize_list_item(x) for x in pred_list]
    ans_norm = [_normalize_list_item(x) for x in ans_list]

    if not ans_norm:
        score = 1.0 if not pred_norm else 0.0
        return score, f"LCS-F1 list match: lcs=0, pred={len(pred_norm)}, answer=0"
    if not pred_norm:
        return 0.0, f"LCS-F1 list match: lcs=0, pred=0, answer={len(ans_norm)}"

    matched = _lcs_len(pred_norm, ans_norm)
    precision = matched / len(pred_norm)
    recall = matched / len(ans_norm)
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return score, f"LCS-F1 list match: lcs={matched}, pred={len(pred_norm)}, answer={len(ans_norm)}"


def _eval_json_list_unordered_match(pred_list, ans_list):
    pred_norm = [_normalize_list_item(x) for x in pred_list]
    ans_norm = [_normalize_list_item(x) for x in ans_list]

    if not ans_norm:
        score = 1.0 if not pred_norm else 0.0
        return score, f"unordered multiset-F1 list match: matched=0, pred={len(pred_norm)}, answer=0"
    if not pred_norm:
        return 0.0, f"unordered multiset-F1 list match: matched=0, pred=0, answer={len(ans_norm)}"

    pred_counts = Counter(pred_norm)
    ans_counts = Counter(ans_norm)
    matched = sum(min(pred_counts[item], ans_counts[item]) for item in ans_counts)
    precision = matched / len(pred_norm)
    recall = matched / len(ans_norm)
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return score, f"unordered multiset-F1 list match: matched={matched}, pred={len(pred_norm)}, answer={len(ans_norm)}"


ORDERED_LIST_PATTERNS = (
    r'\bin\s+the\s+order\b',
    r'\bin\s+order\b',
    r'\border\s+they\s+(?:are\s+)?(?:introduced|appear)\b',
    r'\btop[-\s]+to[-\s]+bottom\b',
    r'\bleft[-\s]+to[-\s]+right\b',
    r'\bchronological(?:ly)?\b',
    r'\bnums\s+is\s+this\s+json\s+array\b',
    r'\bexact\s+integers\s+only\b',
    r'顺序保持一致',
    r'一一对应',
    r'出现顺序',
    r'引入顺序',
    r'从上到下',
    r'从左到右',
)


def _matches_any_pattern(text, patterns):
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _json_list_match_mode(question='', key=''):
    """Choose ordered vs unordered list scoring from strong order cues.

    Only prompts where order is part of the answer semantics use ordered LCS.
    Other lists, including alphabetically sorted output lists, use unordered
    multiset-F1 to avoid over-penalizing reference-specific sort conventions.
    """
    text = str(question or '')
    if _matches_any_pattern(text, ORDERED_LIST_PATTERNS):
        return 'ordered'
    return 'unordered'


def _eval_json_list_by_mode(pred_list, ans_list, question='', key=''):
    mode = _json_list_match_mode(question, key)
    if mode == 'unordered':
        return _eval_json_list_unordered_match(pred_list, ans_list)
    return _eval_json_list_match(pred_list, ans_list)


def eval_json_match(prediction: str, answer: dict, question: str = '') -> tuple:
    pred = _safe_json_loads(prediction)
    if pred is None:
        return 0.0, "Failed to parse prediction as JSON"
    pred = _merge_list_of_dicts(pred)
    answer = _merge_list_of_dicts(answer)

    if isinstance(answer, list):
        pred_list = _coerce_json_list(pred)
        if pred_list is None:
            return 0.0, "Prediction is not a JSON list"
        return _eval_json_list_by_mode(pred_list, answer, question, '<root>')

    if not isinstance(pred, dict):
        return 0.0, "Prediction is not a JSON object"
    if not isinstance(answer, dict):
        return 0.0, "Reference answer is not a JSON object"

    def value_score(key, pred_val, ans_val):
        ans_list = _coerce_json_list(ans_val)
        if ans_list is not None:
            pred_list = _coerce_json_list(pred_val)
            if pred_list is None:
                return 0.0
            return _eval_json_list_by_mode(pred_list, ans_list, question, key)[0]

        p, a = str(pred_val).strip(), str(ans_val).strip()
        suffix = key.rsplit(".", 1)[-1] if "." in key else key
        if suffix == "location":
            return float(normalize_location(p) == normalize_location(a))
        if suffix in ("models", "tasks"):
            return float(normalize_roles(p) == normalize_roles(a))
        if key.startswith("["):
            return float(normalize_roles(p) == normalize_roles(a))
        if "\\" in key:
            return float(normalize_equation(p) == normalize_equation(a))
        pn, an = normalize_number(p), normalize_number(a)
        if pn is not None and an is not None:
            return float(pn == an)
        return float(p == a)

    matched = sum(value_score(k, pred.get(k, ""), v) for k, v in answer.items())
    total = len(answer)
    score = matched / total if total > 0 else 0.0
    matched_str = f"{matched:g}" if float(matched).is_integer() else f"{matched:.2f}"
    return score, f"{matched_str}/{total} keys matched"


def eval_exec_match(prediction: str, answer: dict) -> tuple:
    code = prediction
    m = re.search(r'```(?:python)?\s*\n(.*?)\n```', prediction, re.DOTALL)
    if m:
        code = m.group(1)

    input_path = answer["input_path"]
    reference_script = answer["reference_script"]

    def run_script(script, out_path):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as cf:
            cf.write(script)
            script_file = cf.name
        try:
            result = subprocess.run(
                ["python3", script_file, input_path, out_path],
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

        mse = sum(
            (v / 255.0) ** 2 * count
            for v, count in enumerate(diff.histogram())
        ) / (ref.size[0] * ref.size[1] * len(ref.getbands()))
        if mse <= 10:
            return 0.5, f"Near-match (MSE={mse:.2f})"
        return 0.0, f"Pixel mismatch (MSE={mse:.2f})"
    except subprocess.TimeoutExpired:
        return 0.0, "Execution timed out (>30s)"
    except Exception as e:
        return 0.0, f"Eval error: {e}"
    finally:
        for p in (ref_out, pred_out):
            try:
                os.unlink(p)
            except OSError:
                pass


def _extract_score_denominator(prompt: str):
    """Infer denominator from custom judge prompts that ask for count/N."""
    if not prompt:
        return None

    patterns = [
        r'score\s*=\s*[^/\n]+/\s*(\d+)',
        r'each\s+worth\s+1\s*/\s*(\d+)',
        r'(\d+)\s+operations?\s+total',
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            denom = int(match.group(1))
            if denom > 1:
                return denom
    return None


def _sanitize_unit_score(score, note, denominator=None):
    score = float(score)
    if pd.isna(score):
        return 0.0, f"[invalid score={score}] {note}"
    if 0.0 <= score <= 1.0:
        return score, note

    if (denominator is not None and 1.0 < score <= denominator
            and abs(score - round(score)) < 1e-8):
        return score / denominator, f"[normalized score={score:g}/{denominator}] {note}"

    clipped = min(max(score, 0.0), 1.0)
    return clipped, f"[clipped out-of-range score={score:g}] {note}"


def _sanitize_optional_unit_score(score, note, denominator=None):
    if score is None:
        return None, note
    try:
        if pd.isna(score):
            return None, note
    except Exception:
        pass
    try:
        return _sanitize_unit_score(score, note, denominator)
    except Exception:
        return None, f"[invalid score={score}] {note}"


def _score_values_equal(a, b):
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except Exception:
        pass
    return a == b


def _sanitize_cached_result(cached, line=None):
    """Normalize cached per-sample scores from older evaluator versions."""
    if cached is None:
        return cached, False

    try:
        cached_len = len(cached)
    except TypeError:
        return (0.0, None, f"Invalid cached result: {cached}"), True

    if cached_len == 3:
        answer_score, reasoning_score, note = cached
    elif cached_len == 2:
        answer_score, note = cached
        reasoning_score = None
    else:
        return (0.0, None, f"Invalid cached result: {cached}"), True

    denominator = None
    if line is not None and str(line.get('eval_method', '')) == 'judge':
        judge_prompt = line.get('judge_prompt', '')
        if not isinstance(judge_prompt, float) or not pd.isna(judge_prompt):
            denominator = _extract_score_denominator(str(judge_prompt))

    try:
        new_answer_score, new_note = _sanitize_unit_score(
            answer_score, note, denominator)
    except Exception:
        new_answer_score, new_note = 0.0, f"[invalid score={answer_score}] {note}"
    new_reasoning_score, new_note = _sanitize_optional_unit_score(
        reasoning_score, new_note)
    sanitized = (new_answer_score, new_reasoning_score, new_note)
    changed = (
        cached_len != 3
        or not _score_values_equal(answer_score, new_answer_score)
        or not _score_values_equal(reasoning_score, new_reasoning_score)
        or note != new_note
    )
    return sanitized, changed


def _parse_judge_response(raw: str, denominator=None) -> tuple:
    try:
        result = json.loads(raw)
        score = result.get("score", result.get("reasoning_score", 0))
        note = result.get("eval_note", result.get("reasoning_note", ""))
        return _sanitize_unit_score(score, note, denominator)
    except Exception:
        pass
    try:
        result = json.loads(_repair_json_escapes(raw))
        score = result.get("score", result.get("reasoning_score", 0))
        note = result.get("eval_note", result.get("reasoning_note", ""))
        return _sanitize_unit_score(score, note, denominator)
    except Exception:
        pass
    m = re.search(r'"(?:score|reasoning_score)"\s*:\s*([0-9.]+)', raw)
    if m:
        return _sanitize_unit_score(
            float(m.group(1)), f"[score extracted via regex] {raw}", denominator)
    return 0.0, f"Failed to parse judge response: {raw}"


# ── Judge prompt templates ───────────────────────────────────────────────────

SCIDOC_JUDGE_PROMPT = """\
You are an expert evaluator. You will be given a question, a reference answer, and a model prediction.
Score the prediction from 0.0 to 1.0 based on how well it matches the reference answer in content and accuracy.

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


def eval_judge(judge_model, prediction, answer, prompt, judge_prompt=None):
    template = judge_prompt if judge_prompt else SCIDOC_JUDGE_PROMPT
    message = template.format(prompt=prompt, answer=answer, prediction=prediction)
    denominator = _extract_score_denominator(template)
    raw = judge_model.generate(message, temperature=0)
    return _parse_judge_response(raw, denominator=denominator)


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
    raw = judge_model.generate(message, temperature=0)
    return _parse_reasoning_response(raw)


# ── Parallel evaluation helper ───────────────────────────────────────────────

_judge_model = None


def _parse_field(raw, fallback):
    """Parse a JSON-serialized field from TSV, with fallback."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return fallback
    return raw if isinstance(raw, (dict, list)) else fallback


def _eval_one_item(item_json):
    """Evaluate a single sample. Called by track_progress_rich.

    Returns (answer_score, reasoning_score, note). ``reasoning_score`` is None
    only when the sample's category is outside REASONING_CATEGORIES. For
    whitelisted categories the reasoning axis is evaluated regardless of
    whether the answer is right — wrong-answer cases are often where reasoning
    quality (and hallucination) matter most for error attribution. Answer and
    reasoning scores are independent axes and are NOT multiplied together.
    """
    item = json.loads(item_json)
    raw_prediction = str(item.get('prediction', ''))
    prediction = _strip_thinking_for_answer(raw_prediction)
    eval_method = item.get('eval_method', 'judge')
    category = str(item.get('category', '') or '')

    answer = _parse_field(item.get('answer', '{}'), item.get('answer', ''))

    judge_prompt = item.get('judge_prompt', '')
    if isinstance(judge_prompt, float) or not judge_prompt:
        judge_prompt = None

    question = str(item.get('question', ''))

    try:
        if eval_method == 'json_match':
            answer_score, note = eval_json_match(prediction, answer, question)
        elif eval_method == 'judge':
            answer_str = (json.dumps(answer, ensure_ascii=False)
                          if isinstance(answer, dict) else str(answer))
            answer_score, note = eval_judge(
                _judge_model, prediction, answer_str, question, judge_prompt)
        elif eval_method == 'exec_match':
            answer_score, note = eval_exec_match(prediction, answer)
        else:
            answer_score, note = 0.0, f"Unknown eval_method: {eval_method}"

        reasoning_score = None
        if category in REASONING_CATEGORIES and _judge_model is not None:
            reasoning_score, reason_note = eval_reasoning(
                _judge_model, raw_prediction, question)
            note = (f"answer={answer_score:.2f}, reasoning={reasoning_score:.2f}; "
                    f"{note}; reasoning: {reason_note}")
    except Exception as e:
        return 0.0, None, f"Eval error: {e}"

    return answer_score, reasoning_score, note


# ── Dataset class ────────────────────────────────────────────────────────────


class SciDocBench(ImageBaseDataset):

    TYPE = 'VQA'

    DATASET_URL = {
        'SciDocBench': 'https://opencompass.openxlab.space/utils/VLMEvalKit/SciDocBench.tsv',
    }
    DATASET_MD5 = {
        'SciDocBench': None,
    }

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
            tgt_path = toliststr(line['image_path'])
            read_ok_flag = [read_ok(x) for x in tgt_path]
            if not all(read_ok_flag):
                tgt_path = [osp.join(self.img_root, x) for x in tgt_path]
        return tgt_path

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        raw_image_path = line.get('image_path', None)
        tgt_path = self.dump_image(line)

        segments = _parse_segments(line.get('segments', ''))
        if segments:
            lookup = _segment_image_lookup(tgt_path)
            if raw_image_path is not None:
                _add_segment_image_aliases(lookup, raw_image_path, tgt_path)
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
        model_name = judge_kwargs.get('model', 'gpt-5.4-mini')

        storage = get_intermediate_file_path(eval_file, f'_{model_name}')
        tmp_file = get_intermediate_file_path(eval_file, f'_{model_name}', 'pkl')
        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        indices = [str(line['index']) for line in lines]
        line_by_index = {str(line['index']): line for line in lines}

        if osp.exists(storage):
            logger.info(f'Scoring file {storage} already exists, will reuse.')
        else:
            _judge_model = build_judge(max_tokens=1024, **judge_kwargs)

            # Serialize each row to JSON for the worker function
            tups = []
            for line in lines:
                item = {}
                for col in data.columns:
                    item[col] = _json_safe_value(line[col])
                tups.append(json.dumps(item, ensure_ascii=False))

            # Load checkpoint and skip already-evaluated items
            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
                logger.info(f'Loaded {len(ans)} cached results from {tmp_file}')
                cache_changed = False
                for sid, cached in list(ans.items()):
                    sanitized, changed = _sanitize_cached_result(
                        cached, line_by_index.get(str(sid)))
                    if changed:
                        ans[sid] = sanitized
                        cache_changed = True
                if cache_changed:
                    dump(ans, tmp_file)
                    logger.info(f'Normalized cached scores in {tmp_file}')

            remaining_tups = [x for x, i in zip(tups, indices) if i not in ans]
            remaining_indices = [i for i in indices if i not in ans]

            if len(remaining_indices):
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
                else:
                    cached, _ = _sanitize_cached_result(cached, line)
                    answer_score, reasoning_score, note = cached
                results.append({
                    'index': sid,
                    'category': line.get('category', ''),
                    'eval_method': line.get('eval_method', ''),
                    'score': answer_score,
                    'reasoning_score': reasoning_score,
                    'eval_note': note,
                })

            result_df = pd.DataFrame(results)
            dump(result_df, storage)

        # Load from storage and aggregate
        result_df = load(storage)
        sanitized_rows = []
        storage_changed = False
        for _, row in result_df.iterrows():
            sid = str(row.get('index', ''))
            cached = (row.get('score', 0.0), row.get('reasoning_score', None),
                      row.get('eval_note', ''))
            sanitized, changed = _sanitize_cached_result(
                cached, line_by_index.get(sid))
            answer_score, reasoning_score, note = sanitized
            row = row.copy()
            row['score'] = answer_score
            row['reasoning_score'] = reasoning_score
            row['eval_note'] = note
            sanitized_rows.append(row)
            storage_changed = storage_changed or changed
        if sanitized_rows:
            result_df = pd.DataFrame(sanitized_rows)
        if storage_changed:
            dump(result_df, storage)
            logger.info(f'Normalized per-sample scoring file {storage}')

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
                result_df['category'].isin(REASONING_CATEGORIES)
                & result_df['reasoning_score'].notna()
            ]
            summary_rows.append({
                'Category': 'Reasoning (whitelist)',
                'Num': len(reasoning_subset),
                'Score': _mean_pct(reasoning_subset['reasoning_score']),
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
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
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
