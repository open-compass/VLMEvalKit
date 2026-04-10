import json
import os
import os.path as osp
import re
import subprocess
import tempfile

import pandas as pd

from vlmeval.smp import (decode_base64_to_image_file, dump, get_intermediate_file_path, get_logger,
                         load, read_ok, toliststr)
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils.judge_util import build_judge

logger = get_logger(__name__)

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


# ── Evaluation methods ───────────────────────────────────────────────────────


def eval_json_match(prediction: str, answer: dict) -> tuple:
    pred = _safe_json_loads(prediction)
    if pred is None:
        return 0.0, "Failed to parse prediction as JSON"
    if not isinstance(pred, dict):
        return 0.0, "Prediction is not a JSON object"

    def values_match(key, pred_val, ans_val):
        p, a = str(pred_val).strip(), str(ans_val).strip()
        suffix = key.rsplit(".", 1)[-1] if "." in key else key
        if suffix == "location":
            return normalize_location(p) == normalize_location(a)
        if suffix in ("models", "tasks"):
            return normalize_roles(p) == normalize_roles(a)
        if key.startswith("["):
            return normalize_roles(p) == normalize_roles(a)
        if "\\" in key:
            return normalize_equation(p) == normalize_equation(a)
        pn, an = normalize_number(p), normalize_number(a)
        if pn is not None and an is not None:
            return pn == an
        return p == a

    matched = sum(1 for k, v in answer.items() if values_match(k, pred.get(k, ""), v))
    total = len(answer)
    score = matched / total if total > 0 else 0.0
    return score, f"{matched}/{total} keys matched"


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


def _parse_judge_response(raw: str) -> tuple:
    try:
        result = json.loads(raw)
        return (
            float(result.get("score", result.get("reasoning_score", 0))),
            result.get("eval_note", result.get("reasoning_note", ""))
        )
    except Exception:
        pass
    try:
        result = json.loads(_repair_json_escapes(raw))
        return (
            float(result.get("score", result.get("reasoning_score", 0))),
            result.get("eval_note", result.get("reasoning_note", ""))
        )
    except Exception:
        pass
    m = re.search(r'"(?:score|reasoning_score)"\s*:\s*([0-9.]+)', raw)
    if m:
        return float(m.group(1)), f"[score extracted via regex] {raw}"
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
You are verifying whether a model's reasoning process is correct, independent of its final answer.

Reference reasoning (ground truth derivation):
{reference_reasoning}

Model's full output:
{prediction}

Evaluate ONLY the reasoning process. Check for:
- Factual errors (wrong numbers, wrong table/figure references, wrong page citations)
- Hallucinated evidence (citing data or results that don't exist in the paper)
- Incorrect logical steps (wrong causal chains, flawed deductions)
- Missing critical reasoning steps (skipping key intermediate steps that are necessary for the conclusion)

Do NOT penalize for:
- Different wording or phrasing of the same correct reasoning
- Additional correct reasoning not in the reference
- Minor formatting differences

Respond with a JSON object only, no extra text:
{{"reasoning_score": <float between 0.0 and 1.0>, "reasoning_note": "<brief explanation, no newlines>"}}"""


def eval_judge(judge_model, prediction, answer, prompt, judge_prompt=None):
    template = judge_prompt if judge_prompt else SCIDOC_JUDGE_PROMPT
    message = template.format(prompt=prompt, answer=answer, prediction=prediction)
    raw = judge_model.generate(message, temperature=0)
    return _parse_judge_response(raw)


def eval_reasoning(judge_model, prediction, reasoning_ref):
    message = SCIDOC_REASONING_CHECK_PROMPT.format(
        reference_reasoning=reasoning_ref,
        prediction=prediction
    )
    raw = judge_model.generate(message, temperature=0)
    return _parse_judge_response(raw)


# ── Parallel evaluation helper ───────────────────────────────────────────────

_judge_model = None


def _parse_field(raw, fallback):
    """Parse a JSON-serialized field from TSV, with fallback."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return fallback
    return raw if isinstance(raw, dict) else fallback


def _eval_one_item(item_json):
    """Evaluate a single sample. Called by track_progress_rich."""
    item = json.loads(item_json)
    prediction = str(item.get('prediction', ''))
    eval_method = item.get('eval_method', 'judge')

    answer = _parse_field(item.get('answer', '{}'), item.get('answer', ''))
    answer_note = _parse_field(item.get('answer_note', '{}'), {})

    judge_prompt = item.get('judge_prompt', '')
    if isinstance(judge_prompt, float) or not judge_prompt:
        judge_prompt = None

    try:
        if eval_method == 'json_match':
            score, note = eval_json_match(prediction, answer)
        elif eval_method == 'judge':
            question = item.get('question', '')
            answer_str = (json.dumps(answer, ensure_ascii=False)
                          if isinstance(answer, dict) else str(answer))
            score, note = eval_judge(
                _judge_model, prediction, answer_str, question, judge_prompt)
        elif eval_method == 'exec_match':
            score, note = eval_exec_match(prediction, answer)
        else:
            score, note = 0.0, f"Unknown eval_method: {eval_method}"

        # Reasoning verification
        reasoning_ref = (answer_note.get('reasoning')
                         if isinstance(answer_note, dict) else None)
        if score > 0 and reasoning_ref and _judge_model is not None:
            reason_score, reason_note = eval_reasoning(
                _judge_model, prediction, reasoning_ref)
            original_score = score
            score = score * reason_score
            note = (f"answer={original_score:.2f}, reasoning={reason_score:.2f}, "
                    f"final={score:.2f}; {note}; reasoning: {reason_note}")
    except Exception as e:
        score, note = 0.0, f"Eval error: {e}"

    return score, note


# ── Dataset class ────────────────────────────────────────────────────────────


class SciDocBench(ImageBaseDataset):

    TYPE = 'VQA'

    DATASET_URL = {
        'SciDocBench': 'https://opencompass.openxlab.space/utils/VLMEvalKit/SciDocBench.tsv',
    }
    DATASET_MD5 = {
        'SciDocBench': '8947fc96f82c825bd6c9c1167821cd5b',
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

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

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
        model_name = judge_kwargs.get('model', 'gpt-4o-mini')

        storage = get_intermediate_file_path(eval_file, f'_{model_name}')
        tmp_file = get_intermediate_file_path(eval_file, f'_{model_name}', 'pkl')

        if osp.exists(storage):
            logger.info(f'Scoring file {storage} already exists, will reuse.')
        else:
            data = load(eval_file)
            _judge_model = build_judge(max_tokens=1024, **judge_kwargs)

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            indices = [str(line['index']) for line in lines]

            # Serialize each row to JSON for the worker function
            tups = []
            for line in lines:
                item = {}
                for col in data.columns:
                    val = line[col]
                    if isinstance(val, float) and pd.isna(val):
                        val = ''
                    item[col] = val
                tups.append(json.dumps(item, ensure_ascii=False))

            # Load checkpoint and skip already-evaluated items
            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
                logger.info(f'Loaded {len(ans)} cached results from {tmp_file}')

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

            # Build result rows in original order
            results = []
            for line in lines:
                sid = str(line['index'])
                score, note = ans.get(sid, (0.0, 'Not evaluated'))
                results.append({
                    'index': sid,
                    'category': line.get('category', ''),
                    'eval_method': line.get('eval_method', ''),
                    'score': score,
                    'eval_note': note,
                })

            result_df = pd.DataFrame(results)
            dump(result_df, storage)

        # Load from storage and aggregate
        result_df = load(storage)

        summary_rows = []
        overall_score = result_df['score'].mean() * 100
        summary_rows.append({
            'Category': 'Overall',
            'Num': len(result_df),
            'Score': round(overall_score, 2),
        })

        for method in sorted(result_df['eval_method'].unique()):
            subset = result_df[result_df['eval_method'] == method]
            summary_rows.append({
                'Category': f'method:{method}',
                'Num': len(subset),
                'Score': round(subset['score'].mean() * 100, 2),
            })

        for cat in sorted(result_df['category'].unique()):
            subset = result_df[result_df['category'] == cat]
            summary_rows.append({
                'Category': cat,
                'Num': len(subset),
                'Score': round(subset['score'].mean() * 100, 2),
            })

        summary = pd.DataFrame(summary_rows)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(summary, score_file)
        logger.info(f'SciDocBench evaluation finished. Results saved to {score_file}')
        logger.info(f'\n{summary.to_string(index=False)}')
        return summary
