"""DiffSpot — fine-grained visual change detection on web UIs.

Each example is a pair of webpage screenshots (before / after) differing by a
single CSS-level mutation; the model must describe what changed. Scored by an
operator-aware LLM-as-Judge against the structured mutation log.

Paper:   https://arxiv.org/abs/2605.29615
Dataset: https://huggingface.co/datasets/tencent/DiffSpot  (4,400 pairs)
Repo:    https://github.com/Tencent/DiffSpot

Prompt / judge / metrics are ported verbatim from the official DiffSpot release
so VLMEvalKit reproduces the paper leaderboard.
"""
import base64
# Prompt constants below are verbatim from the DiffSpot release; their long
# lines must not be reflowed, so line-length is suppressed for this file.
# flake8: noqa: E501
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from ..smp import dump, load, toliststr
from .image_base import ImageBaseDataset

# ----------------------------------------------------------------------------
# Canonical prompts (verbatim from diffspot/prompts/{vlm_diff,judge}.txt)
# ----------------------------------------------------------------------------

# vlm_diff.txt and vlm_nodiff.txt are intentionally identical (calibrated to
# avoid priming the "is there a change?" prior). Used for all 4,400 items.
VLM_PROMPT = """You are comparing two screenshots of the same web page (Image A and Image B).
Subtle changes may have been made to one or more elements between A and B.

Carefully list **every** difference you can identify. For each difference, state:
  1. Which element changed (be specific: "the 'Sign Up' button in the top-right header")
  2. What changed (color / size / position / text content / visibility / spacing / alignment / etc.)
  3. The direction or magnitude of the change if observable

If you cannot identify any difference, say so explicitly. Do not invent changes.
Output one bullet point per difference, in plain English."""

NO_DIFF_GT_NOTE = (
    "No changes were applied; the two screenshots are independent renderings of the same HTML."
)

JUDGE_PROMPT = """You are a judge evaluating whether a Vision-Language Model (VLM) correctly identified visual differences between two webpage screenshots.

## Ground Truth Changes (structured)
The following changes were actually made to the webpage:
{gt_json}

## Ground Truth Answer (natural language)
This is the correct answer describing the visual changes in human-readable language:
{gt_answer}

## VLM Response
The VLM was shown both screenshots and asked to find all visual differences. Here is the VLM's response:
---
{vlm_response}
---

## Your Task
For each ground truth change, determine if the VLM correctly identified it. Use BOTH the structured GT and the natural language answer as reference.

### Verdict definitions
- "correct": VLM described this change accurately — the location AND nature of change semantically match. Does NOT need to use the same words or technical terms.
- "partial": VLM mentioned something related — noticed a change in the right area but was imprecise about what changed, OR correctly described the visual effect but attributed it to the wrong cause.
- "missed": VLM did not mention this change at all, and nothing in the VLM response can reasonably be linked to this GT change.

### CRITICAL: Match by visual effect, not technical description
A CSS change often manifests as a different visual effect than its technical name suggests. The VLM sees PIXELS, not code — judge by whether the VLM accurately described what it SAW, not whether it matched the technical GT description.

**Core principle: If the VLM's description is a truthful account of what a human would see, it is "correct" — even if the wording differs completely from the GT.**

Example: GT says "border-color changed from yellow-600 to yellow-100" and GT answer says "the line became lighter". But on a dark background, yellow-600 is nearly invisible while yellow-100 (near-white) stands out — so the actual visual effect is "a white line appeared". If the VLM says "a white horizontal line appeared that was not present before", this is **correct** — the VLM accurately described the real visual effect. Do NOT mark this as "partial" just because the GT says "became lighter" while the VLM says "appeared".

You MUST count these as "correct" (NOT "partial" or "missed"):

- **color/border-color change where contrast with background flips visibility** → VLM says "element appeared/disappeared" or "a line/border appeared/disappeared" → **correct** (the color change made a previously invisible element visible, or vice versa)
- **justify/alignment change** → VLM says "elements moved to the right/left" or "items are more spread out" or "spacing between items changed" → **correct**
- **font-weight change** → VLM says "text looks thinner/lighter" or "text appears faded" or "text is bolder/darker" → **correct**
- **letter-spacing change** → VLM says "text is more spread out" or "text looks wider" or "element position shifted" (because wider text shifts layout) → **correct**
- **line-height change** → VLM says "text block is taller/shorter" or "more/less space between lines" or "paragraph looks more compact" → **correct**
- **opacity change** → VLM says "element is faded" or "element looks lighter" or "element appears dimmed" → **correct**. If VLM says "element was removed/disappeared" when opacity is very low → **correct** (this IS what it looks like visually)
- **grid layout column change** → VLM says "items are arranged differently" or "more/fewer items per row" or "cards are narrower/wider" or "headline wraps differently" → **correct**
- **rounded corners change** → VLM says "shape changed" or "box looks more square/circular" → **correct**
- **border-style change** (solid↔dashed↔dotted↔double) → VLM says "border looks different" or "outline style changed" → **correct**
- **position/translate change** → VLM says "element moved/shifted" in approximately the right direction → **correct**
- **position/translate change causing overlap or disappearance** → VLM says "element is missing" or "text is cut off" or "elements are overlapping" or "text is hidden" → **correct** (the VLM accurately described the visual consequence)
- **spacing/padding change** → VLM says "more/less space" or "gap changed" or "element moved" (because margin change shifts position) → **correct**
- **Any mutation causing secondary visual effects** → If a mutation causes text wrapping, element overlap, content being pushed off-screen, or layout reflow, and the VLM describes these secondary effects, mark as **correct** if the description is visually accurate, or **partial** if it's vague.

### When to use "partial" vs "correct"
- **correct**: VLM identified the right area AND described what it saw accurately (even if wording differs from GT)
- **partial**: VLM noticed something in the right area but the description is vague or imprecise (e.g., "something changed in the header" without specifying what)

### When in doubt, prefer "correct" over "partial", and "partial" over "missed"
If the VLM described ANY visual change that could plausibly be caused by the GT mutation (even indirectly), mark it as at least "partial". If the description matches the actual visual effect (even if not the technical cause), mark it as "correct". This includes cases where:
- The VLM describes a CONSEQUENCE of the change (e.g., "text overlaps" when position was shifted) → **correct**
- The VLM says an element "appeared" or "disappeared" due to a color/opacity/visibility change → **correct**
- The VLM identifies the correct AREA of change but is vague about what changed → **partial**

Also check for hallucinations — differences the VLM claimed to see that don't match ANY ground truth change.

Return ONLY valid JSON (no markdown fences):
{
  "mutations": [
    {"gt": "description of GT change", "type": "mutation_type", "verdict": "correct/partial/missed", "vlm_match": "relevant quote from VLM response or empty"}
  ],
  "hallucinations": ["description of hallucinated difference"],
  "summary": {"correct": 0, "partial": 0, "missed": 0, "hallucinated": 0}
}"""

JUDGE_VERSION = "v2.0"
HF_DATASET_ID = "tencent/DiffSpot"


# ----------------------------------------------------------------------------
# Judge client (gateway, OpenAI-compatible). Mirrors diffspot/judge.py exactly.
# ----------------------------------------------------------------------------

_judge_clients = {}
_judge_lock = threading.Lock()


def _judge_client():
    tid = threading.current_thread().ident
    if tid not in _judge_clients:
        import httpx
        from openai import OpenAI
        with _judge_lock:
            if tid not in _judge_clients:
                _judge_clients[tid] = OpenAI(
                    base_url=os.environ.get('OPENAI_API_BASE', os.environ.get('OPENAI_BASE_URL')),
                    api_key=os.environ.get('OPENAI_API_KEY'),
                    http_client=httpx.Client(trust_env=False),
                    timeout=httpx.Timeout(timeout=1200.0, connect=10.0, read=1200.0),
                )
    return _judge_clients[tid]


def _parse_judge_json(raw):
    raw = (raw or '').strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[1] if '\n' in raw else raw
        if raw.endswith('```'):
            raw = raw[:-3]
        if raw.startswith('json'):
            raw = raw[4:]
    raw = raw.strip()
    start, end = raw.find('{'), raw.rfind('}')
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return json.loads(raw)


def _judge_one(gt_mutations, gt_answer, vlm_response, model, max_retries=3):
    gt_json = json.dumps(gt_mutations, indent=2, ensure_ascii=False)
    prompt = (
        JUDGE_PROMPT
        .replace('{gt_json}', gt_json)
        .replace('{gt_answer}', gt_answer)
        .replace('{vlm_response}', vlm_response)
    )
    client = _judge_client()
    # Optional gateway routing headers / reasoning effort — configured purely
    # via environment so the published code carries no internal values.
    create_kwargs = dict(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=int(os.environ.get('DIFFSPOT_JUDGE_MAX_TOKENS', 16384)),
        temperature=0.0,
    )
    # Optional routing headers for self-hosted gateways — supplied entirely via
    # env as a JSON template (a "{model}" placeholder is substituted), so the
    # source contains no endpoint-specific header names or values.
    hdr_tmpl = os.environ.get('DIFFSPOT_API_EXTRA_HEADERS')
    if hdr_tmpl:
        create_kwargs['extra_headers'] = json.loads(hdr_tmpl.replace('{model}', model))
    effort = os.environ.get('DIFFSPOT_JUDGE_REASONING_EFFORT', 'high')
    if effort:
        create_kwargs['extra_body'] = {'reasoning_effort': effort}
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**create_kwargs)
            return _parse_judge_json(resp.choices[0].message.content or '')
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
        except Exception as e:  # transient API / network
            last_err = e
            import time
            time.sleep(2.0 ** attempt)
    return {'mutations': [], 'hallucinations': [], 'summary': {}, '_judge_error': str(last_err)}


def _reduce(label, is_no_diff):
    mutations = label.get('mutations', []) or []
    hallucinations = label.get('hallucinations', []) or []
    n_correct = sum(1 for m in mutations if m.get('verdict') == 'correct')
    n_halluc = len(hallucinations)
    return {
        'is_true_positive': (not is_no_diff) and n_correct > 0,
        'is_true_negative': is_no_diff and n_halluc == 0,
        'n_correct_mutations': n_correct,
        'n_hallucinations': n_halluc,
    }


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class DiffSpot(ImageBaseDataset):

    TYPE = 'VQA'
    # load_data is overridden to read the published HF dataset directly, so no
    # TSV is hosted; these are placeholders to satisfy the registry.
    DATASET_URL = {'DiffSpot': ''}
    DATASET_MD5 = {'DiffSpot': ''}

    @classmethod
    def supported_datasets(cls):
        return ['DiffSpot']

    def load_data(self, dataset):
        """Load tencent/DiffSpot from HF and adapt to the internal DataFrame.

        image -> [before_b64, after_b64] (dump_image handles the 2-image list).
        """
        from datasets import load_dataset
        ds = load_dataset(HF_DATASET_ID, split='test')
        # Optional debug subset for a fast end-to-end smoke (off by default).
        limit = int(os.environ.get('DIFFSPOT_LIMIT', 0))
        rows = []
        for i, ex in enumerate(ds):
            if limit and i >= limit:
                break
            task_type = ex.get('task_type', '')
            difficulty = ex.get('difficulty', '')
            split = 'no_diff' if task_type == 'no_diff' else difficulty
            raw_mut = ex.get('mutation_dicts_json') or []
            gt_mutations = []
            for s in raw_mut:
                try:
                    gt_mutations.append(json.loads(s))
                except Exception:
                    pass
            operator = gt_mutations[0].get('type') if gt_mutations else None
            rows.append({
                'index': i,
                'id': ex.get('id', str(i)),
                # store as a JSON-encoded list of two base64 PNGs; ImageBaseDataset
                # expects string cells and turns this back into a list (via toliststr).
                'image': json.dumps([_pil_to_b64(ex['image_before']), _pil_to_b64(ex['image_after'])]),
                'question': VLM_PROMPT,
                'answer': ex.get('ground_truth_diff', '') or '',
                'split': split,
                'task_type': task_type,
                'difficulty': difficulty,
                'operator': operator or '',
                'gt_mutations_json': json.dumps(gt_mutations, ensure_ascii=False),
                'domain': ex.get('domain', '') or '',
            })
        return pd.DataFrame(rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)  # [before_path, after_path]
        # Match the official runner order: text first, then before, then after.
        msgs = [dict(type='text', value=line['question'])]
        msgs += [dict(type='image', value=p) for p in toliststr(tgt_path)]
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        # Official DiffSpot judge is gpt-oss-120b (reasoning_effort=high). The
        # concrete model id / endpoint are supplied via --judge or env vars.
        judge_model = judge_kwargs.get('model') or os.environ.get(
            'DIFFSPOT_JUDGE_MODEL', 'gpt-oss-120b')
        nproc = int(judge_kwargs.get('nproc', os.environ.get('DIFFSPOT_JUDGE_NPROC', 64)))

        pred_df = load(eval_file)
        meta = self.data.set_index('index')
        preds = {}
        for _, r in pred_df.iterrows():
            preds[int(r['index'])] = r.get('prediction', '') or ''

        tasks = []
        for idx, m in meta.iterrows():
            is_no_diff = (m['split'] == 'no_diff')
            gt_mut = json.loads(m['gt_mutations_json']) if not is_no_diff else []
            gt_ans = NO_DIFF_GT_NOTE if is_no_diff else (m['answer'] or '')
            tasks.append({
                'index': int(idx), 'id': m['id'], 'split': m['split'],
                'operator': m['operator'], 'is_no_diff': is_no_diff,
                'gt_mutations': gt_mut, 'gt_answer': gt_ans,
                'prediction': preds.get(int(idx), ''),
            })

        judged = []
        with ThreadPoolExecutor(max_workers=nproc) as pool:
            futs = {
                pool.submit(_judge_one, t['gt_mutations'], t['gt_answer'],
                            t['prediction'], judge_model): t
                for t in tasks
            }
            done = 0
            for fut in as_completed(futs):
                t = futs[fut]
                label = fut.result()
                red = _reduce(label, t['is_no_diff'])
                judged.append({**t, **red, '_label': label})
                done += 1
                if done % 200 == 0:
                    print(f'[DiffSpot judge] {done}/{len(tasks)}', flush=True)

        # dump per-item judged records for debugging / audit.
        # Derive output paths from the real extension (eval_file may be
        # .xlsx / .tsv / .json depending on the configured storage format).
        base = os.path.splitext(eval_file)[0]
        detail_file = base + '_diffspot_judged.jsonl'
        with open(detail_file, 'w') as f:
            for j in judged:
                row = {k: v for k, v in j.items() if k != 'gt_mutations'}
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

        report = _aggregate(judged)
        score_df = pd.DataFrame([report])
        dump(score_df, base + '_acc.csv')
        return score_df


def _aggregate(judged):
    from collections import defaultdict
    n_total = len(judged)
    n_tp = n_tn = n_has_diff = n_no_diff = 0
    tier_tp, tier_n = defaultdict(int), defaultdict(int)
    op_tp, op_n = defaultdict(int), defaultdict(int)
    n_halluc_no_diff = 0
    for it in judged:
        split = it['split']
        if split == 'no_diff':
            n_no_diff += 1
            if it['is_true_negative']:
                n_tn += 1
            else:
                n_halluc_no_diff += 1
        else:
            n_has_diff += 1
            tier_n[split] += 1
            if it['is_true_positive']:
                n_tp += 1
                tier_tp[split] += 1
            op = it.get('operator')
            if op:
                op_n[op] += 1
                if it['is_true_positive']:
                    op_tp[op] += 1

    def pct(a, b):
        return round(100.0 * a / b, 2) if b else 0.0

    report = {
        'overall_accuracy': pct(n_tp + n_tn, n_total),
        'diff_overall_recall': pct(n_tp, n_has_diff),
        'no_diff_specificity': pct(n_tn, n_no_diff),
        'easy_recall': pct(tier_tp.get('easy', 0), tier_n.get('easy', 0)),
        'med_recall': pct(tier_tp.get('medium', 0), tier_n.get('medium', 0)),
        'hard_recall': pct(tier_tp.get('hard', 0), tier_n.get('hard', 0)),
        'no_diff_hallucination_rate': pct(n_halluc_no_diff, n_no_diff),
        'n_total': n_total, 'n_has_diff': n_has_diff, 'n_no_diff': n_no_diff,
        'n_tp': n_tp, 'n_tn': n_tn,
    }
    for op in sorted(op_n):
        report[f'op_{op}_recall'] = pct(op_tp[op], op_n[op])
    return report


def _pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')
