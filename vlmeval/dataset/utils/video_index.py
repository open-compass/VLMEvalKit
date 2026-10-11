"""Helpers of the Video-Index benchmark: prompts, option permutations of the blind protocol,
the frame rule of the video protocol and the rule scorer (no judge model)."""
import random
import re

import numpy as np

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ANSWER_INSTR = 'Reply with ONLY the option letter (or the exact short answer if no options).'
PROMPT = '{intro}\n\nQuestion: {q}\n{opts}\n' + ANSWER_INSTR
INTRO_FRAMES = 'You are given {n} frame(s) sampled from a video. Answer the question based on these frames.'
INTRO_VIDEO = 'You are given a video. Answer the question based on this video.'
INTRO_BLIND = 'You are given NO frames from the video. Answer the question from the text alone.'
GROUPS = [
    ('perception', 'Perception'),
    ('temporal', 'Temporal'),
    ('spatial_physical', 'Spatial'),
    ('reasoning_knowledge', 'Reasoning'),
]
N_PERM, PERM_SEED = 4, 42

# leading option letter: "B", "(B)", "B.", "B) text", "[b]", "b: text"
_LEAD = re.compile(r'^\s*[\(\[]?([A-Ja-j])[\)\]\.,:]?(?:\s|$)')
# "answer is (B)" / "Answer: B." / "option B": the last occurrence wins
_STATED = re.compile(
    r'(?:answer|option|choice)\s*(?:is|would\s+be)?\s*[:\-]?\s*[\(\[]?([A-Ja-j])[\)\]\.,:]?(?:\s|$)',
    re.IGNORECASE,
)
_STATED_CN = re.compile(r'答案\s*(?:是|为)?\s*[:：]?\s*[\(\[]?([A-Ja-j])(?![A-Za-z])')


def permutations_for(item_id, k, n_perm=N_PERM, seed=PERM_SEED):
    """Option orders of the blind protocol: perm[j] is the index of the original option shown at position j."""
    rng = random.Random(f'{seed}|{item_id}')
    return [rng.sample(range(k), k) for _ in range(n_perm)]


def render_options(texts):
    return '\n'.join(f'{LETTERS[i]}. {t}' for i, t in enumerate(texts))


def _pre(s):
    s = str(s)
    s = s.replace('（', '(').replace('）', ')').replace('：', ':')
    s = s.replace('。', '.').replace('，', ',')
    return s.replace('*', '').replace('#', '')


def _norm_text(s):
    s = str(s).strip().lower()
    s = re.sub(r'[‘’“”`]', "'", s)
    s = re.sub(r'[^\w\s.\-]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.strip('.').strip()


def _flat(t):
    return re.sub(r'\s+', ' ', re.sub(r'[.\-]', ' ', t)).strip()


def extract_letter(reply, options):
    """Option letter named by a reply, or None. Order: the leading letter, a stated answer
    ("the answer is B"), the option whose text the reply repeats. `options` are the option
    texts in the order shown to the model (at most ten)."""
    if reply is None:
        return None
    s = _pre(reply).strip()
    m = _LEAD.match(s)
    if m:
        letter = m.group(1).upper()
        # a bare "A" / "I" followed by more words is usually the article or the pronoun
        has_delim = any(ch in m.group(0) for ch in '()[].,:')
        if has_delim or letter not in ('A', 'I') or len(s.split()) == 1:
            return letter
    ms = list(_STATED.finditer(s)) or list(_STATED_CN.finditer(s))
    if ms:
        return ms[-1].group(1).upper()
    ns = _norm_text(s)
    lettered = [f'{LETTERS[i]}. {t}' for i, t in enumerate(options)]
    for i, opt in enumerate(options):
        if ns and ns in (_norm_text(lettered[i]), _norm_text(opt)):
            return LETTERS[i]
    nsf = _flat(ns)
    hits = []
    for i, opt in enumerate(options):
        ot = _flat(_norm_text(opt))
        if ot and len(ot) >= 3 and f' {ot} ' in f' {nsf} ':
            hits.append(i)
    if len(hits) == 1:
        return LETTERS[hits[0]]
    if re.fullmatch(r'\d{1,2}', s) and int(s) < len(options):
        return LETTERS[int(s)]
    return None


def score_reply(reply, gold, options):
    """1.0 when the reply names the gold option, else 0.0. A reply that names no option is wrong."""
    letter = extract_letter(reply, options)
    return float(letter is not None and letter == str(gold).strip().upper())


def one_fps_indices(n_frames, native_fps, duration, rate=1.0, cap=512):
    """Indices of the stored frames that are sent: the stored frame nearest to every multiple of
    1 / rate seconds of the timeline [0, duration); every stored frame when the video is stored
    below the rate; uniform thinning to at most `cap` frames."""
    if n_frames <= 0:
        return []
    native = native_fps or 2.0
    ts = np.arange(n_frames, dtype=np.float64) / native
    dur = float(duration) if duration and duration == duration else float(ts[-1] + 0.5)
    if n_frames <= dur * rate + 1:
        idx = list(range(n_frames))
    else:
        targets = np.arange(0.0, max(dur, 0.5 / rate), 1.0 / rate)
        idx = sorted(set(int(np.abs(ts - t).argmin()) for t in targets))
    if cap and len(idx) > cap:
        keep = sorted(set(int(round(x)) for x in np.linspace(0, len(idx) - 1, cap)))
        idx = [idx[i] for i in keep]
    return idx


def read_frames(reader, indices):
    """Frames (numpy arrays, RGB) at the given indices of a decord VideoReader, decoded in stream order.
    Random access returns a neighbouring frame on some of the videos; the videos hold at most 1,024
    stored frames, so decoding from the first frame is affordable."""
    wanted = set(int(i) for i in indices)
    if not wanted:
        return []
    frames = {}
    reader.seek(0)
    for i in range(max(wanted) + 1):
        frame = reader.next()
        if i in wanted:
            frames[i] = frame.asnumpy()
    return [frames[int(i)] for i in indices]
