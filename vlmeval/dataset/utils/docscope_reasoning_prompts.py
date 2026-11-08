"""Shared prompt + parsing helpers for the reasoning-with-citations pipeline.

Citation format (v2 — with global / doc page distinction):

    [page=<int>, doc_page="<string_or_none>", bbox=[x1, y1, x2, y2]]

The model is told that each page image is wrapped by GLOBAL PAGE text markers
so the `page` field refers to the 1-indexed global order of images, while
`doc_page` captures whatever page label is printed inside the page content
(Arabic, Roman, "A-3", or "none" if absent).
"""

# flake8: noqa

from __future__ import annotations
import json
import re
from dataclasses import asdict, dataclass

try:
    import icu  # type: ignore
    _HAS_ICU = True
except ImportError:
    icu = None  # type: ignore
    _HAS_ICU = False

INFER_SYSTEM_PROMPT = """You are an expert document QA system. You are given page images of a PDF document and a question about the document. Each image is wrapped by text markers indicating its global page number.

<hard_constraints>
BEFORE YOU ANSWER, INTERNALIZE THIS CONTRACT. VIOLATING ANY RULE MEANS COMPLETE FAILURE:

1. ZERO TOLERANCE FOR MISSING CITATIONS: EVERY SINGLE sentence stating a fact from the document MUST end with EXACTLY ONE formal citation. No exceptions. No excuses.
2. CITATION FORMAT: The citation MUST strictly match this format: `[page=N, doc_page="...", bbox=[x1, y1, x2, y2]]`
3. SYNTAX ALERT: Pay close attention to the closing brackets. The citation MUST end with TWO right brackets `]]` and then the period. (Correct: `0.512]].` / Incorrect: `0.512].`)
4. FORBIDDEN: NEVER use natural language to cite pages (e.g., DO NOT write "on page 5", "in Table 7 on global page 51", or "image 58"). You MUST use the bracket format.
5. MANDATORY FINAL ANSWER TAG: You MUST conclude your response with a concise final answer wrapped STRICTLY and EXACTLY as:
   `<answer> your final answer </answer>`
6. NO MARKDOWN: Do not use headings, bold, italics, lists, or tables in your reasoning or answer. Plain prose only (fenced code blocks and LaTeX math are allowed when strictly necessary).
</hard_constraints>

## Page Numbering Rule
- Each page image is preceded and followed by a text marker (e.g., `=== GLOBAL PAGE X (start) ===`). `X` is the **global page number** (integer, starting from 1).
- The document itself may print its own page number inside the page (e.g., "12", "iv"). This is the **document page number**.
- The `page` field in your citation MUST use the **global page number**.
- The `doc_page` field MUST use the **document page number** (as a string). If not visible, set `doc_page="none"`.

## CRITICAL Citation Requirement
- **Fact vs. Reasoning**: A sentence is "fact-bearing" if it contains a number, date, name, quote, or claim from the document. A sentence is "pure reasoning" ONLY if it's arithmetic/logical inference based on already-cited facts. EVERY fact-bearing sentence requires a citation.
- **Writing Rule (One Citation Per Sentence)**: After stating ONE fact from the document, immediately close the sentence with the citation and a period. Then start a NEW sentence for the next fact. Prefer short sentences.
- **Coordinates**: The `bbox` uses **normalized coordinates** `[x1, y1, x2, y2]` in the range `[0, 1]`.
- **Uncertainty Fallback**: If you cannot precisely localize the region, provide a conservative bounding box that safely contains the relevant content. NEVER omit a citation due to bbox uncertainty.

## If Unanswerable (CRITICAL)
If the question cannot be answered from the document, briefly explain why (still citing any relevant regions you checked, such as the table of contents). Then, you MUST output exactly:
`<answer> Unanswerable </answer>`
Do NOT output variations like `<answer> I cannot answer </answer>`.

## Good & Bad Examples

BAD EXAMPLE 1 (Natural language citation - FORBIDDEN):
The revenue grew 13% in 2023 based on Table 1 on page 5.

BAD EXAMPLE 2 (Missing the second closing bracket `]` - FORBIDDEN):
Revenue was $5.2B in 2023 [page=5, doc_page="3", bbox=[0.412, 0.215, 0.687, 0.248].

BAD EXAMPLE 3 (Multiple facts, citation in the middle - FORBIDDEN):
Revenue was $5.2B in 2023 [page=5, doc_page="3", bbox=[0.41, 0.21, 0.68, 0.24]] and $4.6B in 2022.

GOOD EXAMPLE:
Let me analyze the financial data. The document reports a total revenue of $5.2 billion for fiscal year 2023 [page=5, doc_page="3", bbox=[0.412, 0.215, 0.687, 0.248]]. The total revenue for fiscal year 2022 was $4.6 billion [page=5, doc_page="3", bbox=[0.412, 0.252, 0.687, 0.285]]. The difference is $0.6B, which corresponds to approximately a 13% increase. This growth is confirmed in the management commentary [page=4, doc_page="none", bbox=[0.085, 0.612, 0.915, 0.648]].
<answer> The company's total revenue in 2023 was $5.2 billion, up approximately 13% from $4.6 billion in 2022. </answer>

## Output Protocol
1. Output your detailed reasoning process step by step.
2. Pre-output Self-Check (Perform silently):
   - Did I use the formal `[page=...]` format instead of saying "on page X"?
   - Does EVERY bbox array end with `]]`?
   - Does EVERY fact-bearing sentence have exactly ONE citation at the end?
   - Is my final answer explicitly wrapped in `<answer>` tags?
3. Output your final concise answer wrapped STRICTLY as:
`<answer> your final answer </answer>`
"""

INFER_SIMPLE_PROMPT = """You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section."""


# ── Judge ─────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for document question answering. You are given \
a question, a gold answer, and a model-predicted answer. Determine whether \
the predicted answer is semantically consistent with the gold answer — i.e., \
they convey the same core factual content, even if they differ in formatting, \
wording, or level of detail.

Rules:
- CONSISTENT means the predicted answer agrees with the gold answer on the core factual content.
- INCONSISTENT means the predicted answer provides different / contradictory factual information, is missing essential information, or refuses unjustifiably.
- Minor formatting differences (e.g., "21%" vs "21", "14 years" vs "14", "$1,000" vs "1000 dollars") are CONSISTENT.
- Different numbers, different entities, contradictory conclusions, wrong units, or "Unanswerable" when a real answer is given are INCONSISTENT.

OUTPUT FORMAT (strict JSON, no markdown fences):
{"consistent": true or false, "reasoning": "<brief explanation>"}
"""

JUDGE_USER_TEMPLATE = """\
Question: {question}

Gold answer: {gold}

Model answer: {pred}

Is the model answer semantically consistent with the gold answer?"""


# ── Input markers ─────────────────────────────────────────────────────────────

def page_marker_start(i: int) -> str:
    return f"=== GLOBAL PAGE {i} (start) ==="


def page_marker_end(i: int) -> str:
    return f"=== GLOBAL PAGE {i} (end) ==="


# ── Parsing ───────────────────────────────────────────────────────────────────

# Citation regex:
#   [page=<int>, doc_page="<str>", bbox=[<f>, <f>, <f>, <f>]]
# doc_page value can be quoted ("12", "iv", "none") or bare (none / 12)
_CITATION_RE = re.compile(
    r"\[\s*page\s*=\s*(\d+)\s*,\s*"
    r"doc_page\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s,\]]+))\s*,\s*"
    r"bbox\s*=\s*\[\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)\s*\]\s*\]",
    re.IGNORECASE,
)

# Loose / "concatenated" form — covers malformed runs like:
#
#   [page=9, doc_page="1", bbox=[77, 553, 159, 568],
#    [page=9, doc_page="1", bbox=[77, 601, 159, 617],
#    [page=12, doc_page="4", bbox=[186, 121, 452, 137]]]
#
# i.e. each item starts with `[page=N, doc_page=..., bbox=[a,b,c,d]` but the
# closing `]]` may be replaced by a single `]` followed by either `,` (next
# item in the run) or end-of-text. doc_page may also be missing entirely.
# Always pulled AFTER the strict pattern, only over text the strict pattern
# didn't already mask.
_LOOSE_CITATION_RE = re.compile(
    r"\[\s*page\s*=\s*(\d+)\s*"
    r"(?:,\s*doc_page\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s,\]]+))\s*)?,\s*"
    r"bbox\s*=\s*\[\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)\s*,\s*"
    r"([-+]?\d*\.?\d+)\s*\]",
    re.IGNORECASE,
)

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)

# Protection regions (masked before ICU sees the text)
_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
_BLOCK_LATEX_RE = re.compile(r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_INLINE_LATEX_RE = re.compile(r"\\\([\s\S]*?\\\)")

# Private-use-area placeholder ranges (one char per masked original char)
PUA_CITATION = "\uE000"
PUA_CODE_BLOCK = "\uE001"
PUA_BLOCK_LATEX = "\uE002"
PUA_INLINE_CODE = "\uE003"
PUA_INLINE_LATEX = "\uE004"
_ALL_PUA = "\uE000\uE001\uE002\uE003\uE004"


@dataclass
class Citation:
    page: int            # global page number
    doc_page: str        # as printed on the page; "none" if absent
    bbox: list[float]
    sentence: str
    span_start: int
    span_end: int
    is_normalized: bool = True   # bbox values all in [0, 1.5]

    def as_dict(self) -> dict:
        return asdict(self)


def _bbox_is_normalized(bbox: list[float]) -> bool:
    if not bbox or len(bbox) != 4:
        return False
    return all(-0.01 <= v <= 1.5 for v in bbox)


@dataclass
class _Mask:
    start: int
    end: int
    kind: str              # "citation" | "code_block" | "block_latex" | "inline_code" | "inline_latex"
    original: str
    meta: dict | None = None   # for citation: parsed metadata


def extract_answer(text: str) -> str | None:
    matches = list(_ANSWER_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


# ── Masking stage ─────────────────────────────────────────────────────────────

def _parse_citation_match(m: re.Match) -> dict | None:
    try:
        page = int(m.group(1))
    except (ValueError, TypeError):
        return None
    doc_page = m.group(2) or m.group(3) or m.group(4) or "none"
    doc_page = doc_page.strip() or "none"
    try:
        bbox = [float(m.group(i)) for i in range(5, 9)]
    except (ValueError, TypeError):
        return None
    return {
        "page": page,
        "doc_page": doc_page,
        "bbox": bbox,
        "is_normalized": _bbox_is_normalized(bbox),
    }


def _apply_mask(text: str, pattern: re.Pattern, pua_char: str, kind: str,
                masks: list[_Mask],
                meta_factory=None) -> str:
    """Replace every non-overlapping, non-already-masked match with equal-length PUA."""
    out = list(text)
    for m in pattern.finditer(text):
        s, e = m.start(), m.end()
        # Skip if any char in this span is already a PUA placeholder
        segment = text[s:e]
        if any(ch in _ALL_PUA for ch in segment):
            continue
        meta = meta_factory(m) if meta_factory else None
        if meta_factory and meta is None:
            # factory rejected this match (e.g., malformed citation)
            continue
        masks.append(_Mask(start=s, end=e, kind=kind, original=segment, meta=meta))
        for i in range(s, e):
            out[i] = pua_char
    return "".join(out)


def mask_regions(text: str) -> tuple[str, list[_Mask]]:
    """Mask citations / fenced code / block latex / inline code / inline latex
    with equal-length PUA placeholders.

    Order matters: mask the outermost / least-ambiguous regions first so later
    patterns do not touch already-masked content.
    """
    masks: list[_Mask] = []
    masked = text

    masked = _apply_mask(masked, _CITATION_RE, PUA_CITATION, "citation",
                         masks, meta_factory=_parse_citation_match)
    # Loose / concatenated form — only fires on text the strict pattern didn't
    # cover (because _apply_mask skips spans containing PUA placeholders).
    masked = _apply_mask(masked, _LOOSE_CITATION_RE, PUA_CITATION, "citation",
                         masks, meta_factory=_parse_citation_match)
    masked = _apply_mask(masked, _FENCED_CODE_RE, PUA_CODE_BLOCK, "code_block",
                         masks)
    masked = _apply_mask(masked, _BLOCK_LATEX_RE, PUA_BLOCK_LATEX, "block_latex",
                         masks)
    masked = _apply_mask(masked, _INLINE_CODE_RE, PUA_INLINE_CODE, "inline_code",
                         masks)
    masked = _apply_mask(masked, _INLINE_LATEX_RE, PUA_INLINE_LATEX, "inline_latex",
                         masks)

    masks.sort(key=lambda m: m.start)
    return masked, masks


# ── Abbreviation protection ───────────────────────────────────────────────────
# ICU's default English sentence rules treat "U.S." / "Dr." / "Fig." as sentence
# terminators. PyICU does not expose FilteredBreakIterator, so we protect the
# dot inside a known abbreviation list by swapping it with U+00B7 (middle dot)
# before ICU sees the text, then swap it back after splitting.

_ABBREV_DOT = "\u00B7"

_ABBREV_SINGLE = {
    # Titles
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Rev", "Hon", "Jr", "Sr", "St",
    # Generic
    "e.g", "i.e", "etc", "vs", "cf", "al", "approx", "est", "ca",
    # Companies / orgs
    "Inc", "Ltd", "Co", "Corp", "LLC", "Bros",
    # Degrees
    "Ph.D", "M.D", "B.A", "M.A", "B.S", "M.S",
    # Months (short)
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept",
    "Oct", "Nov", "Dec",
    # Figures / refs / units in text
    "Fig", "Figs", "Vol", "No", "Nos", "Ch", "Sec", "App", "Eq",
    "Tab", "pp", "p", "ed", "Eds", "eds", "n.d",
}

# Multi-dot abbreviations like U.S., U.S.A., U.K., E.U., Ph.D., M.D. — captured
# via a structural pattern "(Letter\.){2,}".
_MULTI_DOT_ABBREV_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}")

# Single-dot abbreviation pattern, built once from the list above.
_SINGLE_ABBREV_RE = re.compile(
    r"\b(?:" + "|".join(sorted(map(re.escape, _ABBREV_SINGLE), key=len, reverse=True)) + r")\.",
    re.IGNORECASE,
)


def _protect_abbreviations(text: str) -> str:
    """Replace `.` inside known abbreviations with U+00B7 so ICU does not split."""
    def _sub(m: re.Match) -> str:
        return m.group(0).replace(".", _ABBREV_DOT)

    text = _MULTI_DOT_ABBREV_RE.sub(_sub, text)
    text = _SINGLE_ABBREV_RE.sub(_sub, text)
    return text


def _restore_abbreviations(text: str) -> str:
    return text.replace(_ABBREV_DOT, ".")


# ── Sentence splitting ────────────────────────────────────────────────────────

def _icu_split_sentences(text: str, locale: str = "en_US"
                         ) -> list[tuple[int, int]]:
    bi = icu.BreakIterator.createSentenceInstance(icu.Locale(locale))
    bi.setText(text)
    spans: list[tuple[int, int]] = []
    start = bi.first()
    for end in bi:
        spans.append((start, end))
        start = end
    return spans


def _regex_split_sentences(text: str) -> list[tuple[int, int]]:
    """Very rough fallback when ICU is unavailable."""
    spans: list[tuple[int, int]] = []
    s = 0
    term_re = re.compile(r"(?<!\d)[.!?。！？](?!\d)|\n{2,}")
    for m in term_re.finditer(text):
        spans.append((s, m.end()))
        s = m.end()
    if s < len(text):
        spans.append((s, len(text)))
    return spans


def split_sentences(text: str, locale: str = "en_US"
                    ) -> list[tuple[int, int]]:
    if _HAS_ICU:
        return _icu_split_sentences(text, locale)
    return _regex_split_sentences(text)


# ── Citation → sentence attribution ───────────────────────────────────────────

def _restore_non_citation(rendered: str, masks: list[_Mask],
                          slice_start: int) -> str:
    """Given a slice of the masked text, restore code/latex placeholders back to
    original characters (but keep citation placeholders as empty — they will be
    stripped below)."""
    chars = list(rendered)
    for mk in masks:
        if mk.kind == "citation":
            continue
        rel_start = mk.start - slice_start
        rel_end = mk.end - slice_start
        if rel_end <= 0 or rel_start >= len(chars):
            continue
        # Overlap range within this slice
        overlap_s = max(0, rel_start)
        overlap_e = min(len(chars), rel_end)
        # Replace placeholder chars with the corresponding original slice
        orig_s = overlap_s - rel_start
        orig_e = overlap_e - rel_start
        # In-place splice
        original_piece = mk.original[orig_s:orig_e]
        chars[overlap_s:overlap_e] = list(original_piece)
    return "".join(chars)


def _clean_sentence(sentence_text: str) -> str:
    # Strip all citation placeholders (already restored code/latex at this point)
    cleaned = sentence_text.replace(PUA_CITATION, "")
    # Strip leading list/heading markers that might remain
    cleaned = re.sub(r"^\s*(?:[-*+•>]|\d+[.)]|[a-zA-Z][.)])\s+", "", cleaned, count=1)
    cleaned = re.sub(r"^\s*#{1,6}\s+", "", cleaned, count=1)
    # Collapse whitespace, then remove whitespace that appears right before
    # terminal/clause punctuation (a common artefact of stripping a citation
    # placeholder that stood between the last word and its period).
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([.!?。！？,，;；:：)\]\}])", r"\1", cleaned)
    return cleaned


def _attribute_and_render(
    raw: str,
    masked: str,
    masks: list[_Mask],
    sentence_spans: list[tuple[int, int]],
) -> list[Citation]:
    # Filter citation masks (preserve insertion order)
    citation_masks = [m for m in masks if m.kind == "citation"]

    # Build sentence texts:
    #   - start from the MASKED slice (still has PUA_CITATION and PUA_CODE/LATEX)
    #   - restore code/latex placeholders back to their original text
    #   - _clean_sentence() then strips the remaining PUA_CITATION characters
    sentence_texts: list[str] = []
    for (s, e) in sentence_spans:
        masked_slice = masked[s:e]
        restored = _restore_non_citation(masked_slice, masks, s)
        sentence_texts.append(_clean_sentence(restored))

    # Build sentence lookup for citation start positions
    def _find_sentence_idx(pos: int) -> int | None:
        lo, hi = 0, len(sentence_spans)
        while lo < hi:
            mid = (lo + hi) // 2
            s, e = sentence_spans[mid]
            if pos < s:
                hi = mid
            elif pos >= e:
                lo = mid + 1
            else:
                return mid
        return None

    # Pre-pass: for each sentence, check if it is "almost empty" aside from PUA citations.
    # Such sentences are artefacts of "prev_sentence. [cite]" — attribute their citations
    # to the previous non-empty sentence.
    def _sentence_is_only_citation(idx: int) -> bool:
        s, e = sentence_spans[idx]
        seg = masked[s:e]
        # Remove citation placeholders and whitespace.
        leftover = re.sub(rf"[{PUA_CITATION}\s]+", "", seg)
        if not leftover:
            return True
        # If what remains is only punctuation (no letters/digits/CJK chars),
        # the "sentence" is an artefact of stripping the citation out of
        # "prev. [cite]." and should be attributed to the previous sentence.
        return not re.search(r"[^\W_]", leftover, re.UNICODE)

    citations: list[Citation] = []
    for cm in citation_masks:
        idx = _find_sentence_idx(cm.start)
        if idx is None:
            # Fallback: attach to last sentence
            idx = len(sentence_spans) - 1
        # If target sentence is citation-only, walk backwards to a real sentence.
        while idx > 0 and _sentence_is_only_citation(idx):
            idx -= 1
        sentence_text = sentence_texts[idx] if 0 <= idx < len(sentence_texts) else ""
        meta = cm.meta or {}
        bbox = meta.get("bbox", [])
        citations.append(Citation(
            page=meta.get("page", -1),
            doc_page=meta.get("doc_page", "none"),
            bbox=bbox,
            sentence=sentence_text,
            span_start=cm.start,
            span_end=cm.end,
            is_normalized=meta.get("is_normalized", _bbox_is_normalized(bbox)),
        ))
    return citations


# ── Public API ────────────────────────────────────────────────────────────────

def validate_bbox(bbox: list[float]) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = bbox
    except ValueError:
        return False
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def extract_citations(text: str, locale: str = "en_US") -> list[Citation]:
    """End-to-end citation extraction with ICU-based sentence attribution."""
    masked, masks = mask_regions(text)
    # Abbreviation protection: swap `.` in known abbreviations to U+00B7
    # (same char length; indexes remain aligned).
    icu_input = _protect_abbreviations(masked)
    spans = split_sentences(icu_input, locale=locale)
    return _attribute_and_render(text, masked, masks, spans)


def parse_model_output(text: str, locale: str = "en_US") -> dict:
    """Parse the full reasoning output into structured fields."""
    answer = extract_answer(text)
    citations = extract_citations(text, locale=locale)
    reasoning = text
    if answer is not None:
        last = list(_ANSWER_RE.finditer(text))[-1]
        reasoning = text[:last.start()].rstrip()

    return {
        "answer": answer,
        "reasoning": reasoning,
        "citations": [c.as_dict() for c in citations],
        "invalid_citations": [
            c.as_dict() for c in citations if not validate_bbox(c.bbox)
        ],
        "predicted_pages": sorted({c.page for c in citations}),
        "has_answer_tag": answer is not None,
        "sentence_splitter": "icu" if _HAS_ICU else "regex_fallback",
    }


# ── Page-level precision / recall ─────────────────────────────────────────────

def page_prf(predicted: list[int] | set[int],
             gold: list[int] | set[int]) -> dict:
    """Return precision / recall / f1 / TP / FP / FN for page-id sets.

    Treats the task as set retrieval over 1-indexed global pages.
    Returns all zeros with sensible edge-case handling:
      - gold empty & pred empty: precision=recall=f1=1.0
      - gold empty & pred non-empty: precision=0, recall=1.0, f1=0
      - pred empty & gold non-empty: precision=1.0, recall=0, f1=0
    """
    pred_set = set(predicted)
    gold_set = set(gold)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    if not gold_set and not pred_set:
        precision = recall = f1 = 1.0
    elif not gold_set:
        precision = 0.0
        recall = 1.0
        f1 = 0.0
    elif not pred_set:
        precision = 1.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = tp / len(pred_set)
        recall = tp / len(gold_set)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted": sorted(pred_set),
        "gold": sorted(gold_set),
    }


def parse_judge(response: str) -> tuple[bool | None, str]:
    candidates = []
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if m:
        candidates.append(m.group(1))
    m = re.search(r'(\{[^{}]*"consistent"[^{}]*\})', response, re.DOTALL)
    if m:
        candidates.append(m.group(1))
    for raw in candidates:
        try:
            obj = json.loads(raw)
            return bool(obj.get("consistent")), str(obj.get("reasoning", "")).strip()
        except json.JSONDecodeError:
            continue
    m = re.search(r'"consistent"\s*:\s*(true|false)', response, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true", response.strip()[:500]
    return None, response.strip()[:500]
