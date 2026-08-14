"""Official CAPEval checklist-judge prompts. Keep verbatim for leaderboard comparability."""

# Bump this (and judge cache suffix) if the caption / judge prompt text changes.
PROMPT_VERSION = 'official-single-pass-v1'

CAPTION_PROMPT = 'Analyze the image in a comprehensive and detailed manner.'

SINGLE_PASS_SYSTEM = """You are an expert image-caption judge for a research benchmark.

Behavior:
- Apply a moderately strict standard: reward clear, accurate coverage; penalize contradictions.
- When the caption is genuinely ambiguous about a checklist point, prefer "not_mentioned" over guessing "yes".
- Use only the caption text and the checklist metadata provided; do not invent scene facts from prior knowledge.
- Output must be one valid JSON object exactly in the form requested by the user—no markdown code fences, no text before or after."""  # noqa: E501


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
    lines.extend(
        [
            '',
            'Return ONLY a JSON object (no markdown) with exactly this structure:',
            '{"gt_verdicts":['
            '{"item_index":<int>,"verdict":"yes"|"no"|"not_mentioned","reasoning":"<one short sentence>"},'
            '...]}',
            'Rules:',
            '- Every item_index from the checklist must appear exactly once in gt_verdicts.',
            '- Every gt_verdicts entry must include non-empty reasoning (at least a few words).',
        ]
    )
    return '\n'.join(lines)
