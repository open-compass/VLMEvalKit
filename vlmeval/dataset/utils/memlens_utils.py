#!/usr/bin/env python3
"""
MemLens LLM-as-Judge evaluation for VLMEvalKit.

Scoring: 1=correct, 0=incorrect (int). Output: judge_metrics.json + judge_details.json

Prompt style: grading teacher with rationale + score + JSON output.

v3 — thinking model robustness:
  - Degenerate/circular output detection (hedging markers + n-gram repetition)
  - Tail truncation for long outputs (focus judge on final answer)
  - Enhanced prompt with rules for circular reasoning + universal examples
  - Per-sample diagnostics (is_degenerate, was_truncated, hedging_count)
"""

import json
import re
import time
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Optional


# ============================================================
# MemLens parse_utils.py judge helpers
# ============================================================


# ============================================================
# Step 1: Parse model output — strip special tokens
# ============================================================

def parse_model_output(text: str) -> str:
    """
    Strip special tokens from model output to get the actual answer.

    Handles:
    - <think>...</think> (Qwen thinking traces) → keep text after </think>
    - <|endoftext|>, <|im_end|>, <|im_start|> etc → remove
    - <|begin_of_box|>...<|end_of_box|> (GLM box tokens) → unwrap to inner text
    - [CONTENT_FILTER_REJECTED] (Kimi filter) → empty string
    """
    if not text:
        return ""

    # 1. <think>...</think> → keep only text after </think>
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    # 2. Strip special tokens like <|endoftext|>, <|im_end|>, <|im_start|>
    text = re.sub(r'<\|[^>]*\|>', '', text)

    # 3. <|begin_of_box|>...<|end_of_box|> → inner text
    text = re.sub(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', r'\1', text, flags=re.DOTALL)
    text = text.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')

    # 4. [CONTENT_FILTER_REJECTED] → empty
    if '[CONTENT_FILTER_REJECTED]' in text:
        text = text.replace('[CONTENT_FILTER_REJECTED]', '')

    return text.strip()


# ============================================================
# Degenerate output detection
# ============================================================

# Markers that indicate the model is going in circles
_HEDGING_MARKERS = [
    "wait", "let me re-read", "let me re read", "let me double check",
    "let me double-check", "let me reconsider", "let me look again",
    "let's reconsider", "let s reconsider", "re-evaluating", "re evaluating",
    "actually looking", "actually let me", "hold on",
    "let me look at", "let me check", "let's re-read", "let s re-read",
    "i need to reconsider", "wait let me",
]


def detect_degenerate(text: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Detect if output is degenerate (circular reasoning, excessive repetition).

    Returns (is_degenerate, diagnostics_dict).

    A degenerate output is one where the model:
    - Repeatedly revisits the same evidence (>= 6 hedging markers)
    - Has very high n-gram repetition (< 40% unique 5-grams)
    """
    if not text or len(text) < 500:
        return False, {"hedging_count": 0, "unique_5gram_ratio": 1.0}

    text_lower = text.lower()

    # Count hedging/flip-flop markers
    hedging_count = sum(text_lower.count(m) for m in _HEDGING_MARKERS)

    # N-gram repetition ratio
    words = text_lower.split()
    n = 5
    if len(words) >= n:
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        unique_ratio = len(set(ngrams)) / len(ngrams) if ngrams else 1.0
    else:
        unique_ratio = 1.0

    is_degenerate = (hedging_count >= 6) or (unique_ratio < 0.40)

    diagnostics = {
        "hedging_count": hedging_count,
        "unique_5gram_ratio": round(unique_ratio, 3),
    }
    return is_degenerate, diagnostics


# ============================================================
# Step 4: Normalize for LLM judge (parse + detect + truncate)
# ============================================================

# Max characters to send to judge (~2000 tokens)
TAIL_CHAR_LIMIT = 6000
TAIL_FRACTION = 0.3
MIN_LENGTH_FOR_TRUNCATION = 3000


def normalize_for_judge(text: str) -> Dict[str, Any]:
    """
    Prepare text for LLM judge: parse → detect degenerate → tail truncate.

    Returns dict with keys:
      - text: the cleaned/truncated text for the judge
      - original_len: character count before processing
      - was_truncated: whether tail truncation was applied
      - is_degenerate: whether circular reasoning was detected
      - diagnostics: detailed detection metrics
    """
    if not text:
        return {
            "text": "",
            "original_len": 0,
            "was_truncated": False,
            "is_degenerate": False,
            "diagnostics": {},
        }

    original_len = len(text)

    # Step 1: parse special tokens
    text = parse_model_output(text)

    # Step 2: detect degenerate output
    is_degenerate, diagnostics = detect_degenerate(text)

    # Step 3: tail truncation for very long outputs
    was_truncated = False
    if len(text) > MIN_LENGTH_FOR_TRUNCATION and len(text) > TAIL_CHAR_LIMIT:
        tail_start = int(len(text) * (1 - TAIL_FRACTION))
        text = "[...truncated earlier reasoning...]\n" + text[tail_start:]
        was_truncated = True

    return {
        "text": text,
        "original_len": original_len,
        "was_truncated": was_truncated,
        "is_degenerate": is_degenerate,
        "diagnostics": diagnostics,
    }


# ============================================================
# MemLens judge prompts (inlined from judge_prompts/)
# ============================================================

QA_BENCHMARK_JUDGE_PROMPT = """Now your role is a grading teacher. Your task is to review and score student answers based on reference standard answers for a question-answering benchmark. You need to notice the following key points:
- First, extract the final answer from the student's solution, then analyze and judge whether the answer is correct.
- Scoring should only refer to the final answer obtained by the student; there is no need to examine whether the intermediate problem-solving steps are correct.
- If the response contains both hesitation and a clear answer, judge the answer itself.
- If the response contains both a refusal and a guessed answer, judge the final committed answer.
- If the response gives multiple inconsistent answers, assign 0 points.
- If the student's response shows circular reasoning — repeatedly revisiting the same evidence, flip-flopping between answers (e.g., "yes... wait no... actually yes... let me reconsider... no"), with no clear final commitment — assign 0 points. A correct answer mentioned during reasoning does NOT count unless it is the clearly stated final conclusion.
- Only the student's LAST clearly stated answer counts. If the student initially says "yes" but later concludes "no" (or vice versa), score based on the LAST position only. Intermediate answers during reasoning are not final answers.
- If the student's output is entirely a reasoning trace (e.g., "let me check... scanning evidence... re-reading session 5...") with no clear concluding answer statement, assign 0 points — even if the correct answer appears somewhere within the reasoning.
- If the student's output begins with "[...truncated earlier reasoning...]", this means only the final portion of a longer response is shown. Focus on extracting the answer from this final portion.

Below are examples of problematic student outputs that apply to ALL task types. Study these before grading.

[Universal Example A — Circular Reasoning -> 0 points]
<Question>: Is the cat in Mark and Jenny's 5-year anniversary post named Mittens?
<Standard Answer>: Yes
<Student Answer>: let me check session 5 mark has cat named mittens session 7 mark and jenny anniversary post there is no link wait let me re-read session 5 says mittens session 7 says anniversary actually looking at this again wait is it possible let me reconsider mark has mittens in session 5 the post is about couple therefore answer is likely no wait let me double check session 5 mark has cat mittens actually maybe yes but text doesnt say
[Scoring Rationale]: The output flip-flops between "yes" and "no" multiple times with no clear final commitment. The last position is ambiguous ("maybe yes but text doesnt say"). This is circular reasoning.
In summary, the student's answer deserves 0 points.
[JSON]: {{"answer_score": 0}}

[Universal Example B — Redundant But Committed -> Score Normally]
<Question>: Is the trailing plant in the cafe safe for my kitten?
<Standard Answer>: No
<Student Answer>: the plant looks like pothos pothos is toxic to cats therefore answer is no let me double check is pothos safe for cats no is answer no yes wait let me verify image 3 is pothos image 9 is pothos pothos is toxic therefore answer must be no
[Scoring Rationale]: Despite excessive self-verification, the student consistently commits to "no" throughout and never wavers to a different answer. The final answer is clearly "No", which matches the standard answer.
In summary, the student's answer deserves 1 point.
[JSON]: {{"answer_score": 1}}

[Universal Example C — Reasoning Trace With No Answer -> 0 points]
<Question>: How many sessions mentioned the coffee shop?
<Standard Answer>: 3
<Student Answer>: scanning session 1 yes coffee mentioned session 2 no session 3 yes coffee again session 4 no session 5 maybe let me re-read session 5 it mentions cafe is that same as coffee shop need to check session 6 no session 7 unclear let me look at session 5 again
[Scoring Rationale]: The student walks through sessions but never states a final count. The output is entirely a reasoning trace with no concluding answer.
In summary, the student's answer deserves 0 points.
[JSON]: {{"answer_score": 0}}

Now proceed with grading. Remember: the universal examples above apply regardless of task type.

- When analyzing and judging whether the answer is correct, you need to write down the scoring rationale, organize it into clear statements that follow the logical flow. The summary of the scoring rationale should be placed at the end, using the following format: "In summary, the student's answer deserves x points" (where x represents the student's specific score).
- Keep the whole process concise, within 150 words.
- Provide the score based on your analysis and display it in a code block in "JSON" format.
- An item is covered if it is strictly mentioned or unambiguously implied by a semantic equivalence. This includes numerical equivalence (e.g., 10% and 0.1), synonyms (e.g., UK and United Kingdom), plural/singular forms (e.g., "apple" and "apples"), and equivalent date formats (e.g., 2024-01-15 and January 15, 2024). However, do not accept loosely related concepts.
- Ignore minor formatting differences, capitalization, punctuation, and equivalent wording when meaning is unchanged.
Your output format is:
[Scoring Rationale]:
[Score]: x points
[JSON]:
{{"answer_score": <integer_value>}}

Below is the grading rubric:
[Scores]:
The scoring scale consists of 2 levels in total, from highest to lowest: 1 point, 0 points (the minimum is 0 points).
[Tier Details]:
1 point: Assign 1 point if the student's final answer matches the standard answer under the task-specific criteria below.
0 points: Assign 0 points if the student's final answer does not match the standard answer, the student refuses to answer, claims insufficient information (except for AR tasks), or does not clearly answer.

[Task-Specific Criteria]:
{task_criteria}

{examples}
"""

TASK_CRITERIA = {
    "IE_InformationExtraction": """[IE — Information Extraction]
This is an information extraction question.
- Assign 1 point if the student's response contains the core information from the standard answer. Minor wording differences are acceptable, but the essential information must be present and correct.
- Assign 0 points if the core information is missing, contradicted, too vague, refused, or incorrect.""",
    "KU_KnowledgeUpdate": """[KU — Knowledge Update]
This is a knowledge update question testing whether the model tracks the latest state.
The correct current answer and the outdated old answer are both provided.
- Assign 1 point if the student gives the current (correct) answer.
- Assign 0 points if the student gives the outdated old answer, any other incorrect answer, refuses, or does not clearly answer.
- Naming vs. description: the student must name or unambiguously denote the same specific current item as the standard answer. Apply a blind reverse-identification test: using only the student's final answer, a grader should be able to infer the standard answer's specific item without seeing the image, source conversation, or reference answer. A generic visual, physical, or functional description that could plausibly refer to several different items does not count, even if it is consistent with the standard answer. Omitting a non-essential modifier is acceptable only if the remaining answer still uniquely identifies the same item.
  - Counts (1): "ceramic cup" for "ceramic mug" (same specific item, looser wording); "dark leather oxfords" for "leather oxford shoe" (same shoe type and material).
  - Does not count (0): "small fizzy cube" for "bath bomb"; "a snug rolled-up sleep setup" for "sleeping bag"; "making scented soaps" for "soap making kit"; "ankle-height lace-up boots" for "desert boots" — these are generic descriptions that do not pin down the specific named item.""",
    "TR_DurationComparison": """[TR — Duration Comparison (A/B)]
This is a binary choice question. The student must select one of two options.
- Assign 1 point if the student clearly selects the same option as the standard answer, regardless of phrasing.
- Assign 0 points if the student selects the other option, refuses, or does not make a clear selection.""",
    "TR_OrderRanking": """[TR — Order Ranking]
This is an ordering question. The standard answer is an exact sequence of items.
- Assign 1 point ONLY if the student gives the exact same sequence in the same order.
- Assign 0 points if any item is missing, added, duplicated, misplaced, or if the order is unclear.""",
    "TR_DateExtraction": """[TR — Date Extraction]
This is a date extraction question. The standard answer is a specific date.
- Assign 1 point if the student gives the same date in any unambiguous equivalent format (e.g., "Jan 15, 2024" and "2024-01-15" and "15/01/2024").
- Assign 0 points if the date is different, incomplete when a full date is required, ambiguous, or missing.""",
    "MSR_YesNo": """[MSR — Yes/No (Entity Resolution)]
This is a yes/no question.
- Assign 1 point if the student's final answer is semantically equivalent to the standard answer (e.g., "Yes" matches "Yes", "No" matches "No"), even if phrased differently (e.g., "That's correct" for "Yes").
- Assign 0 points if the answer means the opposite, refuses, or does not clearly answer.""",
    "MSR_Counting": """[MSR — Counting]
This is a counting question. The standard answer is a specific number.
- Assign 1 point if the student clearly states the exact number in any equivalent form (e.g., "3", "three", "3.0").
- Assign 0 points if the student gives a different number, a range, an approximation, refuses, or does not clearly answer.""",
    "MSR_Arithmetic": """[MSR — Arithmetic]
This is an arithmetic question. The standard answer is a specific value.
- Assign 1 point if the student gives the same value in an equivalent format (e.g., "0.5" and "1/2" and "50%").
- Assign 0 points if the student gives a different value, an unsupported approximation, refuses, or does not clearly answer.""",
    "AR_AnswerRefusal": """[AR — Answer Refusal]
- Assign 1 point if the student explicitly refuses, says there is not enough information, or indicates the question is unanswerable from the provided context.
- Assign 0 points if the student gives a substantive answer, even if hedged.""",
}

TASK_EXAMPLES = {
    "IE_InformationExtraction": """
[Example 1]
<Question>: What specific word is painted in large white letters on the pavement in the middle lanes?
<Standard Answer>: SCHOOL
<Student Answer>: SCHOOL

[Scoring Rationale]: The student's answer is "SCHOOL", which exactly matches the standard answer.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: Where on the sign is the photo of the mountains located?
<Standard Answer>: In the bottom right corner
<Student Answer>: The photo of the mountains is located in the upper right corner of the sign.

[Scoring Rationale]: The student says "upper right corner" but the standard answer is "bottom right corner". The location is incorrect — "upper" vs "bottom" is a substantive difference.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}

""",
    "KU_KnowledgeUpdate": """
[Example 1]
<Question>: What do I treat as my must-have now that I plan my meals around it?
<Standard Answer>: brioche
<Old (Outdated) Answer>: sourdough
<Student Answer>: You now consider brioche your must-have bread that you plan meals around.

[Scoring Rationale]: The student's answer is "brioche", which matches the current standard answer, not the outdated answer "sourdough".
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: What do I treat as my must-have now that I plan my meals around it?
<Standard Answer>: brioche
<Old (Outdated) Answer>: sourdough
<Student Answer>: A versatile, high-quality protein source like chicken, beans, or tofu is a great starting point for building a week's worth of meals.

[Scoring Rationale]: The student's answer mentions generic protein sources, which is neither the current answer "brioche" nor the outdated answer "sourdough". The answer is simply incorrect.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}

[Example 3]
<Question>: What do I prefer now?
Answer with only the name or phrase, no explanation.
<Standard Answer>: bath bomb
<Student Answer>: a small fizzy cube

[Scoring Rationale]: "A small fizzy cube" is a generic visual description that could refer to many different objects and does not name or uniquely identify the specific current item "bath bomb". The student described an appearance rather than identifying the item.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
    "TR_DurationComparison": """
[Example 1]
<Question>: Which of the following two durations is longer? Duration 1: The time I spent on Miller Hall. Duration 2: The time I spent on Music Festival Job. A. Duration 1 is longer. B. Duration 2 is longer.
<Standard Answer>: B
<Student Answer>: B. Duration 2 is longer.

[Scoring Rationale]: The student clearly selects "B", which matches the standard answer.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: Which of the following two durations is longer? Duration 1: The time I spent with the NutriPeak Nutrition. Duration 2: The time I spent with the BlendIt Pro Blender. A. Duration 1 is longer. B. Duration 2 is longer.
<Standard Answer>: A
<Student Answer>: B. Duration 2 is longer.

[Scoring Rationale]: The student selects "B", but the standard answer is "A". The selection is opposite.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
    "TR_OrderRanking": """
[Example 1]
<Question>: Rank the following cities by population from largest to smallest: Tokyo, Delhi, Shanghai.
<Standard Answer>: Tokyo, Delhi, Shanghai
<Student Answer>: 1. Tokyo 2. Delhi 3. Shanghai

[Scoring Rationale]: The student's sequence is Tokyo, Delhi, Shanghai, which exactly matches the standard answer.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: Rank events A, B, C in chronological order.
<Standard Answer>: B, A, C
<Student Answer>: A, B, C

[Scoring Rationale]: The student's sequence is A, B, C, but the standard answer is B, A, C. The order does not match.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
    "TR_DateExtraction": """
[Example 1]
<Question>: When did I complete the Scandinavian design course at Copenhagen Design Academy?
<Standard Answer>: 2024/08/31
<Student Answer>: You completed the Scandinavian design course on August 31st, 2024.

[Scoring Rationale]: The student's answer "August 31st, 2024" is an unambiguous equivalent of the standard answer "2024/08/31".
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: When did I visit Marutama Ra-men to taste their chicken broth?
<Standard Answer>: 2023/01/20
<Student Answer>: January 20, 2021

[Scoring Rationale]: The student's answer is "January 20, 2021", but the standard answer is "2023/01/20" — the year is wrong (2021 vs 2023).
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
    "MSR_YesNo": """
[Example 1]
<Question>: Is the cat in Mark and Jenny's 5-year anniversary post named Mittens?
<Standard Answer>: Yes
<Student Answer>: Yes, the cat in Mark and Jenny's 5-year anniversary post is named Mittens.

[Scoring Rationale]: The student's final answer is "Yes", which matches the standard answer "Yes". The additional explanation does not contradict the answer.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: Did the cat named after a crypto creator knock over the phone displaying the Binance alert?
<Standard Answer>: Yes
<Student Answer>: The text does not explicitly state whether the cat knocked over the phone. It only mentions the cat was sitting next to the phone when it crashed.

[Scoring Rationale]: The student does not give a clear "Yes". Instead, the student hedges and effectively refuses to commit to an answer. The standard answer is "Yes".
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
    "MSR_Counting": "",
    "MSR_Arithmetic": """
[Example 1]
<Question>: How much total have I spent on coffee makers?
<Standard Answer>: $260.00
<Student Answer>: Based on the conversation, you spent $110 on the first coffee maker and $150 on the second one, totaling $260.

[Scoring Rationale]: The student's final answer is $260, which is numerically equivalent to the standard answer $260.00.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: How much total have I spent on coffee makers?
<Standard Answer>: $260.00
<Student Answer>: I do not have access to your personal financial data, including your coffee maker purchases. I cannot provide you with a total amount spent on coffee makers.

[Scoring Rationale]: The student refuses to answer and claims insufficient information, but this is not an AR task. The information was available in the conversation context.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
    "AR_AnswerRefusal": """
[Example 1]
<Question>: What is the model number printed on the power bank?
<Standard Answer>: Insufficient information
<Student Answer>: I am sorry, but I cannot fulfill your request. The provided conversation history does not contain any information about a power bank or its model number.

[Scoring Rationale]: The student correctly refuses to answer, stating the information is not available in the conversation. This is the expected behavior for an unanswerable question.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: How many vertical cables are visible on one side of the bridge?
<Standard Answer>: Insufficient information
<Student Answer>: Three

[Scoring Rationale]: The student provides a substantive answer ("Three") rather than refusing. For an unanswerable question, the correct response is to refuse or indicate insufficient information.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
}


def infer_question_subtype(question_type: str, question: str, reference: str) -> str:
    """Infer MemLens subtype when the release JSON does not provide one."""
    question = question or ""
    reference = (reference or "").strip()

    if question_type == "temporal_reasoning":
        if "Which of the following two durations is longer?" in question:
            return "duration_comparison"
        if "Duration 1" in question and "Duration 2" in question and reference in ("A", "B"):
            return "duration_comparison"
        if "sort these facts in chronological order" in question:
            return "order_ranking"
        if re.fullmatch(r"(?:\(\d+\))+", reference):
            return "order_ranking"
        return "temporal_info_extraction"

    if question_type == "multi_session_reasoning" and reference in ("Yes", "No"):
        return "entity_resolution"

    return ""


def get_task_key(question_type: str, question_subtype: str, reference: str, question: str = "") -> str:
    """Map (question_type, question_subtype, reference) to a task key."""
    if not question_subtype:
        question_subtype = infer_question_subtype(question_type, question, reference)

    if question_type == "multi_session_reasoning":
        if question_subtype == "arithmetic":
            return "MSR_Arithmetic"
        if question_subtype == "counting":
            return "MSR_Counting"
        if question_subtype == "entity_resolution":
            return "MSR_YesNo" if reference.strip() in ("Yes", "No") else "MSR_Counting"
        return "MSR_Counting"
    if question_type == "temporal_reasoning":
        if question_subtype == "duration_comparison":
            return "TR_DurationComparison"
        if question_subtype == "order_ranking":
            return "TR_OrderRanking"
        if question_subtype == "temporal_info_extraction":
            return "TR_DateExtraction"
        return "TR_DateExtraction"
    if question_type == "knowledge_update":
        return "KU_KnowledgeUpdate"
    if question_type == "answer_refusal":
        return "AR_AnswerRefusal"
    return "IE_InformationExtraction"


def build_judge_prompt(
    task_type: str,
    question: str,
    reference: str,
    prediction: str,
    old_answer: str = None,
) -> str:
    """Build the full judge prompt for a given task type."""
    criteria = TASK_CRITERIA.get(task_type, "")
    examples = TASK_EXAMPLES.get(task_type, "")

    system_prompt = QA_BENCHMARK_JUDGE_PROMPT.format(
        task_criteria=criteria,
        examples=examples,
    )

    if task_type == "KU_KnowledgeUpdate" and old_answer:
        current_case = f"""[Current Case]
<Question>: {question}
<Standard Answer>: {reference}
<Old (Outdated) Answer>: {old_answer}
<Student Answer>: {prediction}

"""
    elif task_type == "AR_AnswerRefusal":
        current_case = f"""[Current Case]
<Question>: {question}
<Standard Answer>: Insufficient information
<Student Answer>: {prediction}

"""
    else:
        current_case = f"""[Current Case]
<Question>: {question}
<Standard Answer>: {reference}
<Student Answer>: {prediction}

"""

    return system_prompt + current_case + "[Scoring Rationale]:"

# ── Question metadata ──
_QUESTIONS_METADATA = None

def _load_questions_metadata(questions_file: Optional[str] = None) -> Dict[str, Dict]:
    global _QUESTIONS_METADATA
    if _QUESTIONS_METADATA is not None:
        return _QUESTIONS_METADATA
    if questions_file is None:
        candidates = [
            Path(__file__).parent.parent / "data" / "final_dataset_test" / "all_questions.json",
            Path.home() / "MEMLENS" / "data" / "all_questions.json",
        ]
        for p in candidates:
            if p.exists():
                questions_file = str(p)
                break
    if questions_file is None or not Path(questions_file).exists():
        print("[Judge] WARNING: all_questions.json not found")
        _QUESTIONS_METADATA = {}
        return _QUESTIONS_METADATA
    with open(questions_file) as f:
        all_qs = json.load(f)
    _QUESTIONS_METADATA = {}
    for q in all_qs:
        qid = q["question_id"]
        _QUESTIONS_METADATA[qid] = {
            "question_subtype": q.get("question_subtype", ""),
            "old_answer": q.get("question_content", {}).get("old_answer", ""),
        }
        if qid.startswith("0x"):
            _QUESTIONS_METADATA[qid[2:]] = _QUESTIONS_METADATA[qid]
    print(f"[Judge] Loaded metadata for {len(all_qs)} questions")
    return _QUESTIONS_METADATA

def enrich_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if item.get("question_subtype") and item.get("old_answer") is not None:
        return item
    metadata = _load_questions_metadata()
    qid = item.get("question_id", "")
    meta = metadata.get(qid) or metadata.get(qid.lstrip("0x")) or {}
    if not item.get("question_subtype"):
        item["question_subtype"] = meta.get("question_subtype", "")
    if not item.get("old_answer"):
        item["old_answer"] = meta.get("old_answer", "")
    return item



# Prompts are in judge_prompts/ — see judge_prompts/__init__.py for API.


# ── Response parsing ──

def _parse_judge_response(text: str) -> int:
    """Extract answer_score from judge response. Returns 0 or 1."""
    if not text:
        return 0

    # Try to find JSON block with answer_score
    m = re.search(r'"answer_score"\s*:\s*(\d+)', text)
    if m:
        score = int(m.group(1))
        return min(score, 1)  # clamp to 0-1

    # Fallback: look for "deserves X point" pattern
    m = re.search(r'deserves\s+(\d+)\s+point', text, re.IGNORECASE)
    if m:
        score = int(m.group(1))
        return min(score, 1)

    # Fallback: look for [Score]: X
    m = re.search(r'\[Score\]\s*:\s*(\d+)', text)
    if m:
        score = int(m.group(1))
        return min(score, 1)

    # Last resort: default to 0
    return 0


# ── Core judge call ──

def _judge_one(question: str, reference: str, prediction_info: Dict[str, Any],
               question_type: str, question_subtype: str = "",
               old_answer: str = "", use_cache: bool = True,
               judge_model: Any = None, judge_name: str = "") -> Tuple[int, Dict]:
    """Score one prediction. Returns (score, diagnostics)."""
    prediction = prediction_info["text"]
    diagnostics = {
        "original_len": prediction_info["original_len"],
        "was_truncated": prediction_info["was_truncated"],
        "is_degenerate": prediction_info["is_degenerate"],
        **prediction_info.get("diagnostics", {}),
    }

    task_key = get_task_key(question_type, question_subtype, reference, question)
    prompt = build_judge_prompt(task_key, question, reference, prediction, old_answer)

    if judge_model is None:
        raise ValueError("MemLens LLM judge requires a VLMEvalKit judge_model.")

    for attempt in range(3):
        try:
            raw = str(judge_model.generate(prompt) or "").strip()
            score = _parse_judge_response(raw)
            diagnostics["judge_raw"] = raw

            return score, diagnostics

        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (2 ** attempt))
                continue
            print(f"[Judge] API error after 3 retries: {e}")
            return 0, diagnostics


# ── Concurrent evaluation ──
def evaluate(data: List[Dict[str, Any]],
             jsonl_path: str,
             use_cache: bool = True,
             max_samples: Optional[int] = None,
             num_workers: int = 8,
             judge_model: Any = None,
             judge_name: str = "") -> Tuple[Dict, List[Dict]]:
    """
    Run LLM judge on all samples with ThreadPoolExecutor + JSONL resume.

    Returns (metrics_dict, details_list).
    """
    eval_data = data[:max_samples] if max_samples else data
    total = len(eval_data)

    if judge_model is None:
        raise ValueError("MemLens LLM judge requires a VLMEvalKit judge_model.")

    # Pre-enrich all items (single-threaded, fast)
    for item in eval_data:
        enrich_item(item)

    # JSONL resume: load completed indices
    completed = {}  # idx -> {question_id, judge_score, ...}
    if Path(jsonl_path).exists():
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                completed[rec["idx"]] = rec
        print(f"[Judge] Resuming: {len(completed)}/{total} already done")

    remaining = [i for i in range(total) if i not in completed]
    if not remaining:
        print(f"[Judge] All {total} samples already judged")
    else:
        print(f"[Judge] Processing {len(remaining)} samples with {num_workers} workers...")

    # Thread-safe JSONL writer
    write_lock = threading.Lock()
    fout = open(jsonl_path, "a")
    done_count = [len(completed)]  # mutable counter

    # Parsed output word count threshold for auto-zero.
    # Valid answers are typically ≤50 words. Anything >500 words in parsed_output
    # (after <think> stripping) is certainly not a concise answer — it's either
    # unstripped reasoning, circular garbage, or repetitive nonsense.
    # Note: we use parsed_output word count, NOT output_len (which includes
    # thinking tokens and varies with gen_max_length across models).
    PARSED_WORDS_THRESHOLD = 500

    def process(idx: int) -> Dict:
        item = eval_data[idx]
        output_len = item.get('output_len', 0)

        raw_pred = item.get('parsed_output') or item.get('prediction', '')
        prediction_info = normalize_for_judge(raw_pred)

        # Check if parsed output is excessively long (degenerate/unstripped reasoning)
        parsed_words = len(prediction_info["text"].split())
        auto_zero = parsed_words > PARSED_WORDS_THRESHOLD

        if auto_zero:
            # Parsed output too long → not a valid answer, auto-score 0
            score = 0
            diagnostics = {
                "original_len": prediction_info["original_len"],
                "was_truncated": prediction_info["was_truncated"],
                "is_degenerate": prediction_info["is_degenerate"],
                **prediction_info.get("diagnostics", {}),
            }
        else:
            score, diagnostics = _judge_one(
                question=item.get('question', ''),
                reference=item['reference_answer'],
                prediction_info=prediction_info,
                question_type=item.get('question_type', 'unknown'),
                question_subtype=item.get('question_subtype', ''),
                old_answer=item.get('old_answer', ''),
                use_cache=use_cache,
                judge_model=judge_model,
                judge_name=judge_name,
            )
        rec = {
            "idx": idx,
            "question_id": item.get('question_id'),
            "question_type": item.get('question_type', ''),
            "question_subtype": item.get('question_subtype', ''),
            "task_key": get_task_key(item.get('question_type', ''),
                                      item.get('question_subtype', ''),
                                      item['reference_answer'],
                                      item.get('question', '')),
            "judge_score": score,
            "judge_raw": diagnostics.get("judge_raw", ""),
            "original_len": diagnostics.get("original_len", 0),
            "was_truncated": diagnostics.get("was_truncated", False),
            "is_degenerate": diagnostics.get("is_degenerate", False),
            "hedging_count": diagnostics.get("hedging_count", 0),
            "unique_5gram_ratio": diagnostics.get("unique_5gram_ratio", 1.0),
            "auto_zero": auto_zero,
            "parsed_words": parsed_words,
            "output_len": output_len,
        }
        with write_lock:
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            done_count[0] += 1
            if done_count[0] % 100 == 0:
                print(f"  {done_count[0]}/{total}")
        return rec

    start = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(process, i): i for i in remaining}
        for future in as_completed(futures):
            rec = future.result()
            completed[rec["idx"]] = rec
    fout.close()
    elapsed = time.time() - start

    if remaining:
        print(f"[Judge] Done {len(remaining)} samples in {elapsed:.1f}s ({len(remaining)/elapsed:.1f} samples/s)")

    # Build ordered details list + metrics with diagnostics
    details = []
    by_type = defaultdict(list)
    degenerate_counts = defaultdict(lambda: {"total": 0, "degenerate": 0, "truncated": 0, "auto_zero": 0})

    for idx in range(total):
        rec = completed[idx]
        details.append({
            "question_id": rec["question_id"],
            "question_type": rec["question_type"],
            "question_subtype": rec.get("question_subtype", ""),
            "task_key": rec["task_key"],
            "judge_score": rec["judge_score"],
            "judge_raw": rec.get("judge_raw", ""),
            "original_len": rec.get("original_len", 0),
            "was_truncated": rec.get("was_truncated", False),
            "is_degenerate": rec.get("is_degenerate", False),
            "hedging_count": rec.get("hedging_count", 0),
            "unique_5gram_ratio": rec.get("unique_5gram_ratio", 1.0),
            "auto_zero": rec.get("auto_zero", False),
            "parsed_words": rec.get("parsed_words", 0),
            "output_len": rec.get("output_len", 0),
        })
        by_type[rec["question_type"]].append(rec["judge_score"])

        qtype = rec["question_type"]
        degenerate_counts[qtype]["total"] += 1
        if rec.get("is_degenerate"):
            degenerate_counts[qtype]["degenerate"] += 1
        if rec.get("was_truncated"):
            degenerate_counts[qtype]["truncated"] += 1
        if rec.get("auto_zero"):
            degenerate_counts[qtype]["auto_zero"] += 1

    all_scores = [d["judge_score"] for d in details]
    yes = sum(all_scores)
    no = total - yes

    total_degenerate = sum(dc["degenerate"] for dc in degenerate_counts.values())
    total_truncated = sum(dc["truncated"] for dc in degenerate_counts.values())
    total_auto_zero = sum(dc["auto_zero"] for dc in degenerate_counts.values())

    by_type_metrics = {}
    for qtype, scores in sorted(by_type.items()):
        t_yes = sum(scores)
        t_total = len(scores)
        dc = degenerate_counts[qtype]
        by_type_metrics[qtype] = {
            "accuracy": t_yes / t_total if t_total else 0,
            "yes": t_yes, "no": t_total - t_yes, "count": t_total,
            "degenerate_count": dc["degenerate"],
            "truncated_count": dc["truncated"],
            "auto_zero_count": dc["auto_zero"],
            "degenerate_pct": round(dc["degenerate"] / dc["total"] * 100, 1) if dc["total"] else 0,
        }

    metrics = {
        "overall": {
            "accuracy": yes / total if total else 0,
            "yes": yes, "no": no, "total": total,
            "degenerate_count": total_degenerate,
            "degenerate_pct": round(total_degenerate / total * 100, 1) if total else 0,
            "truncated_count": total_truncated,
            "truncated_pct": round(total_truncated / total * 100, 1) if total else 0,
            "auto_zero_count": total_auto_zero,
            "auto_zero_pct": round(total_auto_zero / total * 100, 1) if total else 0,
        },
        "by_question_type": by_type_metrics,
    }
    return metrics, details
