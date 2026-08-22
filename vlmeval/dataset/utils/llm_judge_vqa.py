"""Unified LLM-judge re-scoring for VQA-style benchmarks whose default
metric is a brittle string/numeric compare (ChartQA, OCRBench, DocVQA,
InfoVQA, TextVQA).

Behavior: each dataset's strict scoring runs first. For samples that
strict marked wrong, this module asks an LLM judge whether the model's
answer is semantically equivalent to the standard answer, using
dataset-specific examples that explicitly accept unit/format variants.
Output: '1' (consistent) or '0' (inconsistent).
"""

from vlmeval.constant import FAIL_MSG

_CHARTQA = (
    "You are grading a chart-question-answering task. Given the question, the "
    "standard answer, and the model's answer, decide whether the model's "
    "answer is semantically consistent with the standard answer.\n"
    "\n"
    "Rules:\n"
    "- The standard answer is always correct; you are only checking the model.\n"
    "- If the meaning is the same, the answer is CONSISTENT even when the "
    "model adds units, currency symbols, magnitude words (million/billion), "
    "whitespace, or wrapping prose. Examples of consistent answers:\n"
    "    * standard='106', model='106 million'\n"
    "    * standard='1.84', model='$1.84 billion U.S. dollars'\n"
    "    * standard='6750', model='6 750'\n"
    "    * standard='Poor', model='The category with the higher share is Poor.'\n"
    "- If the numeric value differs by more than ~5% of the standard answer, "
    "or the textual category is different, the answer is INCONSISTENT.\n"
    "\n"
    "Reply with a single character: '1' for consistent, '0' for inconsistent. "
    "No other text.\n"
    "\n"
    "Question: {question}\n"
    "Standard Answer: {answer}\n"
    "Model Answer: {prediction}\n"
    "Judgement:"
)

_OCRBENCH = (
    "You are grading an OCR / text-from-image task. Given the question, the "
    "standard answer (the correct text shown in the image), and the model's "
    "answer, decide whether the model's answer contains or matches the "
    "correct text.\n"
    "\n"
    "Rules:\n"
    "- If the correct text appears in the model's response, even within a "
    "sentence or with different case/spacing/punctuation, mark as CONSISTENT. "
    "Examples:\n"
    "    * standard='Coca-Cola', model='The brand shown is COCA COLA.'\n"
    "    * standard='12.50', model='The price reads $12.50'\n"
    "    * standard='STOP', model='It says stop on the sign.'\n"
    "    * standard='hello world', model='The text in the image reads: Hello, World!'\n"
    "- If the model output a different word, a wrong number, or missed the "
    "correct text entirely, mark INCONSISTENT.\n"
    "\n"
    "Reply with a single character: '1' for consistent, '0' for inconsistent.\n"
    "\n"
    "Question: {question}\n"
    "Standard Answer: {answer}\n"
    "Model Answer: {prediction}\n"
    "Judgement:"
)

_DOCVQA = (
    "You are grading a document-question-answering task (forms, invoices, "
    "letters, etc.). Given the question, the standard answer, and the model's "
    "answer, decide whether the model's answer is semantically equivalent to "
    "the standard answer in the document context.\n"
    "\n"
    "Rules:\n"
    "- Same value with different formatting is CONSISTENT. Examples:\n"
    "    * standard='$1,234', model='one thousand two hundred thirty-four dollars'\n"
    "    * standard='May 12, 2020', model='12/05/2020'\n"
    "    * standard='29.5%', model='29.5 percent' or '0.295'\n"
    "    * standard='John Smith', model='The name is John Smith.'\n"
    "- Different value, different entity, or wrong field → INCONSISTENT.\n"
    "\n"
    "Reply with a single character: '1' for consistent, '0' for inconsistent.\n"
    "\n"
    "Question: {question}\n"
    "Standard Answer: {answer}\n"
    "Model Answer: {prediction}\n"
    "Judgement:"
)

_INFOVQA = (
    "You are grading an infographic-question-answering task. Given the "
    "question, the standard answer, and the model's answer, decide whether "
    "the model's answer is semantically consistent with the standard answer.\n"
    "\n"
    "Rules:\n"
    "- Same value with different formatting is CONSISTENT. Examples:\n"
    "    * standard='45', model='45 million people'\n"
    "    * standard='2.3 billion', model='2.3B'\n"
    "    * standard='Asia', model='The continent is Asia.'\n"
    "- Different value or different entity → INCONSISTENT.\n"
    "\n"
    "Reply with a single character: '1' for consistent, '0' for inconsistent.\n"
    "\n"
    "Question: {question}\n"
    "Standard Answer: {answer}\n"
    "Model Answer: {prediction}\n"
    "Judgement:"
)

_TEXTVQA = (
    "You are grading a scene-text VQA task (signs, packaging, screens). "
    "Given the question, the standard answer, and the model's answer, "
    "decide whether the model's answer is consistent with the standard "
    "answer.\n"
    "\n"
    "Rules:\n"
    "- If the correct word/number appears in the model response, even within "
    "a sentence, mark CONSISTENT. Examples:\n"
    "    * standard='stop', model=\"It says 'Stop'.\"\n"
    "    * standard='5.99', model='The price tag reads $5.99.'\n"
    "    * standard='exit', model='The sign says EXIT in red letters.'\n"
    "- TextVQA answers are typically short (1-3 words). Be lenient on extra "
    "context but strict on the actual target word/number.\n"
    "- Different word or wrong number → INCONSISTENT.\n"
    "\n"
    "Reply with a single character: '1' for consistent, '0' for inconsistent.\n"
    "\n"
    "Question: {question}\n"
    "Standard Answer: {answer}\n"
    "Model Answer: {prediction}\n"
    "Judgement:"
)


PROMPTS = {
    'ChartQA_TEST': _CHARTQA,
    'OCRBench': _OCRBENCH,
    'OCRBench_MINI': _OCRBENCH,
    'DocVQA_VAL': _DOCVQA,
    'DocVQA_TEST': _DOCVQA,
    'InfoVQA_VAL': _INFOVQA,
    'InfoVQA_TEST': _INFOVQA,
    'TextVQA_VAL': _TEXTVQA,
    'TextVQA_TEST': _TEXTVQA,
}


def build_llm_judge_prompt(dataset_name, line):
    tmpl = PROMPTS.get(dataset_name, _CHARTQA)
    return tmpl.format(
        question=str(line.get('question', '')).strip(),
        answer=str(line.get('answer', '')).strip(),
        prediction=str(line.get('prediction', '')).strip(),
    )


def LLMJudge_auxeval(model, line, dataset_name=None):
    """Single LLM-judge call. Returns dict(log=..., res='1' or '0')."""
    prompt = build_llm_judge_prompt(dataset_name, line)
    log = ''
    for i in range(3):
        res = model.generate(prompt, temperature=0.0 if i == 0 else 0.3)
        if FAIL_MSG in str(res):
            log += f'Try {i}: API failed.\n'
            continue
        out = str(res).strip()
        first = next((c for c in out if c in '01'), None)
        if first is not None:
            return dict(log=log + 'Succeed', res=first)
        log += f'Try {i}: unparseable output: {out!r}\n'
    return dict(log=log + 'All retries failed', res='0')
