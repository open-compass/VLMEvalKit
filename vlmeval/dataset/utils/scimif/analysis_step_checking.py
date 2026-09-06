import json
import re
from typing import Any, Dict


def evaluate_analysis_steps(response: str,
                            item: Dict[str, Any],
                            instruction_name: str,
                            required_parameters: str = '',
                            instruction_description: str = '',
                            llm_client=None,
                            **kwargs) -> Dict[str, Any]:
    if not response:
        return {'passed': False, 'score': 0.0, 'detail': 'response is empty'}
    if llm_client is not None:
        return _llm_judge(response, item, instruction_name, required_parameters, instruction_description, llm_client)
    return _heuristic_judge(response, item, instruction_name, required_parameters, instruction_description)


def _llm_judge(response: str, item: Dict[str, Any], instruction_name: str, required_parameters: str,
               instruction_description: str, llm_client) -> Dict[str, Any]:
    try:
        prompt = _build_judge_prompt(response, item, instruction_name, required_parameters, instruction_description)
        if hasattr(llm_client, 'chat'):
            result = llm_client.chat(prompt)
        elif hasattr(llm_client, 'complete'):
            result = llm_client.complete(prompt)
        else:
            result = llm_client(prompt)
        if not isinstance(result, str):
            result = getattr(result, 'content', None) or getattr(result, 'text', None) or str(result)
        return _parse_llm_judge_result(result or '')
    except Exception as e:
        return {'passed': False, 'score': 0.0, 'detail': f'LLM-judge error: {e}'}


def _parse_llm_judge_result(result: str) -> Dict[str, Any]:
    json_match = re.search('```(?:json)?\\s*(\\{[\\s\\S]*?\\})\\s*```', result)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return _result_from_parsed_json(data, result)
        except json.JSONDecodeError:
            pass
    json_match = re.search('\\{[^{}]*\\"score\\"[^{}]*\\}', result)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return _result_from_parsed_json(data, result)
        except json.JSONDecodeError:
            pass
    frac_match = re.search('score\\s*[=:]\\s*(\\d+)\\s*/\\s*(\\d+)', result, re.I)
    if frac_match:
        num, den = (int(frac_match.group(1)), int(frac_match.group(2)))
        score = num / den if den > 0 else 0.0
        return {'passed': score >= 1.0, 'score': score, 'detail': f'LLM judge: {num}/{den} steps matched'}
    dec_match = re.search('score\\s*[=:]\\s*([\\d.]+)', result, re.I)
    if dec_match:
        score = float(dec_match.group(1))
        return {'passed': score >= 1.0, 'score': min(1.0, max(0.0, score)), 'detail': f'LLM-judge: score={score}'}
    passed = 'yes' in result.lower() or 'true' in result.lower()
    return {
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': f"LLM judge: {'passed' if passed else 'failed'}"
    }


def _result_from_parsed_json(data: dict, raw_result: str) -> Dict[str, Any]:
    score = data.get('score')
    if score is not None:
        if isinstance(score, (list, tuple)) and len(score) >= 2:
            num, den = (int(score[0]), int(score[1]))
            s = num / den if den > 0 else 0.0
        elif isinstance(score, str) and '/' in score:
            parts = score.split('/')
            if len(parts) == 2:
                num, den = (int(parts[0].strip()), int(parts[1].strip()))
                s = num / den if den > 0 else 0.0
            else:
                s = float(score)
        else:
            s = float(score)
        s = min(1.0, max(0.0, s))
    else:
        matched = data.get('matched', data.get('matched_count', 0))
        total = data.get('total', data.get('total_steps', 1))
        s = matched / total if total > 0 else 0.0
    detail_parts = []
    if 'required_steps' in data:
        detail_parts.append(f"requiredsteps: {len(data['required_steps'])}")
    if 'matched' in data:
        detail_parts.append(f"match: {data['matched']}")
    if 'total' in data:
        detail_parts.append(f"total: {data['total']}")
    detail = 'LLM-judge: ' + ', '.join(detail_parts) if detail_parts else f'LLM-judge: score={s:.2f}'
    return {'passed': s >= 1.0, 'score': s, 'detail': detail}


def _heuristic_judge(response: str, item: Dict[str, Any], instruction_name: str, required_parameters: str,
                     instruction_description: str) -> Dict[str, Any]:
    step_indicators = [
        'step\\s*\\d+', 'steps?\\s*\\d+', 'first', 'second', 'then', 'next', 'finally', '①|②|③|④|⑤', '1\\.|2\\.|3\\.',
        '→|⇒|->'
    ]
    has_steps = any((re.search(p, response, re.I) for p in step_indicators))
    sentences = [s.strip() for s in response.replace('\n', '.').split('.') if len(s.strip()) > 10]
    has_multiple = len(sentences) >= 2
    passed = has_steps or has_multiple
    return {
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': 'Detected a multi-step response' if passed else 'No multi-step response detected'
    }


def _build_judge_prompt(response: str, item: Dict[str, Any], instruction_name: str, required_parameters: str,
                        instruction_description: str) -> str:
    question = item.get('edit_question', item.get('original_question', ''))
    return f"""You are an expert evaluator for scientific questions. Evaluate whether the model's response satisfies the instruction for "reasoning steps" or "reasoning process".\n\n## Task\n1. **First output your reasoning** (brief): Extract the required step framework from the question and required_parameters; extract the actual steps from the model's response.\n2. **Then output a structured conclusion**: Compare each required step against the response, determine if it is mentioned, and compute the score.\n\n## Input\n\n**Instruction type**: {instruction_name}\n**Instruction description**: {instruction_description}\n**Required parameters/steps**: {required_parameters}\n\n**Question (edit_question)**:\n{question[:1500]}\n\n**Model response**:\n{response[:3000]}\n\n## Output Requirements\n1. First write your **reasoning**: List the steps you extracted from required_parameters and edit_question (required_steps), and the steps you extracted from the response (response_steps).\n2. Then write your **conclusion**: Output a JSON block at the end in the following format (do not omit):\n\n```json\n{{\n  "required_steps": ["step 1 description", "step 2 description", ...],\n  "response_steps": ["step 1 from response", "step 2 from response", ...],\n  "matched": <number of required steps correctly mentioned in the response, integer>,\n  "total": <total number of required steps, integer>,\n  "score": <matched/total, or a decimal, max 1.0>\n}}\n```\n\n**Scoring rule**: score = number of correctly mentioned steps / total required steps, full score is 1.0. A step counts as mentioned if the response contains corresponding content."""  # noqa: E501
