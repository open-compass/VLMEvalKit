import re
import logging
import os
from sympy import simplify, expand, trigsimp
from sympy.parsing.latex import parse_latex
from dotenv import load_dotenv
import timeout_decorator


load_dotenv()



def extract_all_boxed_content(latex_response, latex_wrap=r'\\boxed{([^{}]*|{.*?})}'):
    pattern = re.compile(
        r'\\boxed{((?:[^{}]|{(?:[^{}]|{.*?})*})*)}'
        r'|\\\\\[boxed{((?:[^{}]|{(?:[^{}]|{.*?})*})*)}\\\\\]',
        re.DOTALL
    )
    matches = pattern.findall(latex_response)
    if not matches:
        return []
    return [match.strip() for sublist in matches for match in sublist if match.strip()]

def extract_final_answer(latex_response):
    match = re.search(r'\\boxed{(.*?)}|\\\\\[boxed{(.*?)}\\\\\]', latex_response)
    if match:
        return next(group for group in match.groups() if group).strip()
    return latex_response

def extract_final_answer_list(last_answer):
    matches = re.findall(r'\\boxed{\\\[(.*?)\\\]}|\\\\\[boxed{\\\[(.*?)\\\]}\\\\\]', last_answer)
    if matches:
        return [item.strip() for sublist in matches for item in sublist if item for item in item.split(',')]
    return [extract_final_answer(last_answer)]

def extract_final_answer_allform(latex_response, answer_type=None, latex_wrap=r'\\boxed{(.*?)}'):
    boxed_content = extract_all_boxed_content(latex_response, latex_wrap)
    if not boxed_content:
        return []

    if answer_type == 'list':
        return [extract_final_answer_list(item) for item in boxed_content]
    return [extract_final_answer(item) for item in boxed_content]




def _extract_core_eq(expr: str) -> str:
    if "\implies" in expr:
        expr = expr.split("\implies")[-1].strip()
    if "=" in expr:
        expr = expr.split("=")[-1].strip()
    return expr

def _preprocess_latex(string: str) -> str:
    if not string:
        return ""
    string = re.sub(r"_\{.*?\}", "", string)
    string = re.sub(r"_\\?\w", "", string)
    string = string.replace("\left", "").replace("\right", "").replace("\cdot", "*")
    return string

@timeout_decorator.timeout(10, use_signals=False)
def _standardize_expr(expr):
    return simplify(expand(trigsimp(expr)))


def call_llm_to_compare(expr1: str, expr2: str) -> bool:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant that compares LaTeX expressions for equivalence."},
                {"role": "user", "content": f"Compare the following LaTeX expressions and check if the numerical parts are equivalent in meaning.\n\nExpression 1:\n{expr1}\n\nExpression 2:\n{expr2}\n\nReturn True if they are equivalent, otherwise return False. Focus on mathematical content."}
            ]
        )
        return "true" in response.choices[0].message["content"].lower()
    except Exception as e:
        logging.warning(f"LLM comparison failed: {e}")
        return False

def is_equiv(expr1: str, expr2: str, verbose: bool = False) -> dict:
    result_data = {
        "input_expressions": {"expr1": expr1, "expr2": expr2},
        "preprocessed_expressions": {},
        "sympy_result": None,
        "llm_result": None,
        "final_result": None,
        "error": None,
    }

    try:
        if "\text" in expr1 or "\text" in expr2:
            result_data["llm_result"] = call_llm_to_compare(expr1, expr2)
            result_data["final_result"] = result_data["llm_result"]
            return result_data

        expr1_processed = _preprocess_latex(expr1)
        expr2_processed = _preprocess_latex(expr2)
        expr1_core = _extract_core_eq(expr1_processed)
        expr2_core = _extract_core_eq(expr2_processed)

        try:
            expr1_sympy = _standardize_expr(parse_latex(expr1_core))
            expr2_sympy = _standardize_expr(parse_latex(expr2_core))
            result_data["preprocessed_expressions"] = {
                "expr1": str(expr1_sympy),
                "expr2": str(expr2_sympy)
            }

            sympy_result = simplify(expr1_sympy - expr2_sympy) == 0 or expr1_sympy.equals(expr2_sympy)
        except Exception as e:
            result_data["error"] = str(e)
            sympy_result = None

        result_data["sympy_result"] = sympy_result

        if sympy_result:
            result_data["final_result"] = True
        else:
            result_data["llm_result"] = call_llm_to_compare(expr1, expr2)
            result_data["final_result"] = result_data["llm_result"]

    except Exception as e:
        result_data["error"] = str(e)

    return result_data
