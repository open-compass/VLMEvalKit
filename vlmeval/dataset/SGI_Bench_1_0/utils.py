import re
#
# def timeout_limit(timeout=None):
#     """
#     A decorator for enforcing function execution time limits.
#
#     Args:
#         timeout (float or None): Maximum allowed execution time in seconds.
#                                 If None, no time limit is applied.
#
#     Returns:
#         The function's return value if completed in time.
#         Raises TimeoutError if execution exceeds the allowed time.
#     """
#
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             if timeout is None:
#                 # No time restriction
#                 return func(*args, **kwargs)
#
#             result = [None]
#             exc = [None]
#
#             def target():
#                 try:
#                     result[0] = func(*args, **kwargs)
#                 except Exception as e:
#                     exc[0] = e
#
#             thread = threading.Thread(target=target)
#             thread.start()
#             thread.join(timeout)
#
#             if thread.is_alive():
#                 raise TimeoutError(f"[Error][Function timeout after {timeout}s]")
#
#             if exc[0] is not None:
#                 raise exc[0]
#
#             return result[0]
#
#         return wrapper
#
#     return decorator
#
#
# class LLM:
#     def __init__(self, model='gpt-4.1', **kwargs):
#         self.api_key = kwargs.get('api_key', os.environ.get('OPENAI_API_KEY'))
#         self.base_url = kwargs.get('base_url', os.environ.get('OPENAI_BASE_URL'))
#         self.model = model
#
#         if not self.api_key:
#             raise ValueError("API key is required.")
#
#         self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
#
#         self.use_chat = False
#         self.use_responses = False
#         # -----------------------------
#         # Try chat.completions
#         # -----------------------------
#         try:
#             self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": "test"},
#                     {"role": "user", "content": "hello"},
#                 ],
#             )
#             self.use_chat = True
#             # print(f"[INFO][LLM] chat.completions supported → using chat")
#             return
#
#         except Exception as e_chat:
#             # print(f"[INFO][LLM] chat.completions unsupported: {e_chat}")
#             pass
#
#         # -----------------------------
#         # Try responses
#         # -----------------------------
#         try:
#             self.client.responses.create(
#                 model=self.model,
#                 input="hello",
#             )
#             self.use_responses = True
#             # print(f"[INFO][LLM] responses API supported → using responses")
#             return
#
#         except Exception as e_resp:
#             # print(f"[INFO][LLM] responses unsupported: {e_resp}")
#             pass
#
#         # -----------------------------
#         # Neither works → fail init
#         # -----------------------------
#         raise RuntimeError(
#             f"Model '{self.model}' supports neither chat.completions nor responses API."
#         )
#
#     @timeout_limit(5 * 60)
#     def __call__(self, query=None, **kwargs):
#         system_prompt = kwargs.get('system_prompt', 'You are a helpful assistant.')
#         max_tokens = kwargs.get('max_tokens', None)
#         temperature = kwargs.get('temperature', 0)
#
#         if self.use_chat:
#             messages = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": query},
#             ]
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=messages,
#                 max_tokens=max_tokens,
#                 temperature=temperature,
#             )
#             return response.choices[0].message.content
#
#         elif self.use_responses:
#             response = self.client.responses.create(
#                 model=self.model,
#                 input=query,
#                 max_output_tokens=max_tokens,
#                 temperature=temperature,
#             )
#             return response.output_text
#
#
# class VLM:
#     def __init__(self, model='gpt-4.1', **kwargs):
#         self.api_key = kwargs.get('api_key', os.environ.get('OPENAI_API_KEY'))
#         self.base_url = kwargs.get('base_url', os.environ.get('OPENAI_BASE_URL'))
#         self.model = model
#
#         if not self.api_key:
#             raise ValueError("API key is required.")
#
#         self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
#
#         self.use_chat = False
#         self.use_responses = False
#         # -----------------------------
#         # Try chat.completions
#         # -----------------------------
#         try:
#             self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": "test"},
#                     {"role": "user", "content": [{"type": "text", "text": "hello"}]},
#                 ],
#             )
#             self.use_chat = True
#             # print(f"[INFO][VLM] chat.completions supported → using chat")
#             return
#
#         except Exception as e_chat:
#             # print(f"[INFO][VLM] chat.completions unsupported: {e_chat}")
#             pass
#
#         # -----------------------------
#         # Try responses
#         # -----------------------------
#         try:
#             self.client.responses.create(
#                 model=self.model,
#                 input="hello",
#             )
#             self.use_responses = True
#             # print(f"[INFO][VLM] responses API supported → using responses")
#             return
#
#         except Exception as e_resp:
#             # print(f"[INFO][VLM] responses unsupported: {e_resp}")
#             pass
#
#         # -----------------------------
#         # Neither works → fail init
#         # -----------------------------
#         raise RuntimeError(
#             f"Model '{self.model}' supports neither chat.completions nor responses API."
#         )
#
#     def b64_encode_image(self, img):
#         buffered = io.BytesIO()
#         img.save(buffered, format="PNG")
#         return base64.b64encode(buffered.getvalue()).decode("utf-8")
#
#     @timeout_limit(5 * 60)
#     def __call__(self, images=None, query=None, **kwargs):
#         system_prompt = kwargs.get('system_prompt', 'You are a helpful assistant.')
#         max_tokens = kwargs.get('max_tokens', None)
#         temperature = kwargs.get('temperature', 0)
#
#         if self.use_chat:
#             image_blocks = []
#             if images:
#                 for img in images:
#                     b64 = self.b64_encode_image(img)
#                     image_blocks.append({
#                         "type": "image_url",
#                         "image_url": {"url": f"data:image/png;base64,{b64}"}
#                     })
#             input_blocks = image_blocks + [{"type": "text", "text": query}]
#             messages = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": input_blocks},
#             ]
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=messages,
#                 max_tokens=max_tokens,
#                 temperature=temperature,
#             )
#             return response.choices[0].message.content
#
#         elif self.use_responses:
#             image_blocks = []
#             if images:
#                 for img in images:
#                     b64 = self.b64_encode_image(img)
#                     image_blocks.append({
#                         "type": "input_image",
#                         "image_url": f"data:image/png;base64,{b64}"
#                     })
#             input_blocks = image_blocks + [{"type": "input_text", "text": query}]
#             messages = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": input_blocks},
#             ]
#             response = self.client.responses.create(
#                 model=self.model,
#                 input=messages,
#                 max_output_tokens=max_tokens,
#                 temperature=temperature,
#             )
#             return response.output_text
#
#
# class AnswerPaser:
#     def __init__(self):
#         self.paser = LLM('gpt-4.1-mini')
#
#     def get_parser_prompt(self, text: str, example: Union[str, list, dict]):
#         system_prompt = f"""
# You are an expert in structured data parsing. Your task is to convert text content into a standardized structured output based on a provided example data structure.
#
# ### Instructions
#
# 1.  **Analyze Example Structure:** Carefully analyze the example data structure provided within the `<example>` tags (e.g., it can be a dictionary, list, string, or single character) to understand the desired output format and hierarchy.
# 2.  **Determine Output Type:** Ensure the overall data type of the final output strictly adheres to the type specified within the `<type>` tags.
# 3.  **Transform Content:** Parse the text content from the `<input_text>` tags and transform it into a structured output that precisely matches the data format and content defined by `<example>`.
# 4.  **Preserve Semantics:** During the transformation process, only adjust the format and structure; do not alter the original semantic content of the text within the `<input_text>` tags.
# 5.  **Ignore Explanatory Text:** If the content within the `<input_text>` tags includes additional explanatory text or descriptions, ignore them and only extract and parse the core, final output data.
# 6.  **Clean Output:** Your final output must contain only the transformed structured content, without any additional explanations, descriptions, or irrelevant text and symbols.
#
# <example>
# {json.dumps(example, indent=4) if isinstance(example, list) or isinstance(example, dict) else example}
# </example>
#
# <type>
# {'One letter' if isinstance(example, str) and len(example) == 1 else type(example)}
# </type>
# """
#
#         query = f"""
# <input_text>
# {text}
# </input_text>
# """
#         return system_prompt, query
#
#     def __call__(self, text: str, example: Union[str, list, dict]):
#         if not isinstance(text, str):
#             text = str(text)
#         final_answer = extract_final_answer(text)
#         if final_answer is None:
#             final_answer = text
#
#         system_prompt, query = self.get_parser_prompt(final_answer, example)
#
#         output = self.paser(query=query, system_prompt=system_prompt)
#         return output
#
#
# def check_syntax(code_string):
#     try:
#         # Try to compile the code string
#         compile(code_string, '<string>', 'exec')
#         return True
#     except SyntaxError as e:
#         return False
#
#
# def get_function_lines(file_content):
#     node = ast.parse(file_content)
#
#     function_lines = {}
#
#     for item in node.body:
#         if isinstance(item, ast.FunctionDef):
#             func_name = item.name
#             start_line = item.lineno
#             end_line = item.end_lineno
#             function_lines[func_name] = (start_line, end_line)
#
#     return function_lines
#
#
# def replace_code(content_1, start_line_1, end_line_1, content_2, start_line_2, end_line_2):
#     lines_1 = content_1.splitlines(keepends=True)
#     lines_2 = content_2.splitlines(keepends=True)
#
#     lines_1[start_line_1 - 1:end_line_1] = lines_2[start_line_2 - 1:end_line_2]
#
#     return ''.join(lines_1)
#
#
# def replace_function(main_code, new_code, function_name):
#     assert check_syntax(main_code), "wrong main_code"
#     assert check_syntax(new_code), "wrong new_code"
#     functions_dict_1 = get_function_lines(main_code)
#     functions_dict_2 = get_function_lines(new_code)
#
#     start_line_1, end_line_1 = functions_dict_1[function_name]
#     start_line_2, end_line_2 = functions_dict_2[function_name]
#
#     main_code_after_replacing = replace_code(main_code, start_line_1, end_line_1, new_code, start_line_2, end_line_2)
#     assert check_syntax(main_code_after_replacing), "wrong main_code after replacing"
#     return main_code_after_replacing


############################## Idea Generation ##############################
def format_idea_data(idea_data):
    fields = [
        "Idea",
        "ImplementationSteps",
        "ImplementationOrder",
        "Dataset",
        "EvaluationMetrics",
        "ExpectedOutcome"
    ]

    formatted_text = ""
    for field in fields:
        if field in idea_data and idea_data[field]:
            formatted_text += f"{field}: {idea_data[field]}\n\n"

    return formatted_text.strip()


def get_context_from_data(data):
    context_fields = [
        "related_work",
        "challenge",
        "limitation",
        "motivation",
        "task_objective",
        "existing_solutions"
    ]
    context = ""
    for field in context_fields:
        if field in data and data[field]:
            context += f"{field}: {data[field]}\n\n"

    return context.strip()


def flip_evaluation_result(result):
    flipped = {}
    mapping = {
        "win_A": "win_B",
        "win_B": "win_A"
    }

    for key, value in result.items():
        if isinstance(value, dict) and "judgment" in value:
            flipped[key] = {
                "judgment": mapping.get(value["judgment"], value["judgment"]),
                "reason": value.get("reason", "")
            }
        else:
            flipped[key] = mapping.get(value, value)

    return flipped


def get_evaluation_prompt_modified(hypothesis_A, hypothesis_B, context=None):
    context_text = f"Context:\n{context}\n\n" if context else ""

    prompt = f"""
You are assisting researchers tasked with comparing TWO research hypotheses (Hypothesis A and Hypothesis B).
Your job is to evaluate both hypotheses across five separate dimensions defined below, and to choose a winner (either Hypothesis A or Hypothesis B) for each dimension. Ties are NOT allowed — you MUST pick one winner per dimension. Base your judgments on scientific principles and the provided context only.

##Background context:
{context_text}

##Hypothesis A:
{hypothesis_A}

##Hypothesis B:
{hypothesis_B}

##Definition of each dimension:
###1) Effectiveness
Which hypothesis is more likely to produce a successful experimental or empirical outcome in service of the stated research objective? Evaluate the likelihood that, if implemented using standard practices in the relevant discipline, the hypothesis will achieve the intended measurable result. Focus on mechanistic plausibility, causal logic, and whether the hypothesis addresses the core problem directly.

###2)Novelty
Novelty: Which hypothesis presents more innovative or original approaches? Compare the similarity between the idea and the related work and existing solutions in the background to assess its novelty. A lower similarity to the core idea indicates greater novelty.

###3) Detailedness (Level of Specification)
Which hypothesis provides clearer, more actionable, and more complete specification of mechanisms, assumptions, experimental steps, required variables, and dependencies? Detailedness rewards clarity that would enable a competent researcher to design an experiment or implementation with minimal ambiguity.

###4) Feasibility
Which hypothesis presents a more realistic and implementable solution given current technological constraints?

###5) Overall
Considering the overall aspects together but emphasizing conceptual coherence and scientific grounding, which hypothesis is superior overall? This is a synthesis judgment: prefer the hypothesis that is logically consistent, grounded in accepted principles, avoids critical unstated assumptions or contradictions, and is most defensible as a scientific proposition.

Unified constraints:
- Use only the provided context and widely accepted scientific principles in the relevant discipline. Do NOT invent facts external to the context unless they are broadly standard domain knowledge.
- When a dimension explicitly says to ignore other factors (e.g., Novelty should ignore feasibility), strictly follow that guidance for that dimension. When evaluating a certain dimension, it should focus on this dimension itself and ignore the influence of other dimensions.
- Be concise but specific: for each dimension provide a short judgment line (exact format below) plus 1–3 sentences of succinct reasoning grounded in the definitions above.
- Format must match exactly (case-insensitive for "Win A/Win B") and include a reason after "because".


##Output format (MUST FOLLOW EXACTLY)

Format your response exactly as follows:
Effectiveness: [Win A/Win B] because ...
Novelty: [Win A/Win B] because ...
Detailedness: [Win A/Win B] because ...
Feasibility: [Win A/Win B] because ...
Overall: [Win A/Win B] because ...
"""
    return prompt


def parse_evaluation_result(result):
    dimensions = ["effectiveness", "novelty", "detailedness", "feasibility", "overall"]
    parsed_results = {}
    all_valid = True

    for dim in dimensions:
        judgment = extract_win_lose(result, dim.capitalize())
        reason = extract_reason(result, dim.capitalize())

        if judgment is None:
            all_valid = False
            break

        parsed_results[dim] = {
            "judgment": judgment,
            "reason": reason
        }

    if not all_valid:
        return None

    return parsed_results


def extract_win_lose(result_text, dimension):
    pattern = rf"{dimension}\s*:\s*\[\s*(Win\s*A|Win\s*B)\s*\]"
    match = re.search(pattern, result_text, re.IGNORECASE)
    if match:
        judgment = match.group(1).strip().upper()
        if "WIN A" in judgment:
            return "win_A"
        else:
            return "win_B"

    backup_pattern = rf"{dimension}\s*:\s*(Win\s*A|Win\s*B)\s+"
    match = re.search(backup_pattern, result_text, re.IGNORECASE)
    if match:
        judgment = match.group(1).strip().upper()
        if "WIN A" in judgment:
            return "win_A"
        else:
            return "win_B"

    line_pattern = rf"{dimension}[^\n]*?(Win\s*A|Win\s*B)"
    match = re.search(line_pattern, result_text, re.IGNORECASE)
    if match:
        judgment = match.group(1).strip().upper()
        if "WIN A" in judgment:
            return "win_A"
        else:
            return "win_B"

    return None


def extract_reason(result_text, dimension):
    pattern = rf"{dimension}\s*:\s*\[[^\]]+\]\s*because\s*(.*?)(?=\n\w|$)"
    match = re.search(pattern, result_text, re.IGNORECASE | re.DOTALL)
    if match:
        reason = match.group(1).strip()
        return reason

    backup_pattern = rf"{dimension}\s*:[^\n]*?(because|due to|as|since)([^\n]+)"
    match = re.search(backup_pattern, result_text, re.IGNORECASE)
    if match:
        reason = match.group(2).strip()
        return reason

    fallback_pattern = rf"{dimension}\s*:[^\n]*(.*?)(?=\n\w+:|$)"
    match = re.search(fallback_pattern, result_text, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
        reason = re.sub(r"\[(Win\s*A|Win\s*B)\]", "", text).strip()
        return reason

    return "No specific reason provided"


############################## Idea Generation ##############################


def mean(l: list):
    assert len(l) > 0, "list length must > 0"
    return sum(l) / len(l)


def show_results(results: list[dict], metric_name: str, category_name: str = None, precision: int = 2, scale=1):
    category_dict = {}
    for item in results:
        if category_name is None:
            item_category = 'default'
        else:
            item_category = item[category_name]
        if item_category not in category_dict:
            category_dict[item_category] = []

        item_metric = float(item[metric_name])
        category_dict[item_category].append(item_metric)

    for k, v in category_dict.items():
        category_dict[k] = round(mean(v) * scale, precision)

    if category_name is None:
        return category_dict['default']
    else:
        return category_dict