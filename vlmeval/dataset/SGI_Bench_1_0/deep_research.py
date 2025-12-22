from typing import Any, Dict, List
from datasets import load_dataset
from ...smp import *
from ..text_base import TextBaseDataset
from ..utils.judge_util import *
from ...smp.file import dump, load, get_intermediate_file_path
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm
from json_repair import repair_json
import os


def extract_final_answer(answer_with_thinking: str, start_tag='<answer>', end_tag='</answer>'):
    answer_with_thinking = str(answer_with_thinking)
    start_index = answer_with_thinking.rfind(start_tag)
    if start_index != -1:
        end_index = answer_with_thinking.find(end_tag, start_index)
        if end_index != -1:
            return answer_with_thinking[start_index + len(start_tag):end_index].strip()
    return None


class LLM:
    def __init__(self, model='gpt-4.1', **kwargs):
        self.api_key = kwargs.get('api_key', os.environ.get('OPENAI_API_KEY'))  # export OPENAI_API_KEY="xxxxx"
        self.base_url = kwargs.get('base_url', os.environ.get('OPENAI_API_BASE'))
        self.base_url = self.base_url[:-17]  # export OPENAI_BASE_URL="xxxxx"
        self.model = model
        if not self.api_key:
            raise ValueError("API key is required.")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def __call__(self, query=None, **kwargs):
        system_prompt = kwargs.get('system_prompt', 'You are a helpful assistant.')
        max_tokens = kwargs.get('max_tokens', None)
        temperature = kwargs.get('temperature', 0)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        assistant_response = response.choices[0].message.content
        return assistant_response


def multi_process(inp_list, function, max_workers=40):
    results = [None] * len(inp_list)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(function, **item): index
            for index, item in enumerate(inp_list)
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(future_to_index)):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                print(f"Error processing item {inp_list[index]}: {str(e)}")

    return results


judge = LLM('o4-mini')


def eval_model_output(ques_dict):
    newline = '\n'
    prompt = f"""
You are an expert in systematically validating and evaluating LLM-generated solutions. Your task is to rigorously analyze the correctness of a provided solution by comparing it step-by-step against the reference solution, and output **only** a structured verification list—with no additional text.

## Instructions  
1. Break down the given LLM solution into individual steps and evaluate each one against the corresponding reference solution steps.  
2. For each step, include the following three components:  
   - **solution_step**: The specific part of the LLM solution being evaluated.  
   - **reason**: A clear, critical explanation of whether the step contains errors, omissions, or deviations from the reference approach. Be stringent in your assessment.  
   - **judge**: Your verdict: either `"correct"` or `"incorrect"`.  
3. If the final LLM answer is incorrect, you must identify at least one step in your analysis as incorrect.  
4. Justify your judgments rigorously, pointing out even minor inaccuracies or logical flaws.  
5. Do not attempt to answer the original question—your role is strictly to evaluate.  
6. Output **only** a list of dictionaries in the exact format provided below. Do not include any other text or comments.

## Question  
{ques_dict['question']}

## Reference Solution Steps  
{newline.join(ques_dict['steps'])}

## Reference Answer  
{ques_dict['answer']}

## LLM Solution Steps
{ques_dict['prediction']}

## LLM Answer
{extract_final_answer(ques_dict['prediction'])}

## Output Example  
[  
    {{"solution_step": "step content", "reason": "reason of the judgement", "judge": "correct or incorrect"}},  
    {{"solution_step": "step content", "reason": "reason of the judgement", "judge": "correct or incorrect"}},
]
"""

    try:
        llm_judge = judge(prompt)
        start_index = llm_judge.find('[')
        end_index = llm_judge.rfind(']') + 1
        llm_judge = eval(repair_json(llm_judge[start_index:end_index]))
        correct_step_count = 0
        for step in llm_judge:
            if step["judge"] == "correct":
                correct_step_count += 1
        step_level_acc = correct_step_count / len(llm_judge)
    except:
        llm_judge = None

    ques_dict['exact_match'] = 1 if (
                ques_dict['answer'] == ques_dict['prediction'] or ques_dict['answer'] == extract_final_answer(
            ques_dict['prediction'])) else 0
    ques_dict['llm_judge'] = llm_judge
    ques_dict['step_level_acc'] = step_level_acc
    return ques_dict


class SGI_Bench_Deep_Research(TextBaseDataset):
    TYPE = 'QA'

    @classmethod
    def supported_datasets(cls):
        return ["SGI-DeepResearch"]

    def load_data(self, dataset):
        hf = load_dataset("InternScience/SGI-DeepResearch",split="test")

        rows: List[Dict[str, Any]] = []
        idx = 0
        for prob in hf:
            rows.append(
                {
                    "index": idx,
                    "id": prob["idx"],
                    "question": prob["question"],
                    "steps": prob["steps"],
                    "answer": prob["answer"],
                    "discipline": prob["discipline"],
                    "direction": prob["direction"],
                    "type": prob["type"]
                }
            )
            idx += 1
        return pd.DataFrame(rows)

    
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        question = line['question'] + """
You can reason step by step before giving the final answer. The final answer should be enclosed by <answer> and </answer>.

Example:
Step 1. ...
Step 2. ...
...
<answer>1.00</answer>
"""

        msgs = [{'type': 'text', 'value': question}]
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data = pd.DataFrame(data)

        inp_list = [{"ques_dict": item} for item in data.to_dict(orient="records")]
        out_list = multi_process(inp_list, eval_model_output, 48)

        exact_match = sum([item['exact_match'] for item in out_list]) / len(out_list)
        step_level_acc = sum([item['step_level_acc'] for item in out_list]) / len(out_list)

        result = {
            'Exact Match': exact_match,
            'Step Level Acc': step_level_acc
        }

        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        result_file = get_intermediate_file_path(eval_file, '_result', 'json')
        dump(out_list, score_file)
        dump(result, result_file)
        return result