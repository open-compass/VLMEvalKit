import re
from typing import Any, Dict, List
from datasets import load_dataset
from vlmeval import *

from openai import OpenAI

from ...smp import *
from ..image_base import ImageBaseDataset

def extract_answer_from_response(response):
    match = re.search(r"\\boxed\{([A-Z])\}", response)
    if match:
        return match.group(1)
    else:
        return None


def mm_reasoning_is_correct(pred, gold):
    try:
        ans = extract_answer_from_response(pred).strip()
    except:
        return False
    return ans.lower() == gold.lower()


def b64_encode_image(img) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class VLM:
    def __init__(self, model='gpt-4.1', **kwargs):
        self.api_key = kwargs.get('api_key', os.environ.get('OPENAI_API_KEY')) # export OPENAI_API_KEY="xxxxx"
        self.base_url = kwargs.get('base_url', os.environ.get('OPENAI_API_BASE')) # export OPENAI_BASE_URL="xxxxx"
        self.base_url = self.base_url[:-17]
        self.model = model
        if not self.api_key:
            raise ValueError("API key is required.")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def __call__(self, images=None, query=None, **kwargs):
        system_prompt = kwargs.get('system_prompt', 'You are a helpful assistant.')
        max_tokens = kwargs.get('max_tokens', None)
        temperature = kwargs.get('temperature', 0)

        image_msgs = []
        if images is not None:
            for img in images:
                b64 = b64_encode_image(img)
                image_msgs.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": image_msgs + [{"type": "text", "text": query}]},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        assistant_response = response.choices[0].message.content
        return assistant_response


judge = VLM('o4-mini')


class SGI_Bench_Experimental_Reasoning(ImageBaseDataset):
    TYPE = 'MCQ'

    @classmethod
    def supported_datasets(cls):
        return ["SGI-Experimental-Reasoning"]

    def load_data(self, dataset):
        hf = load_dataset("InternScience/SGI-Reasoning", split="test")

        rows: List[Dict[str, Any]] = []
        idx = 0
        for prob in hf:
            rows.append(
                {
                    "index": idx,
                    "id": prob["idx"],
                    "question": prob["question"],
                    "image": [encode_image_to_base64(img) for img in prob["images"]],
                    "options": prob["options"],
                    "steps": prob["steps"],
                    "step_images": [encode_image_to_base64(img) for img in prob["step_images"]],
                    "answer": prob["answer"],
                    "image_type": prob["image_type"],
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
        tgt_path = self.dump_image(line)
        question = """
Please solve the following multiple-choice question step-by-step. Each question is provided with several options labeled A, B, C, D, E, etc. Carefully analyze the question and each option, reason step-by-step, then select the single most correct option.

Your final output **must** include both **the reasoning** and **the final answer**. The final answer must meet two core requirements:
1. It consists solely of the corresponding letter of the correct option (e.g., A, B, C, D, E, etc.);
2. This letter is enclosed in the \\boxed{} format. Example: \\boxed{A}
    """.strip() + "\n\nQuestion:\n" + line['question'] + "\n\nOptions:\n"
        for i, option in enumerate(line['options']):
            option_label = chr(ord('A') + i)
            question += f"{option_label}. {option}\n"

        msgs = []
        for p in tgt_path:
            msgs.append({'type': 'image', 'value': p})
        msgs.append({'type': 'text', 'value': question})
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data = pd.DataFrame(data)

        data['mcc'] = 0
        data['rv'] = 0

        all_mcc, all_rv = [], []
        for index, row in data.iterrows():
            reference_steps = row["steps"]
            reference_steps = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(reference_steps)])

            judge_prompt = f"""You are a strict evaluator assessing the **validity of the model prediction's reasoning process**. You must score this reasoning validity on a scale from 0 to 10, where 0 means the reasoning is completely invalid and 10 means the reasoning is fully rigorous.
# Input
Question:
```
{row['question']}
```
Reference Reasoning:
```
{reference_steps}
```
Model Prediction:
```
{row['prediction']}
```
# Evaluation Rules
1. First, identify the **complete reasoning process** from the model prediction (ignore only the final answer if it is not accompanied by reasoning).
2. Evaluate reasoning validity against two core criteria:
   - **Logical Coherence**: Check if the reasoning steps are sequential, self-consistent, and free of contradictions (e.g., no conflicting premises or illogical deductions).
   - **Alignment with Reference Reasoning**: Check if the reasoning direction, key premises, and deduction logic match the reference reasoning (partial alignment counts for partial credit).
3. Deduct points for:
   - Irrelevant content (reasoning that does not address the question or key conditions).
   - Missing key reasoning steps (even if the final answer is correct).
   - Flawed logic (e.g., circular reasoning, false premises leading to conclusions).
4. Do not prioritize the correctness of the **final answer**—a correct answer with invalid reasoning still scores low, while an incorrect answer with partially valid reasoning may score higher.
# Scoring Guide
- **10**: Reasoning is fully rigorous, logically coherent (no contradictions), and perfectly aligned with the reference reasoning (all key steps and logic match).
- **7-9**: Reasoning is mostly coherent, with minor logical gaps or partial misalignment with the reference reasoning (no major contradictions).
- **4-6**: Reasoning has obvious logical flaws (e.g., one missing key step, minor contradictions) or limited alignment with the reference reasoning (only some core logic matches).
- **1-3**: Reasoning is barely valid, with severe logical flaws (e.g., multiple contradictions) or almost no alignment with the reference reasoning (only tangentially related to the question).
- **0**: Reasoning is completely invalid, contradictory (self-conflicting logic), or irrelevant (no connection to the question or key conditions).
# Strict Output format example
6"""

            try:
                images = [decode_base64_to_image(b64) for b64 in row['step_images'] ]
                llm_judge = judge(images, judge_prompt).strip()
                pattern = r"(\d+)"
                match = re.search(pattern, llm_judge)
                rv_score = float(match.group(1)) if match else 0.0
            except:
                rv_score = 0.0

            mcc_score = mm_reasoning_is_correct(row['prediction'], chr(ord('A') + int(row['answer'])))

            all_mcc.append(mcc_score)
            data.at[index, 'mcc'] = 1 if mcc_score else 0
            all_rv.append(rv_score)
            data.at[index, 'rv'] = rv_score / 10.0

        
        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        result = {"MCC": sum(all_mcc)/len(all_mcc), "RV": sum(all_rv)/(10.0*len(all_rv))}
        result_file = get_intermediate_file_path(eval_file, '_result', 'json')
        dump(data, score_file)
        dump(result, result_file)
        return result
