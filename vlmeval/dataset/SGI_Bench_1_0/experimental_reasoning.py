import ast
import re
from typing import Any, Dict, List
from datasets import load_dataset
import pandas as pd
from ..utils.judge_util import build_judge
from ...smp.vlm import encode_image_to_base64 ,read_ok,decode_base64_to_image_file
from ...smp.misc import toliststr
from ...smp.file import load ,dump , get_intermediate_file_path
import os
import io
import base64
import os.path as osp
from ..image_base import ImageBaseDataset

def extract_answer_from_response(response):
    match = re.search(r"\\boxed\{([A-Za-z])\}", response)
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

class SGI_Bench_Experimental_Reasoning(ImageBaseDataset):
    TYPE = 'MCQ '

    @classmethod
    def supported_datasets(cls):
        return ["SGI-Experimental-Reasoning"]

    def dump_images(self, line):
        step_dir = osp.join(self.img_root, 'step_images')
        os.makedirs(self.img_root, exist_ok=True)
        os.makedirs(step_dir, exist_ok=True)

        results = {}
        def _process_field(key_name, path_key_name, save_root):
            tgt_paths = []
            if key_name in line:
                content = line[key_name]
                if path_key_name in line and isinstance(line[path_key_name], list):
                    fnames = line[path_key_name]
                else:
                    count = len(content) if isinstance(content, list) else 1
                    fnames = [f"{line['index']}_{i}.png" for i in range(count)]
                imgs = content if isinstance(content, list) else [content]
                for img, fname in zip(imgs, fnames):
                    full_path = osp.join(save_root, fname)
                    if not read_ok(full_path):
                        decode_base64_to_image_file(img, full_path)
                    tgt_paths.append(full_path)

            elif path_key_name in line:
                paths = toliststr(line[path_key_name])
                read_ok_flag = [read_ok(x) for x in paths]

                if not all(read_ok_flag):
                    paths_abs = [osp.join(save_root, x) for x in paths]
                    read_ok_flag = [read_ok(x) for x in paths_abs]
                    assert read_ok_flag, f"Field `{key_name}` missing and files not found: {paths}"
                    tgt_paths = paths_abs
                else:
                    tgt_paths = paths

            return tgt_paths

        if 'image' in line or 'image_path' in line:
            results['image'] = _process_field('image', 'image_path', self.img_root)
        if 'step_images' in line or 'step_image_path' in line:
            results['step_images'] = _process_field('step_images', 'step_image_path', step_dir)

        return results

    def load_data(self, dataset):
        hf = load_dataset("InternScience/SGI-Reasoning", split="test")

        rows: List[Dict[str, Any]] = []
        idx = 0

        for prob in hf:
            current_row = {
                "index": idx,  #
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
            saved_paths = self.dump_images(current_row)
            if 'image' in saved_paths:
                current_row['image'] = saved_paths['image']

            if 'step_images' in saved_paths:
                current_row['step_images'] = saved_paths['step_images']
            rows.append(current_row)
            idx += 1

        return pd.DataFrame(rows)

    
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
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
        if isinstance(line['image'], list):
            for p in line['image']:
                msgs.append({'type': 'image', 'value': p})
        elif isinstance(line['image'], str):
            msgs.append({'type': 'image', 'value': line['image']})
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
            if judge_kwargs.get('model') is None:
                judge_kwargs['model'] = 'o4-mini'
            if judge_kwargs.get('max_tokens') is None:
                judge_kwargs['max_tokens'] = None
            judge = build_judge(**judge_kwargs)
            try:
                msgs = []
                msgs.append({'role': 'system', 'value': 'You are a helpful assistant.'})
                msgs.append({'role': 'user', 'type':'text' ,'value': judge_prompt})

                images = ast.literal_eval(row['step_images'])
                for image in images:
                    msgs.append({'role': 'user', 'value': image, 'type': 'image'})
                llm_judge = judge.generate(msgs).strip()
                pattern = r"(\d+)"
                match = re.search(pattern, llm_judge)
                rv_score = float(match.group(1)) if match else 0.0
            except Exception as e:
                rv_score = 0.0

            mcc_score = mm_reasoning_is_correct(row['prediction'], chr(ord('A') + int(row['answer'])))

            all_mcc.append(mcc_score)
            data.at[index, 'mcc'] = 1 if mcc_score else 0
            all_rv.append(rv_score)
            data['rv'] = data['rv'].astype(float)
            data.at[index, 'rv'] = rv_score / 10.0

        
        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        result = {"MCC": sum(all_mcc)/len(all_mcc), "RV": sum(all_rv)/(10.0*len(all_rv))}
        result_file = get_intermediate_file_path(eval_file, '_result', 'json')
        dump(data, score_file)
        dump(result, result_file)
        return result
