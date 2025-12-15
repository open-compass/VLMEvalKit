from typing import Any, Dict, List
from datasets import load_dataset
from ...smp import *
from ..text_base import TextBaseDataset


def extract_final_answer(answer_with_thinking: str, start_tag='<answer>', end_tag='</answer>'):
    answer_with_thinking = str(answer_with_thinking)
    start_index = answer_with_thinking.rfind(start_tag)
    if start_index != -1:
        end_index = answer_with_thinking.find(end_tag, start_index)
        if end_index != -1:
            return answer_with_thinking[start_index + len(start_tag):end_index].strip()
    return None


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
        
        data['exact_match'] = 0
        exact_match_count = 0
        for index, row in data.iterrows():
            if extract_final_answer(row['prediction']) == row['answer']:
                data.loc[index, 'exact_match'] = 1
                exact_match_count += 1

        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        result = {"Exact_Match": exact_match_count/len(data)}
        result_file = get_intermediate_file_path(eval_file, '_result', 'json')
        dump(data, score_file)
        dump(result, result_file)
        return result