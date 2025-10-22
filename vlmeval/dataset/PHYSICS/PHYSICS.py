import os
import pandas as pd
import numpy as np
from jinja2 import Template
from functools import partial
from transformers import AutoTokenizer, Qwen2Tokenizer
import os
import json
import warnings
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import argparse
import contextlib
from tqdm import tqdm
from collections import defaultdict
from reward_score import compute_score
from reward_manager import verifier_manager
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        
class PHYSICS():
    DATASET_URL = {
        'PHYSICS-test': 'https://huggingface.co/datasets/desimfj/PHYSICS'
    }

    def load_data(path, read_num=None, repeat_time=1):
        name = os.path.basename(path).replace('.jsonl', '')
        df = pd.read_json(path, lines=True)
        df['dataset'] = name

        # 采样（若指定 read_num）
        sample_kwargs = {'n': read_num} if read_num else {'frac': 1}
        df = df.sample(**sample_kwargs).drop_duplicates(subset=['question']).reset_index(drop=True)

        # 重复（若指定 repeat_time）
        if repeat_time > 1:
            df = df.loc[np.repeat(df.index, repeat_time)].reset_index(drop=True)

        return df


    def build_prompt(line):
        SYSTEM_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it.
    The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags."""
        
        INSTRUCTION_TEMPLATE = """Below is an open-ended problem in undergraduate-level Physics. Please answer this problem adhering to the following rules:
    1. Use LaTeX format for formulas.
    2. Put the final answer(s) in \\boxed{}, without units.
    3. If multiple answers exist, separate them by commas in \\boxed{}.
    Problem: {{prompt}}"""

        # 获取问题文本（支持 Series 或 dict）
        prompt_text = line["question"] if isinstance(line, dict) else line

        # 应用指令模板
        prompt_text = Template(INSTRUCTION_TEMPLATE).render(prompt=prompt_text)

        # 拼接最终的完整 prompt
        full_prompt = (
            f"System: {SYSTEM_PROMPT}\n\n"
            f"User: {prompt_text}\n"
        )

        return full_prompt
    
    def write_jsonl(data_path, dataset, indent=0, mode='w'):
        with open(data_path, mode, encoding='UTF-8') as f:
            if not isinstance(dataset, list):
                dataset = [dataset]
            for data in dataset:
                line = json.dumps(data, ensure_ascii=False, indent=indent if indent != 0 else None)
                f.write(line + '\n')
            
    def evaluate(self, eval_file, **judge_kwargs):
        output_path = '../logs/phyics-test.log'
        for item in tqdm(eval_file, desc="Scoring"):
            # result = compute_score(item['model_output'], item['ground_truth'], item['problem']) # olympiadbench 
            try:
                with timeout(30):
                    result = compute_score(item['test_result'], item['answer'], item['question'], **judge_kwargs)
                    item['rule_based_acc'] = result['rule_based_acc']
                    item['acc'] = result['acc']
                    item['extracted_gt'] = result['extracted_gt']
                    item['extracted_pred'] = result['extracted_pred']
                    self.write_jsonl(output_path, item, mode='a')
            except TimeoutError:
                print(f"Timeout processing item: {item.get('question', 'Unknown')}")
                continue
        return eval_file