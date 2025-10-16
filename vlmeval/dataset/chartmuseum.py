import os
import json
from typing import Dict, List, Tuple, Any, Union
import pandas as pd
import warnings

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import misc, file
from vlmeval import utils
from vlmeval.dataset.utils import build_judge
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset
import re
from collections import defaultdict
from vlmeval.smp.file import get_intermediate_file_path


# gpt4_key = "your-key"
# client = OpenAI(api_key = gpt4_key)

COMPARE_ANSWER_PROMPT = """You are provided with a question and two answers. Please determine if these answers are equivalent. Follow these guidelines:

1. Numerical Comparison:
   - For decimal numbers, consider them as equivalent if their relative difference is sufficiently small. 
   For example, the following pairs are equivalent:
    - 32.35 and 32.34
    - 90.05 and 90.00
    - 83.3% and 83.2%
    - 31 and 31%
   The following pairs are not equivalent:
   - 32.35 and 35.25
   - 90.05 and 91.05
   - 83.3% and 45.2%

   Note that if the question asks for years or dates, please do the exact match with no error tolerance.

2. Unit Handling:
   - If only one answer includes units (e.g. '$', '%', '-', etc.), ignore the units and compare only the numerical values
   For example, the following pairs are equivalent:
   - 305 million and 305 million square meters
   - 0.75 and 0.75%
   - 0.6 and 60%
   - $80 and 80
   The following pairs are not equivalent:
   - 305 million and 200 million square meters
   - 0.75 and 0.90%

3. Text Comparison:
   - Ignore differences in capitalization
   - Treat mathematical expressions in different but equivalent forms as the same (e.g., "2+3" = "5")

Question: [QUESTION]
Answer 1: [ANSWER1]
Answer 2: [ANSWER2]

Please respond with:
- "Yes" if the answers are equivalent
- "No" if the answers are different"""


def get_question(QUESTION):

    QA_PROMPT = f"""Please answer the question using the chart image.

    Question: {QUESTION}

    Please first generate your reasoning process and then provide the user with the answer. Use the following format:

    <think> 
    ... your thinking process here ... 
    </think> 
    <answer> 
    ... your final answer (entity(s) or number) ...
    </answer>"""

    return QA_PROMPT

def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text + "</answer>", re.DOTALL)
    return m.group(1).strip() if m else ""

def gpt_compare(question: str, answer1: str, answer2: str) -> bool:

    prompt = (
        COMPARE_ANSWER_PROMPT
        .replace("[QUESTION]", question)
        .replace("[ANSWER1]", answer1)
        .replace("[ANSWER2]", answer2)
    )
    resp = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14"  ,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return "yes" in resp.choices[0].message.content.strip().lower()


class ChartMuseum(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {
        "ChartMuseum_dev": "https://huggingface.co/datasets/yujieouo/ChartMuseum/blob/main/ChartMuseum_dev.tsv",
        "ChartMuseum_test": "https://huggingface.co/datasets/yujieouo/ChartMuseum/blob/main/ChartMuseum_test.tsv",
    }
    DATASET_MD5 = {
        "ChartMuseum_dev": "05dbce1f4bd5e5ba0e4b0d606efb707e",
        "ChartMuseum_test": "983586eace6ee33cdb189d63124768c8",
    }

    def build_prompt(self, line: Union[int, pd.Series], qa_type: str = 'test') -> List[Dict[str, str]]:
        """
        Build a prompt for the model from a data line.

        Args:
            line: Either an index into the dataset or a pandas Series
            qa_type: Choose from ['dev', 'test']

        Returns:
            List of message dictionaries containing the image and question
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = misc.toliststr(line["image"])
        else:
            tgt_path = self.dump_image(line)

        # determine qa_type, default value : 'test' 
        if "dev" in self.dataset_name:
            qa_type = "dev"
        elif "test" in self.dataset_name:
            qa_type = "test"

        # load data line elements
        question = line['question']
        answer = line['answer']
        category = line['category']

        # build prompt from question
        question_context = get_question(question)

        # form messages
        msgs = []
        msgs = [dict(type='image', value=tgt_path[0])]
        msgs.append(dict(type='text', value=question_context))

        return msgs

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> pd.DataFrame:
        """
        Evaluate model predictions on the ChartMuseum dataset.
        Args:
            eval_file: Path to the file containing model predictions
            **judge_kwargs: Additional arguments for the judge model
        Returns:
            DataFrame with evaluation scores by category
        """

        if "dev" in self.dataset_name:
            qa_type = "dev"
        elif "test" in self.dataset_name:
            qa_type = "test"

        benchmark = pd.DataFrame(load_dataset("lytang/ChartMuseum")[qa_type])
        questions = benchmark["question"].tolist()
        gts = benchmark["answer"].astype(str).tolist()
        categories = benchmark['reasoning_type'].astype(str).tolist()

        data = file.load(eval_file)
        pred_list = data['prediction'].astype(str).tolist()
        pred_answers = [extract_answer(p) for p in pred_list]

        category_flags = defaultdict(list)

        for cat, q, gt, pa in tqdm(zip(categories, questions, gts, pred_answers), total=len(pred_answers)):
            flag = int(gpt_compare(q, gt, pa))
            category_flags[cat].append(flag)

        score = {}
        for cat, flags in category_flags.items():
            score[cat] = [sum(flags) / len(flags) * 100]

        all_flags = [f for flags in category_flags.values() for f in flags]
        score["Overall"] = [sum(all_flags) / len(all_flags) * 100]
        score_file = get_intermediate_file_path(eval_file, "_acc", "csv")
        out_score = pd.DataFrame(score)
        file.dump(out_score, score_file)
        
        return out_score