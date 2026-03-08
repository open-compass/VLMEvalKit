import os
import json
from typing import Dict, List, Tuple, Any, Union
import pandas as pd
import warnings
import ast
import math
import os.path as osp

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import *
from vlmeval.smp.file import get_intermediate_file_path
from vlmeval.dataset.utils import build_judge
from vlmeval.dataset.utils.chartqapro import *


class ChartQAPro(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {
        "ChartQAPro": "https://opencompass.openxlab.space/utils/VLMEval/chartqapro.tsv",
        "ChartQAPro_CoT": "https://opencompass.openxlab.space/utils/VLMEval/chartqapro.tsv",
        "ChartQAPro_PoT": "https://opencompass.openxlab.space/utils/VLMEval/chartqapro.tsv",
    }
    DATASET_MD5 = {
        "ChartQAPro": "27653ea8dd8dd3a85bc4f432db96447a",
        "ChartQAPro_CoT": "27653ea8dd8dd3a85bc4f432db96447a",
        "ChartQAPro_PoT": "27653ea8dd8dd3a85bc4f432db96447a",
    }

    # deepseek-v3-0324, serve as extractor if needed
    DEFAULT_JUDGE = 'deepseek'
    JUDGE_FORMAT = '{model_name}_{dataset_name}_{judge_name}.tsv'
    RATING_FORMAT = '{model_name}_{dataset_name}_rating.json'

    def build_prompt(self, line: Union[int, pd.Series], qa_type: str = 'Direct') -> List[Dict[str, str]]:
        """
        Build a prompt for the model from a data line.

        Args:
            line: Either an index into the dataset or a pandas Series
            qa_type: Choose from ['Direct', 'CoT', 'PoT']

        Returns:
            List of message dictionaries containing the image and question
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = misc.toliststr(line["image"])
        else:
            tgt_path = self.dump_image(line)

        # determine qa_type, default value : 'Direct'
        if "CoT" in self.dataset_name:
            qa_type = "CoT"
        elif "PoT" in self.dataset_name:
            qa_type = "PoT"

        # load data line elements
        question = ast.literal_eval(line['question'])
        answer = ast.literal_eval(line['answer'])
        question_type = line['question_type']
        # image = line['image']
        # year = ast.literal_eval(line['year'])
        paragraph = line['paragraph']
        if paragraph != paragraph:  # treat nan
            paragraph = ''
        assert isinstance(question, list)
        assert len(tgt_path) == 1

        # build prompt from question
        question_context = prompt_context(question, answer, question_type, qa_type)

        # form messages
        msgs = []
        msgs = [dict(type='image', value=tgt_path[0])]
        msgs.append(dict(type='text', value=paragraph))
        msgs.append(dict(type='text', value=question_context))

        return msgs

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> pd.DataFrame:
        """
        Evaluate model predictions on the ChartQAPro dataset.

        Args:
            eval_file: Path to the file containing model predictions
            **judge_kwargs: Additional arguments for the judge model

        Returns:
            DataFrame with evaluation scores by category
        """
        nproc = judge_kwargs.pop('nproc', 16)

        model_name = judge_kwargs['model']

        tgt_file = get_intermediate_file_path(eval_file, f"_{model_name}", 'tsv')
        rating_file = get_intermediate_file_path(eval_file, "_rating", 'json')

        if osp.exists(tgt_file):
            data = load(tgt_file)
        else:
            model = build_judge(**judge_kwargs)
            data = load(eval_file)
            data['prediction'] = data['prediction'].astype(str)
            jobs = [dict(model=model, line=line) for _, line in data.iterrows()]
            ret = track_progress_rich(
                ChartQAPro_auxeval,
                jobs,
                nproc=nproc,
                desc='ChartQAPro Evaluation'
            )
            for k in ['score', 'extracted']:
                data[k] = [item[k] for item in ret]
            dump(data, tgt_file)
        cates = list(set(data['question_type']))
        score = {}
        for c in cates:
            score[c] = np.mean(data[data['question_type'] == c]['score'])
        score['Overall'] = np.mean(data['score'])
        dump(score, rating_file)
        return score

    @classmethod
    def report_score(cls, model_name, dataset_name, root, verbose=False, **kwargs):
        rating_file = cls.RATING_FORMAT.format(model_name=model_name, dataset_name=dataset_name)
        rating_file = osp.join(root, rating_file)
        rating = load(rating_file)
        rating = {k: v * 100 for k, v in rating.items()}
        res = {'overall': rating['Overall']}
        if verbose:
            res['rating'] = rating
        return res


EXTRACT_PROMPT = """\
You are an AI assistant to help me extract the final answer from a model prediction. \
You will be provided with a question and a model prediction. \
Your should output **one single short answer** without any additional content. \
If there are multiple final answers, put them in brackets using this format [’Answer1’, ’Answer2’].

Example 1:
Question: estimate the year in which wind capacity first exceeds 100 gw based on the trend shown in the chart.
Prediction: Based on the chart, the "Wind" capacity is represented by the green bars. In a stacked bar chart, the value is typically read from the top of the segment to understand the cumulative capacity up to that point, or the segment height for individual capacity.\n\nThe individual height of the green "Wind" segment does not appear to exceed 100 GW on its own (it is roughly 70-75 GW at its peak in 2049-50). However, interpreting the question as asking when the cumulative capacity level *reached by the Wind segment* (the top of the green bar) crosses the 100 GW threshold:\n\n1.  Locate the 100 GW line on the Y-axis.\n2.  Trace the top of the green bars across the years.\n3.  In **2035-36**, the top of the green bar is clearly below the 100 GW line.\n4.  In **2036-37**, the top of the green bar is just touching or slightly below the line.\n5.  In **2037-38**, the top of the green bar clearly exceeds the 100 GW line.\n\nTherefore, the year in which the level first clearly exceeds 100 GW is 2037-38.\n\nAnswer: 2037-3
Your output: 2037-3

Example 2:
Question: which player had the secnd highest rating average at age 21?
Prediction: Marcus Bontempelli
Your output: Marcus Bontempelli

Your Task:
Question: {question}?\nPrediction: {prediction}\nYour output:
"""  # noqa: E501


def ChartQAPro_auxeval(model, line):
    answer_list = ast.literal_eval(line['answer'])
    year_list = ast.literal_eval(line['year'])
    prediction = line['prediction'].strip(".")
    # will only use the last answer
    answer = answer_list[-1].strip('.').strip("\n")
    q_type = line['question_type']
    if q_type == 'Conversational':
        year = year_list[-1:]
        # year = year_list
    else:
        year = year_list
    questions = ast.literal_eval(line['question'])
    question = questions[-1] if q_type == 'Conversational' else questions[0]
    extracted = None

    score = relaxed_correctness_chartqapro(answer, prediction, year_flags=year)
    if score != 1 and len(prediction) > len(answer):
        # Will Give it another try w. LLM extraction
        extract_prompt = EXTRACT_PROMPT.format(question=question, prediction=prediction)
        extracted = model.generate(extract_prompt)
        score2 = relaxed_correctness_chartqapro(answer, extracted, year_flags=year)
        if score2 > score:
            score = score2
        else:
            extracted = None

    return dict(score=score, extracted=extracted)
