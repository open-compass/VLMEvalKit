import os
import pandas as pd
import string
from ..smp import load, dump, get_cache_path
from .image_base import ImageBaseDataset
from .utils import build_judge, parse_choice

FAIL_MSG = 'Failed to obtain answer via API.'


class M4Bench(ImageBaseDataset):
    """
    Dataset class for M4Bench, handling single and dual image inputs.
    """
    TYPE = 'M4Bench'

    DATASET_URL = {
        "State_Invariance": "https://huggingface.co/datasets/Anonymous8976/M4Bench/resolve/main/State_Invariance.tsv",  # noqa: E501
        "State_Comparison": "https://huggingface.co/datasets/Anonymous8976/M4Bench/resolve/main/State_Comparison.tsv",  # noqa: E501
        "Spatial_Perception": "https://huggingface.co/datasets/Anonymous8976/M4Bench/resolve/main/Spatial_Perception.tsv",  # noqa: E501
        "Instance_Comparison": "https://huggingface.co/datasets/Anonymous8976/M4Bench/resolve/main/Instance_Comparison.tsv",  # noqa: E501
        "Detailed_Difference": "https://huggingface.co/datasets/Anonymous8976/M4Bench/resolve/main/Detailed_Difference.tsv"  # noqa: E501
    }

    DATASET_MD5 = {
        "State_Invariance": "f61bf638ac2b7ab05924c4d2be65e09e",
        "State_Comparison": "b4d2a4020f1bb9452e2d5a3b2e1b89c4",
        "Spatial_Perception": "30ffa09a224aac6f475509f95f3cc020",
        "Instance_Comparison": "f86d792a78962b10564616be34652952",
        "Detailed_Difference": "107ae999b60bd6762229d6b10edabf3d"
    }

    def build_prompt(self, line):
        """
        Builds a multimodal prompt for the given data line.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = [line['image1_path']]
            if 'image2_path' in line and not pd.isna(line['image2_path']):
                tgt_path.append(line['image2_path'])
        else:
            tgt_path = [self.dump_image(line, 'image_1')]
            if 'image_2' in line and self.has_image(line, 'image_2'):
                tgt_path.append(self.dump_image(line, 'image_2'))

        query = line['query']

        msgs = []
        if len(tgt_path):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        msgs.append(dict(type='text', value=query))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluates the model predictions against the ground truth.
        """
        results_df = load(eval_file)

        dataset_name = None
        for name in self.DATASET_URL:
            if name in eval_file:
                dataset_name = name
                break

        if dataset_name is None:
            raise ValueError(
                f"Could not determine dataset name from eval_file path: {eval_file}")

        # Load ground truth data
        gt_file = get_cache_path(self.DATASET_URL[dataset_name])
        gt_df = pd.read_csv(gt_file, sep='\t', on_bad_lines='warn')

        # Merge predictions with ground truth
        df = pd.merge(results_df, gt_df, on='index')

        if judge_kwargs:
            # Use LLM as a judge to parse the prediction
            judge = build_judge(**judge_kwargs)

            # Prepare data for the judge
            def extract_question(q):
                return q.split('\n(')[0]

            def extract_options(q):
                parts = q.split('\n(')
                return '\n('.join(parts[1:]) if len(parts) > 1 else ''

            df['question_text'] = df['query'].apply(extract_question)
            df['options_text'] = df['query'].apply(extract_options)

            prompt_tmpl = (
                'You are an AI assistant who will help me to match '
                'an answer with several options of a single-choice question. '
                'You are provided with a question, several options, and an answer, '
                'and you need to find which option is most similar to the answer. '
                'If the meaning of all options are significantly different from the answer, output Z. '
                'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
                'Example 1: \n'
                'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
                'Answer: a cute teddy bear\nYour output: A\n'
                'Example 2: \n'
                'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
                'Answer: Spider\nYour output: Z\n'
                'Example 3: \n'
                'Question: {question}\nOptions: {options}\nAnswer: {prediction}\nYour output: '
            )

            df['parsed_pred'] = judge.score(
                question=df['question_text'].tolist(),
                options=df['options_text'].tolist(),
                prediction=df['prediction'].tolist(),
                prompt_tmpl=prompt_tmpl
            )['text']

        else:
            # Fallback to simple parsing if no judge is provided
            options = {chr(ord('A') + i) for i in range(26)}
            df['parsed_pred'] = [
                parse_choice(
                    pred,
                    options) for pred in df['prediction']]

        # Calculate score
        df['score'] = (df['parsed_pred'] == df['response'])

        # Save detailed results
        base_name = os.path.splitext(os.path.abspath(eval_file))[0]
        details_file = base_name + '_details.xlsx'
        dump(df, details_file)

        # Calculate and return accuracy
        acc = df['score'].mean() * 100
        results = {'acc': acc, 'details': details_file}

        return results
