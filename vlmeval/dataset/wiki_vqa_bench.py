import random

import pandas as pd

from .image_mcq import ImageMCQDataset


class WikiVQABench(ImageMCQDataset):
    """WikiVQA benchmark - Image MCQ with question, correct answer, and 3 wrong answers

    Data loaded from HuggingFace: ibm-research/WikiVQABench
    - question: str
    - correct: str
    - wrongs: list of 3 strings
    - image: str (base64 encoded image)

    Converts to standard image MCQ format with randomized option positions.
    """

    RANDOM_SEED = None

    DATASET_URL = {}
    DATASET_MD5 = {}

    def load_data(self, dataset):
        from datasets import load_dataset

        hf_dataset = load_dataset('ibm-research/WikiVQABench', split='train')

        random.seed(self.RANDOM_SEED)

        result_rows = []
        for idx, row in enumerate(hf_dataset):
            correct_answer = row['correct']
            wrong_answers = row['wrongs']

            all_options = [correct_answer] + list(wrong_answers)
            random.shuffle(all_options)

            correct_pos = all_options.index(correct_answer)
            correct_letter = chr(ord('A') + correct_pos)

            converted = {
                'index': idx,
                'question': row['question'],
                'A': all_options[0],
                'B': all_options[1],
                'C': all_options[2],
                'D': all_options[3],
                'answer': correct_letter,
                'image': row['image'],
            }
            result_rows.append(converted)

        return pd.DataFrame(result_rows)

    @classmethod
    def supported_datasets(cls):
        return ['WikiVQABench']
