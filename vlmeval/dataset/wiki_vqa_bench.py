import os.path as osp
import random
import pandas as pd
from .image_mcq import ImageMCQDataset
from ..smp import LMUDataRoot


class WikiVQABench(ImageMCQDataset):
    """WikiVQA benchmark - Image MCQ with question, correct answer, and 3 wrong answers

    Data format (parquet):
    - question: str
    - correct: str
    - wrongs: list of 3 strings
    - image: str (base64 encoded image)

    Converts to standard image MCQ format with randomized option positions.
    """

    RANDOM_SEED = None #42

    def __init__(self, dataset='WikiVQABench', **kwargs):
        # Set data_root before parent init calls load_data()
        self.data_root = LMUDataRoot()
        super().__init__(dataset=dataset, **kwargs)

    def load_data(self, dataset):
        # Load parquet file from data root
        data_path = osp.join(self.data_root, f'{dataset}.parquet')
        data = pd.read_parquet(data_path)

        # Seed random for reproducibility
        random.seed(self.RANDOM_SEED)

        result_rows = []
        for idx, row in data.iterrows():
            # Get correct and wrong answers
            correct_answer = row['correct']
            wrong_answers = row['wrongs']

            # Combine all options and shuffle
            all_options = [correct_answer] + list(wrong_answers)
            random.shuffle(all_options)

            # Find which position the correct answer is at (A=0, B=1, C=2, D=3)
            correct_pos = all_options.index(correct_answer)
            correct_letter = chr(ord('A') + correct_pos)

            # Build row with randomized options
            converted = {
                'index': idx,
                'question': row['question'],
                'A': all_options[0],
                'B': all_options[1],
                'C': all_options[2],
                'D': all_options[3],
                'answer': correct_letter,
                'image': row['image']  # Base64 encoded image
            }
            result_rows.append(converted)

        return pd.DataFrame(result_rows)

    @classmethod
    def supported_datasets(cls):
        return ['WikiVQABench']
