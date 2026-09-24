from ..image_base import ImageBaseDataset
from .metrics import mdpbench_evaluator


class MDPBench(ImageBaseDataset):
    MODALITY = 'IMAGE'
    TYPE = 'QA'

    # Kept empty until dataset is published.
    DATASET_URL = {
        'MDPBench':
        'https://huggingface.co/datasets/Delores-Lin/MDPBench-VLMEvalKit/resolve/main/MDPBench_public.tsv'
    }
    DATASET_MD5 = {'MDPBench': '6aa9a03dcea532be3f92c81635d21883'}
    system_prompt = (
        'You are an advanced hybrid OCR engine capable of processing multilingual text mixed '
        'with mathematical notation. '
        'Your goal is to transcribe the content with high fidelity. Strict Rules:\n\n'
        '1. Multilingual Precision: Transcribe text exactly as it appears in the original language. '
        'Do not translate, summarize, or correct original spelling errors.\n\n'
        '2. Math Formatting: Identify all mathematical expressions and convert them into LaTeX.\n\n'
        '3. Inline Math: Use single dollar signs ($x$) for inline math (formulas within a sentence).\n\n'
        '4. Display Math: Use double dollar signs ($$x$$) for display math '
        '(standalone formulas on their own lines).\n\n'
        '5. Layout & Structure: Use Markdown to preserve the visual structure (headers, paragraphs, lists).\n\n'
        '6. Table Formatting: Use HTML tags (e.g., <table>, <tr>, <th>, <td>) '
        'to generate any tables found in the text.\n\n'
        '7. Output Only: Output the transcribed text directly without any conversational filler.'
    )

    def __init__(self, dataset='MDPBench', **kwargs):
        super().__init__(dataset, **kwargs)

    def build_prompt(self, line):
        image_path = self.dump_image(line)[0]
        msg = [
            dict(type='image', value=image_path),
            dict(type='text', value=self.system_prompt)
        ]
        return msg

    def evaluate(self, eval_file, **judge_kwargs):
        evaluator = mdpbench_evaluator(eval_file, self.data_path, **judge_kwargs)
        return evaluator.score()
