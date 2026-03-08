from .image_base import ImageBaseDataset
from .utils.judge_util import build_judge
from .utils.extractor import LLM_Extractor
from .utils.multiple_choice import report_acc
import os.path as osp
from vlmeval.smp import load, dump, get_intermediate_file_path, track_progress_rich


class MathKangaroo(ImageBaseDataset):

    TYPE = "VQA"
    DATASET_URL = {
        "MathKangaroo": "https://opencompass.openxlab.space/utils/VLMEval/MathKangaroo.tsv"
    }
    DATASET_MD5 = {"MathKangaroo": "d24236c9a0550695715f7acccb15a588"}
    DEFAULT_JUDGE = 'gpt-4o-mini'
    EXTRACT_PROMPT = 'Please extract an uppercase letter from the response and directly output it. '

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file).sort_values(by='index')
        judge = build_judge(**judge_kwargs)
        judge_file = get_intermediate_file_path(eval_file, '_judge', 'tsv')
        predictions = list(data['prediction'])
        # Define the verifier
        verifier = lambda x: str(x) in ['A', 'B', 'C', 'D', 'E']  # noqa: E731

        if not osp.exists(judge_file):
            extractor = LLM_Extractor(judge, self.EXTRACT_PROMPT, verifier=verifier)
            extracted = track_progress_rich(extractor.extract, predictions, nproc=16, desc='Extracting')
            extracted = [x if x is not None else 'Z' for x in extracted]
            data['extracted'] = extracted
            data['hit'] = [x == y for x, y in zip(data['answer'], data['extracted'])]
            dump(data, judge_file)

        df = load(judge_file)
        result = report_acc(df)
        result_file = get_intermediate_file_path(eval_file, '_acc')
        dump(result, result_file)
        return result
