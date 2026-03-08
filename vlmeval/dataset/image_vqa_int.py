from .image_vqa import ImageVQADataset
from .utils.judge_util import build_judge
from .utils.extractor import LLM_Extractor
from .utils.multiple_choice import report_acc
import os.path as osp
from vlmeval.smp import load, dump, get_intermediate_file_path, track_progress_rich


class VPCTDataset(ImageVQADataset):

    TYPE = "VQA"
    DATASET_URL = {
        "VPCT": "https://opencompass.openxlab.space/utils/VLMEval/VPCT.tsv"
    }
    DATASET_MD5 = {"VPCT": "491a3d102b642d47afea525197d3b181"}
    DEFAULT_JUDGE = 'gpt-4o-mini'
    EXTRACT_PROMPT = 'Please extract an integer from the response and directly output it. '
    JUDGE_FORMAT = '{model_name}_{dataset_name}_judge.tsv'

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)
        msgs = []
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
        ques = line['question']
        sp = line['system']
        question = f'{sp}\n\n{ques}'
        msgs.append(dict(type='text', value=question))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file).sort_values(by='index')
        judge = build_judge(**judge_kwargs)
        judge_file = get_intermediate_file_path(eval_file, '_judge', 'tsv')
        predictions = list(data['prediction'])
        # Pre-Extract
        ans_map = {'answer(1)': 1, 'answer(2)': 2, 'answer(3)': 3}
        predictions = [ans_map[x] if x in ans_map else x for x in predictions]
        # Define the verifier
        verifier = lambda x: str(x) in ['1', '2', '3']  # noqa: E731

        if not osp.exists(judge_file):
            extractor = LLM_Extractor(judge, self.EXTRACT_PROMPT, verifier=verifier)
            extracted = track_progress_rich(extractor.extract, predictions, nproc=16, desc='Extracting')
            extracted = [x if x is not None else -1 for x in extracted]
            data['extracted'] = extracted
            data['hit'] = [int(x) == int(y) for x, y in zip(data['answer'], data['extracted'])]
            dump(data, judge_file)

        data = load(judge_file)
        result = report_acc(data)
        result_file = get_intermediate_file_path(eval_file, '_acc')
        dump(result, result_file)
        return result
