from vlmeval.dataset.utils import build_judge
from vlmeval.smp import *
from .image_base import ImageBaseDataset
from ..utils import track_progress_rich
from ..smp import load, dump, decode_base64_to_image
from .utils import DEBUG_MESSAGE

import zipfile


VQA_SYSTEM_PROMPT = json.dumps({
    'task': 'Answer the question presented to you truthfully.',
    'requirements': [
        'Analyze the image(s) first, then answer the question. If you are given a list of possible answers, you must choose from it.',
        'You must answer in the following json format: {"analysis": "(write your analysis here)", "answer": "(your answer)"}'
    ]
})

EVAL_SYSTEM_PROMPT = json.dumps({
    'task': 'Evaluate whether the answer to a question is correct.',
    'requirements': [
        'Compare an answer to a question with the ground truth answer. Determine whether it is correct.',
        'You must ignore any analysis of the problem if present. You must focus only on the final answer.',
        'You must answer in the following json format: {"verdict": "(1 for correct, 0 for incorrect)"}'
    ]
})

class MOAT(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MOAT': "https://huggingface.co/datasets/waltsun/MOAT/resolve/main/MOAT.tsv",
        'MOAT_images': "https://huggingface.co/datasets/waltsun/MOAT/resolve/main/MOAT_images.zip",
    }
    DATASET_MD5 = {
        'MOAT': '803b5a176a5b01aa1b8094fae73532a2',
        'MOAT_images': 'c0818a3e0a3f0bc7ee2be89ff04d73a6',
    }

    def post_build(self, dataset):
        assert dataset == "MOAT", f"Wrong dataset name {dataset}"
        ROOT = LMUDataRoot()
        self.img_root = osp.join(ROOT, 'MOAT_images')
        if not osp.exists(self.img_root):
            with zipfile.ZipFile(osp.join(ROOT, 'MOAT_images.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.img_root)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        question, choices, images, outside_knowledge_text, outside_knowledge_images = line['question'], line['choices'], line['images'], line['outside_knowledge_text'], line['outside_knowledge_images']
        choices, images, outside_knowledge_images = toliststr(choices), toliststr(images), toliststr(outside_knowledge_images)

        msgs = [
            {
                'type': 'text',
                'value': VQA_SYSTEM_PROMPT + '\n' + question + (f'\nThe choices are: {choices}' if choices else ''),
            },
        ]
        for img in images:
            msgs.append({'type': 'image', 'value': osp.join(self.img_root, img)})
        if not pd.isna(outside_knowledge_text):
            msgs.append({'type': 'text', 'value': outside_knowledge_text})
        for img in outside_knowledge_images:
            msgs.append({'type': 'image', 'value': osp.join(self.img_root, img)})
        return msgs
    
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        result_path = eval_file.replace(f'.{suffix}', f'_{judge_kwargs['model']}.xlsx')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(result_path):
            data = load(eval_file)
            model = build_judge(**judge_kwargs)
            assert model.working(), ('MOAT evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

            def verdict_one(model, line):
                prompt = EVAL_SYSTEM_PROMPT + '\n' + f'The answer to evaluate is {line['prediction']}\nThe ground truth answer is {line['answer']}'
                res = model.generate(prompt)
                return int(json.loads(res)['verdict'])
            verdict_list = track_progress_rich(
                lambda line: verdict_one(model, line),
                [data.iloc[i] for i in range(len(data))],
                nproc=nproc,
                chunksize=nproc,
            )
            data['verdict'] = verdict_list
            dump(data, result_path)
        
        data = load(result_path)
        overall_acc = data['verdict'].mean()
        capability_set = set()
        for i in range(len(data)):
            capability_set.update(toliststr(data.iloc[i]['capability']))
        capability_score_map = {capability: (0, 0) for capability in capability_set}
        for i in range(len(data)):
            line = data.iloc[i]
            capabilities = toliststr(line['capability'])
            verdict = line['verdict']
            for capability in capabilities:
                capability_score_map[capability] = (capability_score_map[capability][0] + verdict, capability_score_map[capability][1] + 1)
        capability_score_map = {capability: score[0] / score[1] for capability, score in capability_score_map.items()}
        return {
            'overall_acc': overall_acc,
            'result_path': result_path,
            'capability_acc': capability_score_map,
        }
