from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.multiple_choice import report_acc
from ..smp import *


PW_TMPL = (
    "You are an AI assistant who will help me to tell if a model prediction "
    "matches with the groundtruth answer of a question. "
    'The decision are only Yes / No. \n'
    "If the model prediction include an exact match of the groundtruth answer "
    "(except non-alphanumeric characters), output Yes. "
    'Otherwise, output No. \n'
    'Model Prediction: {}; Groundtruth Answer: {}.\n'
    'Please make your decision based on the above information. '
)


def YOrN_Extraction(output):
    s = output.lower()
    words = process_punctuation(s).split()
    if 'yes' in words and 'no' not in words:
        return 1
    if 'yes' not in words and 'no' in words:
        return 0
    return 'Unknown'


def PuzzleWorld_auxeval(model, line):
    def fetch_answer(prediction):
        lines = prediction.split('\n')
        answer_cands = []
        for line in lines:
            if line.upper().startswith('ANSWER:'):
                answer_cands.append(line.upper().replace('ANSWER:', '').strip())
        if len(answer_cands) > 1:
            warnings.warn("Multiple answer candidates found, only return the last one. ")
            return answer_cands[-1]
        elif len(answer_cands) == 1:
            return answer_cands[0]
        elif len(answer_cands) == 0:
            warnings.warn("No answer candidate found, will return the original prediction. ")
            return prediction.upper()

    answer = fetch_answer(line['prediction'])
    # Extraction Succeed, but the answer is toooo long.
    if len(answer) != len(line['prediction']) and len(answer) >= len(line['answer']) * 4:
        return dict(hit=0, log=(
            f"Extracted {answer}, which is way much longer (4x) "
            "than the groundtruth {line['answer']}. "
        ))
    prompt = PW_TMPL.format(answer, line['answer'])
    retry = 3
    for i in range(retry):
        output = model.generate(prompt, temperature=0.2 * i)
        ans = YOrN_Extraction(output)
        if ans in [0, 1]:
            return dict(hit=ans, log=f"Extracted: {answer}; GPT Response: {output}")
    return dict(hit=0, log=f"Extracted {answer}; GPT Judge Failed. ")


class PuzzleWorldDataset(ImageBaseDataset):

    TYPE = 'VQA'

    DATASET_URL = {
        'PuzzleWorld': 'https://opencompass.openxlab.space/utils/VLMEval/PuzzleWorld.tsv',
        'PuzzleWorld_ImgOnly': 'https://opencompass.openxlab.space/utils/VLMEval/PuzzleWorld.tsv',
    }

    DATASET_MD5 = {
        'PuzzleWorld': '8a1254217d018c6d074c6d954e07e9e1',
        'PuzzleWorld_ImgOnly': '8a1254217d018c6d074c6d954e07e9e1',
    }

    # include 30%, 50%, 70% of the cot in the prompt, 3 testing configuration
    # Will firstly support 50%
    for ratio in [30, 50, 70]:
        for k in ['PuzzleWorld', 'PuzzleWorld_ImgOnly']:
            DATASET_URL[f'{k}_{ratio}'] = DATASET_URL[k]
            DATASET_MD5[f'{k}_{ratio}'] = DATASET_MD5[k]

    def build_prompt(self, line):
        # Image Goes First, Dialog Goes Second
        cot = json.loads(line['cot'])
        # Last Step in CoT always include the final answer
        cot = cot[:-1]

        inst = (
            "Please solve the puzzle above and provide the **final answer**. \n"
            "The final answer should be in a **SINGLE SEPARATE LINE** at the end of your response, "
            "starting with `Answer: `. \n"
        )
        if 'ImgOnly' in self.dataset_name:
            question = inst
        else:
            question = str(line['question']) + '\n' + inst

        cot_suffix = []
        if '0' in self.dataset_name:
            assert '50' in self.dataset_name, 'Reasoning w. 30%, 70% partial cot has not been implemented yet. '
            n_cot = len(cot) // 2
            if n_cot:
                cot_prompt = 'Here are the first few steps to solve the puzzle: \n'
                images = []
                for i in range(n_cot):
                    step = cot[i]
                    assert 'text' in step, step
                    cot_prompt += f"Step {i + 1}: {step['text']}\n"
                    if 'image' in step:
                        cot_prompt += '<image>'
                        images.append(self.dump_image_atomic(step['image'], f"{line['index']}_cot_step_{i}.png"))
                if len(images):
                    cot_splits = cot_prompt.split('<image>')
                    assert len(cot_splits) == len(images) + 1, (len(cot_splits), len(images))
                    question += cot_splits[0]
                    for i in range(len(images)):
                        cot_suffix.append(dict(type='image', value=images[i]))
                        if cot_splits[i + 1] != '':
                            cot_suffix.append(dict(type='text', value=cot_splits[i + 1]))
                else:
                    question += cot_prompt

        images = self.dump_image(line)
        prompt = []
        for im in images:
            prompt.append(dict(type='image', value=im))
        prompt.append(dict(type='text', value=question))
        if len(cot_suffix):
            prompt.extend(cot_suffix)
        return prompt

    # It returns a DataFrame
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        _ = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        storage = get_intermediate_file_path(eval_file, '_judge')
        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        nproc = judge_kwargs.pop('nproc', 16)

        if not osp.exists(storage):
            ans_map = {} if not osp.exists(tmp_file) else load(tmp_file)

            model = judge_kwargs.pop('model', 'gpt-4o-1120')
            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(model=model, **judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None

            if model is not None:
                lines = [data.iloc[i] for i in range(len(data))]
                indices = [x['index'] for x in lines if x['index'] not in ans_map]
                lines = [x for x in lines if x['index'] not in ans_map]
                tups = [(model, line) for line in lines]

                if len(lines):
                    res = track_progress_rich(
                        PuzzleWorld_auxeval, tups, nproc=nproc, chunksize=nproc, keys=indices, save=tmp_file)
                    for k, v in zip(indices, res):
                        ans_map[k] = v

            judge_results = [ans_map[x] for x in data['index']]
            data['hit'] = [x['hit'] for x in judge_results]
            data['log'] = [x['log'] for x in judge_results]
            dump(data, storage)

        data = load(storage)
        acc = report_acc(data)

        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)
        return acc
