import os
import re
import tempfile
from functools import partial

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import track_progress_rich


class ImageVQADataset(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'OCRVQA_TEST':
        'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TEST.tsv',
        'OCRVQA_TESTCORE':
        'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TESTCORE.tsv',
        'TextVQA_VAL':
        'https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv',
        'DocVQA_VAL':
        'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv',
        'DocVQA_TEST':
        'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_TEST.tsv',
        'InfoVQA_VAL':
        'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv',
        'InfoVQA_TEST':
        'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_TEST.tsv',
        'ChartQA_TEST':
        'https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv',
        'GQA_TestDev_Balanced':
        'https://opencompass.openxlab.space/utils/VLMEval/GQA_TestDev_Balanced.tsv',
    }

    DATASET_MD5 = {
        'OCRVQA_TEST': 'ca46a6d74b403e9d6c0b670f6fc00db9',
        'OCRVQA_TESTCORE': 'c5239fe77db8bdc1f2ad8e55e0d1fe97',
        'TextVQA_VAL': 'b233b31f551bbf4056f2f955da3a92cd',
        'DocVQA_VAL': 'd5ee77e1926ff10690d469c56b73eabf',
        'DocVQA_TEST': '6a2f28cac26ef2d3447374e8c6f6c8e9',
        'InfoVQA_VAL': '2342e9c225222f0ef4dec545ebb126fe',
        'InfoVQA_TEST': 'df535bf51b88dc9718252c34131a6227',
        'ChartQA_TEST': 'c902e0aa9be5582a7aad6dcf52734b42',
        'GQA_TestDev_Balanced': '99b62f22e224d9b2f32dcbe41359d1c9',
    }

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        msgs[-1][
            'value'] += '\nAnswer the question using a single word or phrase.'
        return msgs

    # It returns a DataFrame
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.vqa_eval import hit_calculate, process_line

        data = load(eval_file)
        dataset = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        lt = len(data)
        pool = mp.Pool(16)
        lines = [data.iloc[i] for i in range(lt)]
        if listinstr(['TextVQA'], dataset):
            res = pool.map(partial(process_line, method='vqa_score'), lines)
        elif listinstr(['ChartQA'], dataset):
            res = pool.map(partial(process_line, method='relaxed_accuracy'),
                           lines)
        elif listinstr(['OCRVQA', 'GQA'], dataset):
            res = pool.map(partial(process_line, method='accuracy'), lines)
        elif listinstr(['DocVQA', 'InfoVQA'], dataset):
            res = pool.map(partial(process_line, method='anls'), lines)
        else:  # default using vqa_score to calculate score
            res = pool.map(process_line, lines)
        hit = hit_calculate(res, dataset)
        ret = dict()
        if 'split' in data:
            splits = set(data['split'])
            for sp in splits:
                sub = [r for l, r in zip(lines, res) if l['split'] == sp]
                # [np.mean(x['match']) >= full_score_weight for x in sub]
                hit = hit_calculate(sub, dataset)
                ret[sp] = np.mean(hit) * 100
            sub = [r for l, r in zip(lines, res)]
            hit = hit_calculate(sub, dataset)
            ret['Overall'] = np.mean(hit) * 100
        else:
            ret['Overall'] = np.mean(hit) * 100
            if 'category' in data:
                cates = list(set(data['category']))
                cates.sort()
                for c in cates:
                    sub = [r for l, r in zip(lines, res) if l['category'] == c]
                    # [np.mean(x['match']) >= full_score_weight for x in sub]
                    hit = hit_calculate(sub, dataset)
                    ret[c] = np.mean(hit) * 100
        ret = d2df(ret)
        ret.round(2)

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)
        return ret


class VizWiz(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'VizWiz': 'https://opencompass.openxlab.space/utils/VLMEval/VizWiz.tsv'
    }
    DATASET_MD5 = {'VizWiz': 'fa4ac4164467563ed2fac6eac6631bd0'}

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.vqa_eval import hit_calculate, process_line

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')

        if not osp.exists(result_file):
            data = load(eval_file)
            assert 'answers' in data and 'prediction' in data
            data['prediction'] = [str(x) for x in data['prediction']]
            data['answer'] = [str(x) for x in data['answers']]

            lt = len(data)
            pool = mp.Pool(16)
            lines = [data.iloc[i] for i in range(lt)]
            res = pool.map(process_line, lines)

            hit = hit_calculate(res, 'VizWiz')
            ret = dict()

            ret['Overall'] = np.mean(hit) * 100
            ret = d2df(ret)
            ret.round(2)

            dump(ret, result_file)

        retz = pd.read_csv(result_file)
        return retz


class OCRBench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'OCRBench':
        'https://opencompass.openxlab.space/utils/VLMEval/OCRBench.tsv',
        # For internal test only
        'OCRBench_MINI':
        'https://opencompass.openxlab.space/utils/TEST/OCRBench_MINI.tsv'
    }
    DATASET_MD5 = {'OCRBench': 'e953d98a987cc6e26ef717b61260b778'}

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        OCRBench_score = {
            'Regular Text Recognition': 0,
            'Irregular Text Recognition': 0,
            'Artistic Text Recognition': 0,
            'Handwriting Recognition': 0,
            'Digit String Recognition': 0,
            'Non-Semantic Text Recognition': 0,
            'Scene Text-centric VQA': 0,
            'Doc-oriented VQA': 0,
            'Key Information Extraction': 0,
            'Handwritten Mathematical Expression Recognition': 0,
        }

        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            predict = str(line['prediction'])
            answers = eval(line['answer'])
            category = line['category']
            if category == 'Handwritten Mathematical Expression Recognition':
                for j in range(len(answers)):
                    answer = answers[j].strip().replace('\n',
                                                        ' ').replace(' ', '')
                    predict = predict.strip().replace('\n',
                                                      ' ').replace(' ', '')
                    if answer in predict:
                        OCRBench_score[category] += 1
                        break
            else:
                for j in range(len(answers)):
                    answer = answers[j].lower().strip().replace('\n', ' ')
                    predict = predict.lower().strip().replace('\n', ' ')
                    if answer in predict:
                        OCRBench_score[category] += 1
                        break

        final_score_dict = {}
        final_score_dict['Text Recognition'] = \
            (OCRBench_score['Regular Text Recognition'] + OCRBench_score['Irregular Text Recognition']
             + OCRBench_score['Artistic Text Recognition'] + OCRBench_score['Handwriting Recognition']
             + OCRBench_score['Digit String Recognition'] + OCRBench_score['Non-Semantic Text Recognition'])
        final_score_dict['Scene Text-centric VQA'] = OCRBench_score[
            'Scene Text-centric VQA']
        final_score_dict['Doc-oriented VQA'] = OCRBench_score[
            'Doc-oriented VQA']
        final_score_dict['Key Information Extraction'] = OCRBench_score[
            'Key Information Extraction']
        final_score_dict['Handwritten Mathematical Expression Recognition'] = \
            (OCRBench_score['Handwritten Mathematical Expression Recognition'])
        final_score_dict['Final Score'] = \
            (final_score_dict['Text Recognition'] + final_score_dict['Scene Text-centric VQA']
             + final_score_dict['Doc-oriented VQA'] + final_score_dict['Key Information Extraction']
             + final_score_dict['Handwritten Mathematical Expression Recognition'])
        final_score_dict['Final Score Norm'] = (
            float(final_score_dict['Final Score']) / 10)
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict


class MathVista(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MathVista_MINI':
        'https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv'
    }
    DATASET_MD5 = {'MathVista_MINI': 'f199b98e178e5a2a20e7048f5dcb0464'}

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mathvista import MathVista_auxeval, MathVista_acc

        model = judge_kwargs['model']
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), 'MathVista evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MathVista_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v[
                        'res']

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score = MathVista_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        dump(score, score_pth)
        return score


class MathVerse(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MathVerse_MINI':
        'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIV.tsv',  # noqa
        'MathVerse_MINI_Vision_Only':
        'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIVOnly.tsv',  # noqa
        'MathVerse_MINI_Vision_Only_cot':
        'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIVOnly.tsv',  # noqa
        'MathVerse_MINI_Vision_Dominant':
        'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIVDom.tsv',  # noqa
        'MathVerse_MINI_Vision_Intensive':
        'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINIVInt.tsv',  # noqa
        'MathVerse_MINI_Text_Lite':
        'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINITLite.tsv',  # noqa
        'MathVerse_MINI_Text_Dominant':
        'http://opencompass.openxlab.space/utils/benchmarks/MathVerse/MathVerse_MINITDom.tsv',  # noqa
    }
    DATASET_MD5 = {
        'MathVerse_MINI': '5017caca32b7fa110c350a1bea861b65',
        'MathVerse_MINI_Vision_Only': '68a11d4680014ac881fa37adeadea3a4',
        'MathVerse_MINI_Vision_Only_cot': '68a11d4680014ac881fa37adeadea3a4',
        'MathVerse_MINI_Vision_Dominant': 'b8fb63852d261ab2aaefba29cc2414d3',
        'MathVerse_MINI_Vision_Intensive': '01cbd35be202bb0c4873a4186a63bc19',
        'MathVerse_MINI_Text_Lite': '19e4b13bdd30b89a03b2e358bcfefa04',
        'MathVerse_MINI_Text_Dominant': '4f5cd2fa6630ea00bb11d6fde1f6fe6a',
    }

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)
        if 'cot' in self.dataset_name:
            question = line['query_cot']
        else:
            question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mathverse import MathVerse_auxeval_extract, MathVerse_auxeval_score, MathVerse_acc

        model = judge_kwargs['model']
        suffix = eval_file.split('.')[-1]
        storage_extract = eval_file.replace(f'.{suffix}', f'_{model}_extract.xlsx')
        tmp_file_extract = eval_file.replace(f'.{suffix}', f'_{model}_extract.pkl')
        storage_score = eval_file.replace(f'.{suffix}', f'_{model}_score.xlsx')
        tmp_file_score = eval_file.replace(f'.{suffix}', f'_{model}_score.pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        # stage1: extract the answer
        if not osp.exists(storage_extract):
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), 'MathVerse evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file_extract):
                ans = load(tmp_file_extract)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MathVerse_auxeval_extract,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file_extract,
                )
                ans = load(tmp_file_extract)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log_extract'] == v['log_extract'] and ans[
                        k]['extract'] == v['extract']

            data['extract'] = [ans[idx]['extract'] for idx in data['index']]
            data['log_extract'] = [
                ans[idx]['log_extract'] for idx in data['index']
            ]
            dump(data, storage_extract)

        # stage2: score the answer
        if not osp.exists(storage_score):
            data = load(storage_extract)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), 'MathVerse evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file_score):
                ans = load(tmp_file_score)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MathVerse_auxeval_score,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file_score,
                )
                ans = load(tmp_file_score)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log_score'] == v['log_score'] and ans[k][
                        'score'] == v['score']

            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log_score'] = [
                ans[idx]['log_score'] for idx in data['index']
            ]
            dump(data, storage_score)

        score = MathVerse_acc(storage_score)
        score_pth = storage_score.replace('.xlsx', '.csv')
        dump(score, score_pth)
        return score


class MathVision(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MathVision':
        'https://opencompass.openxlab.space/utils/VLMEval/MathVision.tsv',
        'MathVision_MINI':
        'https://opencompass.openxlab.space/utils/VLMEval/MathVision_MINI.tsv'
    }
    DATASET_MD5 = {
        'MathVision': '93f6de14f7916e598aa1b7165589831e',
        'MathVision_MINI': '060fe4fa5d868987ce179307bd5f8a33'
    }

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mathv import MATH_V_auxeval, MATH_V_acc

        if 'model' in judge_kwargs:
            model = judge_kwargs['model']
        else:
            model = os.path.basename(os.environ.get('LOCAL_LLM'))
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), 'MATH-Vision evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MATH_V_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v[
                        'res']

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score = MATH_V_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        dump(score, score_pth)
        return score


class Physics_yale(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'atomic_dataset':
        'http://opencompass.openxlab.space/utils/benchmarks/physics/atomic_dataset.tsv',
        'electro_dataset':
        'http://opencompass.openxlab.space/utils/benchmarks/physics/electro_dataset.tsv',
        'mechanics_dataset':
        'http://opencompass.openxlab.space/utils/benchmarks/physics/mechanics_dataset.tsv',
        'optics_dataset':
        'http://opencompass.openxlab.space/utils/benchmarks/physics/optics_dataset.tsv',
        'quantum_dataset':
        'http://opencompass.openxlab.space/utils/benchmarks/physics/quantum_dataset.tsv',
        'statistics_dataset':
        'http://opencompass.openxlab.space/utils/benchmarks/physics/statistics_dataset.tsv',
        'Physics': 'http://opencompass.openxlab.space/utils/benchmarks/physics/Physics.tsv'
    }
    DATASET_MD5 = {
        'atomic_dataset': 'b927fae6bcc6163b0bd89041e4421c70',
        'electro_dataset': '66db62cdbc468bb003e6d09592b94b59',
        'mechanics_dataset': '11f287a18ccc6227bea15fa89f24de67',
        'optics_dataset': '39ab9028ae4a33c06f78ce8618668172',
        'quantum_dataset': 'd2610f9938ad1e848259ccbcd5ac3acf',
        'statistics_dataset': '78242aa2431a477782b5b3de1c18d633',
        'Physics': 'b4136f27f09339698f636111c07824e9'
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image'])
        else:
            tgt_path = self.dump_image(line)

        instruction = (
            "You are a physics expert assistant. Solve the following question step-by-step.\n\n"
            "At the VERY END of your answer, output ONLY the FINAL ANSWER in this format:\n\n"
            "\\[\n\\boxed{your_final_answer_here}\n\\]\n\n"
            " You MUST put the final answer in the \\boxed{} environment.\n"
            " This applies even if the answer is a text explanation like \"The singlet state is lower in energy.\"\n"
            "Do NOT include multiple boxes.\n"
            "Do NOT include \\boxed anywhere else in your reasoning.\n"
            " The box must appear on the last line of the response.\n\n"
            "WARNING: DO NOT forget to include \\boxed{} with the final answer. Responses without it will be considered INVALID.\n\n"  # noqa: E501
            "Example:\n"
            "Question: What is the energy difference between n=2 and n=1 in hydrogen?\n"
            "Answer: The energy levels are E_n = -13.6 / n² (in eV).\n"
            "E_2 = -13.6 / 4 = -3.4 eV\n"
            "E_1 = -13.6 eV\n"
            "ΔE = 13.6 - 3.4 = 10.2 eV\n"
            "\\[\n\\boxed{10.2\\ \\text{eV}}\n\\]\n\n"
            f"Question: {line['question']}\nAnswer:")

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([{"type": "image", "value": p} for p in tgt_path])
        else:
            msgs.append({"type": "image", "value": tgt_path})

        msgs.append({"type": "text", "value": instruction})

        return msgs

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.physic import PHYSIC_acc, PHYSIC_auxeval

        if 'LOCAL_LLM' in os.environ:
            model = os.path.basename(os.environ.get('LOCAL_LLM'))
            print(f'Using local model as judge model for PHYSICS: {model}')
        else:
            model = judge_kwargs.setdefault('model', 'gpt-4o-mini')
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            judge_kwargs['max_tokens'] = 4096
            model = build_judge(**judge_kwargs)
            assert model.working(), 'Physics_yale evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    PHYSIC_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v[
                        'res']

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score = PHYSIC_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        dump(score, score_pth)
        return score


class OlympiadBench(ImageBaseDataset):
    TYPE = 'VQA_ex_prompt'
    DATASET_URL = {
        'OlympiadBench':
        'https://opencompass.openxlab.space/utils/VLMEval/OlympiadBench.tsv',
        'OlympiadBench_EN':
        'https://opencompass.openxlab.space/utils/VLMEval/OlympiadBench_EN.tsv',
        'OlympiadBench_CN':
        'https://opencompass.openxlab.space/utils/VLMEval/OlympiadBench_CN.tsv'
    }
    DATASET_MD5 = {
        'OlympiadBench': '9735ae0f0299eae1e7d07f5a7feab914',
        'OlympiadBench_EN': '5c68e100d394351fc7049f29d4d4efed',
        'OlympiadBench_CN': 'ea01b16788955702c79650c701e5b623'
    }

    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        tgt_path_z = []
        if isinstance(line['image'], list):
            for i in range(len(line['image'])):
                tgt_path = osp.join(self.img_root,
                                    f"{line['index']}--{i + 1}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'][i], tgt_path)
                tgt_path_z.append(tgt_path)
        else:
            tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
            if not read_ok(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path_z.append(tgt_path)
        return tgt_path_z

    def build_prompt(self, line):

        from .utils.olympiadbench import get_answer_type_text, make_input

        self.is_chinese = 'zh' in line['source']
        self.is_math = 'maths' in line['source']
        self.is_theorem_proving = 'TP' in line['source']

        if self.is_chinese:
            subject_content = '数学' if self.is_math else '物理'
            if self.is_theorem_proving:
                prompt = (
                    f"以下是中国{subject_content}竞赛中的证明题。请根据题目的要求，运用逻辑推理及常用定理证明题目中的命题。"
                    "证明过程中使用的变量和公式请使用LaTeX格式表示。")
            else:
                answer_type_text = get_answer_type_text(
                    line['answer_type'],
                    is_chinese=True,
                    multiple_answer=line['is_multiple_answer'])
                if line['is_multiple_answer']:
                    multiple_answer_text = '\\boxed{用英文逗号连接的多个答案}'
                else:
                    multiple_answer_text = '\\boxed{答案}'
                unit_text = ''
                if line['unit']:
                    multiple_answer_text += '(单位)'
                    unit_text = '，注意答案的单位不要放在\\boxed{}中'
                prompt = (
                    f'以下是中国{subject_content}竞赛中的解答题{answer_type_text}。请根据题目的要求和所提供的信息计算得出答案。'
                    f'解答过程和结果中使用的变量和公式请使用LaTeX格式表示。请在最后以“所以最终答案是{multiple_answer_text}。”'
                    f'显式给出结果{unit_text}。')
        else:
            subject_content = 'Math' if self.is_math else 'Physics'
            if self.is_theorem_proving:
                prompt = (
                    f'The following is a theorem proving problem from an International {subject_content} competition. '
                    'Please use logical reasoning and common theorems to prove the proposition in the problem '
                    'according to the given requirements. '
                    'Please use LaTeX format to represent the variables and formulas used in the proof.'
                )
            else:
                if line['is_multiple_answer']:
                    multiple_answer_text = '\\boxed{multiple answers connected with commas}'
                else:
                    multiple_answer_text = '\\boxed{answer}'
                unit_text = ''
                if line['unit']:
                    multiple_answer_text += '(unit)'
                    unit_text = ', note that the unit of the answer should not be included in \\boxed{}'
                answer_type_text = get_answer_type_text(
                    line['answer_type'],
                    is_chinese=False,
                    multiple_answer=line['is_multiple_answer'])
                prompt = (
                    f'The following is an open-ended problem from an International {subject_content} competition. '
                    f'{answer_type_text}Please calculate the answer according to the given requirements and '
                    'the information provided. Please use LaTeX format to represent the variables and formulas '
                    'used in the solution process and results. Please end your solution with "So the final answer '
                    f'is {multiple_answer_text}." and give the result explicitly{unit_text}.'
                )

        if self.is_math:
            input = make_input(prompt, line['question'])
        else:
            if 'context' in line.keys() and str(
                    line['context']) != 'nan':  # cannot be null
                input = make_input(prompt,
                                   line['context'] + '\n' + line['question'])
            else:
                input = make_input(prompt, line['question'])

        ret = [dict(type='text', value=input)]
        tgt_path = self.dump_image(line)

        ret.extend([dict(type='image', value=s) for s in tgt_path])

        return ret

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.olympiadbench import MathJudger, extract_answer
        judger = MathJudger()

        suffix = eval_file.split('.')[-1]
        name_str1 = 'judge'
        name_str2 = 'score'
        result_file = eval_file.replace(f'.{suffix}',
                                        f'_{name_str1}_result.xlsx')
        score_file = eval_file.replace(f'.{suffix}',
                                       f'_{name_str2}_result.csv')

        if not osp.exists(result_file):
            data = load(eval_file)
            scorez = []

            for i in tqdm(data.iterrows()):
                line = i[1]
                model_answer = line['prediction']
                is_chinese = 'zh' in line['source']
                model_answer = extract_answer(is_chinese,
                                              model_answer,
                                              is_deepseek=False)
                answer_type = line['answer_type']

                final_answer = line['final_answer'][2:-2]

                if str(answer_type) != 'nan' and 'Tuple' in answer_type:
                    judge_result = judger.judge(model_answer, final_answer)
                else:
                    if str(line['error']) != 'nan':
                        if ',' in line['error']:
                            precisions = line['error'].split(',')
                            precisions = [
                                float(p) if p else 1e-8 for p in precisions
                            ]
                            judge_result = judger.judge(
                                model_answer, final_answer, precisions)
                        else:
                            precision = float(line['error'])
                            judge_result = judger.judge(
                                model_answer, final_answer, precision)
                    else:
                        judge_result = judger.judge(model_answer, final_answer)
                scorez.append(judge_result)

            data['score'] = scorez
            dump(data, result_file)

        judge_file = load(result_file)

        if not osp.exists(score_file):
            name_list = [
                'OE_MM_maths_en_COMP', 'OE_MM_maths_zh_CEE',
                'OE_MM_maths_zh_COMP', 'OE_MM_physics_en_COMP',
                'OE_MM_physics_zh_CEE', 'OE_TO_maths_en_COMP',
                'OE_TO_maths_zh_CEE', 'OE_TO_maths_zh_COMP',
                'OE_TO_physics_en_COMP', 'OE_TO_physics_zh_CEE'
            ]

            sample_list = [[] for _ in range(len(name_list))]
            for i in judge_file.iterrows():
                line = i[1]
                for j in range(len(name_list)):
                    if line['source'] == name_list[j]:
                        sample_list[j].append(line['score'])

            acc_dict = {}
            correct_list = []

            # fine-grained
            for i in range(len(name_list)):
                correct_num = 0
                for j in sample_list[i]:
                    if j:
                        correct_num += 1
                correct_list.append(correct_num)
                acc = 100 * correct_num / len(sample_list[i])
                acc_dict[name_list[i]] = [acc]

            # 4 grained
            labela = ['zh', 'en']
            labelb = ['maths', 'physics']

            grain_list = [[x, y] for x in labela for y in labelb]
            for j in grain_list:
                dict_name = j[0] + "_" + j[1]
                correct_num = 0
                full_num = 0
                for i in range(len(name_list)):
                    if all(k in name_list[i] for k in j):
                        correct_num += correct_list[i]
                        full_num += len(sample_list[i])
                acc = 100 * correct_num / full_num
                acc_dict[dict_name] = [acc]

            # 2 grained
            grain_list = ['maths', 'physics']
            for j in grain_list:
                dict_name = j
                correct_num = 0
                full_num = 0
                for i in range(len(name_list)):
                    if j in name_list[i]:
                        correct_num += correct_list[i]
                        full_num += len(sample_list[i])
                acc = 100 * correct_num / full_num
                acc_dict[dict_name] = [acc]

            # AVG
            correct_num = sum(correct_list)
            acc = 100 * correct_num / len(judge_file)
            acc_dict['AVG'] = [acc]

            acc_pd = pd.DataFrame(acc_dict)
            acc_pd.to_csv(score_file, index=False, encoding='gbk')

        accdz = pd.read_csv(score_file)
        return accdz


class LogicVista(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'LogicVista':
        'https://opencompass.openxlab.space/utils/VLMEval/LogicVista.tsv'
    }
    DATASET_MD5 = {'LogicVista': '41c5d33adf33765c399e0e6ae588c061'}

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.logicvista import LogicVista_auxeval, evaluate_logicvista

        # model = judge_kwargs['model']
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in [
            'exact_matching', 'gpt-4-0125', 'gpt-4-turbo', 'gpt-4o-mini'
        ], model
        name_str_map = {
            'gpt-4-0125': 'gpt4',
            'gpt-4-turbo': 'gpt4-turbo',
            'gpt-4o-mini': 'gpt4o-mini'
        }
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn(
                    'OPENAI API is not working properly, will use exact matching for evaluation'
                )
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn(
                'OPENAI_API_KEY is not set properly, will use exact matching for evaluation'
            )
            model = None

        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{name_str}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{name_str}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage) and model is not None:
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
            assert model.working(), 'LogicVista evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    LogicVista_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v[
                        'res'] and ans[k]['hit'] == v['hit']

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            data['hit'] = [ans[idx]['hit'] for idx in data['index']]

            dump(data, storage)
        if osp.exists(storage):
            accuracy_scores = evaluate_logicvista(storage)
            score_pth = storage.replace('.xlsx', '_score.csv')
            dump(accuracy_scores, score_pth)

            return accuracy_scores


class MME_CoT(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MME_CoT_TEST':
        'https://huggingface.co/datasets/CaraJ/MME-CoT_VLMEvalKit/resolve/main/MME-CoT.tsv'  # noqa
    }
    DATASET_MD5 = {
        'MME_CoT_TEST': 'a612dee0f2d702e01fe50267201302e0',
    }

    def split_MME_CoT(self, msgs):
        text, images = None, []

        # Separate images and text from msgs
        for s in msgs:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                assert text is None  # Ensure only one text entry is expected
                text = s['value']

        # Split text by <image> tags
        text_segs = text.split('<image>')

        # Initialize the segments list
        segs = []
        # Iterate through the text segments and images
        for i, seg in enumerate(text_segs):
            # Append the image if this is not the first segment and there are still images left
            if i > 0 and i - 1 < len(images):
                segs.append(dict(type='image', value=images[i - 1]))
            # Append the text segment (if it's non-empty)
            if len(seg.strip()) > 0:
                segs.append(dict(type='text', value=seg))

        return segs

    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                if 'image_path' in line:
                    image_path_list = line['image_path']
                else:
                    image_path_list = [
                        f"{line['index']}--{i + 1}.jpg"
                        for i in range(len(line['image']))
                    ]
                for img, im_name in zip(line['image'], image_path_list):
                    path = osp.join(self.img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])

        return tgt_path

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)

        prompt = line['question']

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        prompt = prompt + '\n' + '\n'.join(
            [f'{key}. {item}' for key, item in options.items()])

        # add cot prompt
        if os.environ.get('USE_COT_PROMPT', '1') == '1':
            prompt += "\nPlease generate a step by step answer, include all your intermediate reasoning process, and provide the final answer at the end."  # noqa: E501
        else:
            prompt += "\nPlease directly provide the final answer without any other output."

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        msgs = self.split_MME_CoT(msgs)
        return msgs

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        print(
            "\033[1;31;40m"
            "[MME-CoT Evaluation]: Please refer to the official repository for evaluation: https://github.com/CaraJ7/MME-CoT/tree/main"  # noqa: E501
            "\033[0m")
        dummy_result = dict(dummy_result=0)
        return pd.DataFrame(dummy_result, index=[0])


class LLaVABench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'LLaVABench':
        'https://opencompass.openxlab.space/utils/VLMEval/LLaVABench.tsv'
    }
    DATASET_MD5 = {'LLaVABench': 'd382a093f749a697820d3dadd61c8428'}

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.llavabench import (
            build_prompt,
            LLaVABench_atomeval,
            LLaVABench_score,
        )

        suffix = '.' + eval_file.split('.')[-1]
        record_file = eval_file.replace(suffix, '_openai_result' + suffix)
        score_file = eval_file.replace(suffix, '_score.csv')
        nproc = judge_kwargs.pop('nproc', 4)
        system_prompt = 'You are a helpful and precise assistant for checking the quality of the answer.'

        if not osp.exists(record_file):
            data = load(eval_file)
            lines = [data.iloc[i] for i in range(len(data))]
            model = build_judge(temperature=0.2,
                                system_prompt=system_prompt,
                                **judge_kwargs)
            assert model.working(), 'LLaVABench evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE

            prompts = [build_prompt(line) for line in lines]
            tups = [(model, prompt) for prompt in prompts]
            scores = track_progress_rich(LLaVABench_atomeval,
                                         tups,
                                         nproc=nproc,
                                         chunksize=nproc)
            data['gpt4_score'] = [x[0] for x in scores]
            data['score'] = [x[1] for x in scores]
            dump(data, record_file)

        data = load(record_file)
        ret = LLaVABench_score(data).round(1)
        dump(ret, score_file)
        return ret


class VGRPBench(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'VGRPBench':
        'https://huggingface.co/datasets/VGRP-Bench/VGRP-Bench/resolve/main/data/vgrpbench_dataset_easy_100_samples.tsv'
    }  # noqa: E501

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        print("VGRPBench evaluation")

        from .utils.vgrpbench.evaluation import (
            build_prompt,
            VGRPBench_atomeval,
            VGRPBench_score,
            VGRPBench_get_system_prompt,
        )

        suffix = '.' + eval_file.split('.')[-1]
        record_file = eval_file.replace(suffix, '_openai_result' + suffix)
        score_file = eval_file.replace(suffix, '_score.csv')

        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(record_file):
            data = load(eval_file)
            lines = [data.iloc[i] for i in range(len(data))]

            system_prompts = [
                VGRPBench_get_system_prompt(line) for line in lines
            ]

            models = [
                build_judge(temperature=0.0,
                            system_prompt=system_prompt,
                            **judge_kwargs) for system_prompt in system_prompts
            ]

            prompts = [build_prompt(line) for line in lines]

            tups = [(model, prompt, line)
                    for model, prompt, line in zip(models, prompts, lines)]

            # Original parallel processing
            scores = track_progress_rich(VGRPBench_atomeval,
                                         tups,
                                         nproc=nproc,
                                         chunksize=nproc)

            data['perception_correct'] = [
                x['perception_correct'] for x in scores
            ]
            data['answer_correct'] = [x['answer_correct'] for x in scores]
            data['number_of_samples'] = [
                x['number_of_samples'] for x in scores
            ]
            dump(data, record_file)

        data = load(record_file)

        ret = VGRPBench_score(data).round(1)
        dump(ret, score_file)

        return ret


class MMVet(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MMVet':
        'https://opencompass.openxlab.space/utils/VLMEval/MMVet.tsv',
        'MMVet_Hard':
        'http://opencompass.openxlab.space/utils/VLMEval/MMVet_Hard.tsv'
    }
    DATASET_MD5 = {
        'MMVet': '748aa6d4aa9d4de798306a63718455e3',
        'MMVet_Hard': '63a598819a936a2e77c410a78a21ff16'
    }

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mmvet import MMVet_auxeval, MMVet_acc

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs['model']
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=3, **judge_kwargs)
            assert model.working(), 'MMVet evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MMVet_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['score'] == v[
                        'score']
            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score, score_fine = MMVet_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        score_fine_pth = storage.replace('.xlsx', '_score_fine.csv')
        dump(score, score_pth)
        dump(score_fine, score_fine_pth)
        return score


class MTVQADataset(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MTVQA_TEST':
        'https://opencompass.openxlab.space/utils/VLMEval/MTVQA_TEST.tsv'
    }
    DATASET_MD5 = {'MTVQA_TEST': 'd87c17dbab934b7cd89c0a3c1c5657f4'}

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data and 'category' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        if 'split' in data:
            assert np.all([x.lower() == 'test' for x in data['split']
                           ]), 'We only support MTVQA_TEST for now. '
        lt = len(data)
        category_scores = defaultdict(list)
        for i in range(lt):
            line = data.iloc[i]
            ans = line['answer'].strip().lower().replace('.', '')
            pred = line['prediction'].strip().lower().replace('.', '')
            cate = line['category']
            score = 1.0 if ans in pred else 0.0
            category_scores[cate].append(score)
            category_scores['Average'].append(score)
        # Calculate the average score for each category, the score is normalized to [0, 100]
        category_averages = {
            category: np.mean(scores) * 100
            for category, scores in category_scores.items()
        }

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.json')
        dump(category_averages, result_file)

        return category_averages

    # MT-VQA adopts a custom prompt
    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert sum([x['type'] == 'text' for x in msgs]) == 1
        for item in msgs:
            if item['type'] == 'text':
                item[
                    'value'] += '\nAnswer the question using a word or phrase in the language of the question.'
        return msgs


class WildDocBenchmark(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'WildDoc':
        "https://huggingface.co/datasets/jingqun/wilddoc-vlmeval/resolve/main/WildDoc.tsv"
    }
    DATASET_MD5 = {
        'WildDoc': '7b9a95e7ae26dad58be05685f59867aa',
    }

    def evaluate(self, eval_file, **judge_kwargs):
        import pandas as pd

        data = load(eval_file)
        DocVQA_df = pd.DataFrame(columns=data.columns)
        CharQA_df = pd.DataFrame(columns=data.columns)
        TableVQA_df = pd.DataFrame(columns=data.columns)

        for i in range(len(data)):
            line = data.iloc[i]
            benchmark_name = line["index"].split("-")[0]
            if benchmark_name == "DocVQA":
                DocVQA_df = pd.concat(
                    [DocVQA_df, pd.DataFrame([line])], ignore_index=True)
            elif benchmark_name == "ChartQA":
                CharQA_df = pd.concat(
                    [CharQA_df, pd.DataFrame([line])], ignore_index=True)
            elif benchmark_name == "TableVQA":
                TableVQA_df = pd.concat(
                    [TableVQA_df, pd.DataFrame([line])], ignore_index=True)
            else:
                raise ValueError(f"Unknown benchmark name {benchmark_name}")

        # calculate three subset separately
        from .utils.vqa_eval import hit_calculate, process_line_WildDoc, calculate_consistency_WildDoc, calculate_overall_accuracy_WildDoc  # noqa: E501

        # 1. DocVQA
        data = DocVQA_df
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        lt = len(data)
        pool = mp.Pool(16)
        lines = [data.iloc[i] for i in range(lt)]
        DocVQA_res = pool.map(partial(process_line_WildDoc, method='anls'),
                              lines)
        hit = hit_calculate(DocVQA_res, "DocVQA")
        DocVQA_overall = np.mean(hit) * 100
        DocVQA_consistency_score = calculate_consistency_WildDoc(DocVQA_res)

        # 2. ChartQA
        data = CharQA_df
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        lt = len(data)
        pool = mp.Pool(16)
        lines = [data.iloc[i] for i in range(lt)]
        ChartQA_res = pool.map(
            partial(process_line_WildDoc, method='relaxed_accuracy'), lines)
        hit = hit_calculate(ChartQA_res, "ChartQA")
        ChartQA_overall = np.mean(hit) * 100
        ChartQA_consistency_score = calculate_consistency_WildDoc(ChartQA_res)

        # 3. TableVQA
        data = TableVQA_df
        assert 'answer' in data and 'prediction' in data
        import pandas as pd
        from .utils.tablevqabench import evaluate_fintabnet, evaluate_tabfact, evaluate_wtq

        data['prediction'] = data['prediction'].str.replace('^Answer: ',
                                                            '',
                                                            regex=True)
        ##################
        TableVQA_res = {
            "fintabnetqa": [],
            "vtabfact": [],
            "vwtq": [],
            "vwtq_syn": []
        }
        lt = len(data)
        for i in range(lt):
            line = data.iloc[i]
            ret = {'index': line["index"]}
            ans = line['answer']
            pred = line["prediction"]
            from .utils.tablevqabench import fintabnet_normalize, tsv_unescape_list, to_value_list, check_denotation

            subset = line["index"].split("-")[1]
            if subset == "fintabnetqa":
                pred, preds = fintabnet_normalize(pred)
                gt, gts = fintabnet_normalize(ans)
                correct = 1 if gt == pred or any(_pred == _gt
                                                 for _pred in preds
                                                 for _gt in gts) else 0
            elif subset == "vtabfact":
                pred = pred.lower()
                gt = ans
                if 'true' in pred and 'false' in pred:
                    correct = 0
                elif 'true' in pred and gt == '1':
                    correct = 1
                elif 'false' in pred and gt == '0':
                    correct = 1
                else:
                    correct = 0
            elif subset == "vwtq_syn" or subset == "vwtq":
                pred = str(pred).replace('||', '|')
                if pred == "nan":
                    pred = ""
                gt = ans
                original_strings = tsv_unescape_list(gt)
                target_values = to_value_list(original_strings)

                predicted_strings = tsv_unescape_list(pred)
                predicted_values = to_value_list(predicted_strings)
                correct = 1 if check_denotation(target_values,
                                                predicted_values) else 0
            else:
                raise ValueError(f"Unknown benchmark name {benchmark_name}")

            ret["pred"] = pred
            ret["gt"] = gt
            ret["match"] = correct
            TableVQA_res[subset].append(ret)

        TableVQA_overall = np.mean([
            np.mean(hit_calculate(x, "TableVQA"))
            for x in TableVQA_res.values()
        ]) * 100
        TableVQA_consistency_score = np.mean(
            [calculate_consistency_WildDoc(x) for x in TableVQA_res.values()])

        eval_results = {
            "DocVQA": {
                "Overall": DocVQA_overall,
                "Consistency": DocVQA_consistency_score
            },
            "ChartQA": {
                "Overall": ChartQA_overall,
                "Consistency": ChartQA_consistency_score
            },
            "TableVQA": {
                "Overall": TableVQA_overall,
                "Consistency": TableVQA_consistency_score
            },
            # Overall
            "WildDoc": {
                "Overall":
                np.mean([DocVQA_overall, ChartQA_overall, TableVQA_overall]),
                "Consistency":
                np.mean([
                    DocVQA_consistency_score, ChartQA_consistency_score,
                    TableVQA_consistency_score
                ])
            }
        }

        # 转换为长格式DataFrame
        ret_df = pd.DataFrame([
            {"Task": task, "Metric": metric, "Score": score}
            for task, metrics in eval_results.items()
            for metric, score in metrics.items()
        ])
        return ret_df

    # WildDoc adopts a custom prompt for each subset
    def build_prompt(self, line):
        index = line["index"]
        benchmark_name = index.split("-")[0]
        print(benchmark_name)

        from .utils.tablevqabench import FINTABNETQA_PROMPT, VTABFACT_PROMPT, VWTQ_PROMPT

        msgs = super().build_prompt(line)
        assert sum([x['type'] == 'text' for x in msgs]) == 1
        for item in msgs:
            if item['type'] == 'text':
                if benchmark_name == "TableVQA":
                    split = index.split("-")[1]
                    if split == 'fintabnetqa':
                        item['value'] = FINTABNETQA_PROMPT.format_map(
                            {'question': item['value']})
                    elif split == 'vtabfact':
                        item['value'] = VTABFACT_PROMPT.format_map(
                            {'question': item['value']})
                    elif split == 'vwtq_syn' or split == 'vwtq':
                        item['value'] = VWTQ_PROMPT.format_map(
                            {'question': item['value']})
                    else:
                        raise ValueError(f"Unknown split {line['split']}")
                    print(item)
                else:
                    item[
                        'value'] += '\nAnswer the question using a single word or phrase.'
        print(item)
        return msgs


class TableVQABench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'TableVQABench':
        'https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/mentor-vil/datasets/tablevqa-bench.tsv'
    }
    DATASET_MD5 = {'TableVQABench': '2550adc61bdc82d8e62f3b003de7c62d'}

    from .utils.tablevqabench import FINTABNETQA_PROMPT, VTABFACT_PROMPT, VWTQ_PROMPT

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        import pandas as pd
        from .utils.tablevqabench import evaluate_fintabnet, evaluate_tabfact, evaluate_wtq

        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data

        data['prediction'] = data['prediction'].str.replace('^Answer: ',
                                                            '',
                                                            regex=True)
        data_group = dict(tuple(data.groupby('split')))
        eval_result = {'split': [], 'average_scores': []}
        for split in ['fintabnetqa', 'vtabfact', 'vwtq', 'vwtq_syn']:
            data_split = data_group[split].to_dict(orient='records')
            if split == 'fintabnetqa':
                split_eval_meta = evaluate_fintabnet(data_split, ['accuracy'])
            elif split == 'vtabfact':
                split_eval_meta = evaluate_tabfact(data_split, ['accuracy'])
            elif split == 'vwtq' or split == 'vwtq_syn':
                split_eval_meta = evaluate_wtq(data_split, ['accuracy'])
            eval_result['split'].append(split)
            eval_result['average_scores'].append(
                split_eval_meta['average_scores'])

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        eval_result = pd.DataFrame(eval_result)
        dump(eval_result, result_file)

        return eval_result

    # TableVQABench adopts a custom prompt
    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert sum([x['type'] == 'text' for x in msgs]) == 1
        for item in msgs:
            if item['type'] == 'text':
                if line['split'] == 'fintabnetqa':
                    item['value'] = self.FINTABNETQA_PROMPT.format_map(
                        {'question': item['value']})
                elif line['split'] == 'vtabfact':
                    item['value'] = self.VTABFACT_PROMPT.format_map(
                        {'question': item['value']})
                elif line['split'] == 'vwtq_syn' or line['split'] == 'vwtq':
                    item['value'] = self.VWTQ_PROMPT.format_map(
                        {'question': item['value']})
        return msgs


class CustomVQADataset(ImageBaseDataset):
    TYPE = 'VQA'

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get(
                    'FORCE_LOCAL', None):
                from ..tools import LOCALIZE

                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def evaluate(self, eval_file, **judge_kwargs):
        raise NotImplementedError


class CRPE(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'CRPE_EXIST':
        'https://huggingface.co/datasets/petter12321/crpe_vlmevalkit/resolve/main/CRPE_EXIST.tsv',
        'CRPE_RELATION':
        'https://huggingface.co/datasets/petter12321/crpe_vlmevalkit/resolve/main/CRPE_RELATION.tsv'
    }
    DATASET_MD5 = {
        'CRPE_EXIST': '315584e23ac1ff7f8719ed3b7ad90f08',
        'CRPE_RELATION': 'bad7094cde0b572288f4b119c2d0c656'
    }

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.crpe import is_correct
        # find-image, count-text, find-text,
        # infer-choose, count-image, visual-reasoning
        score = {
            'exist': 0,
            'subject': 0,
            'predicate': 0,
            'object': 0,
            'total': 0,
        }
        num = {
            'exist': 0,
            'subject': 0,
            'predicate': 0,
            'object': 0,
            'total': 0,
        }
        final_score_dict = {
            'exist': 0,
            'subject': 0,
            'predicate': 0,
            'object': 0,
            'total': 0,
        }
        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            predict = str(line['prediction'])
            answers = str(line['answer'])
            # print("predict =", predict)
            # print("answers =", answers)
            category = line['category']
            if is_correct(answers, predict):
                score[category] += 1
                score['total'] += 1
            num[category] += 1
            num['total'] += 1

        for category in ['exist', 'subject', 'predicate', 'object', 'total']:
            if num[category] != 0:
                final_score_dict[category] = score[category] / num[category]
            else:
                final_score_dict[category] = None

        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict

    def build_prompt(self, line):
        ROOT = LMUDataRoot()
        msgs = super().build_prompt(line)
        for msg in msgs:
            if msg['type'] == 'image':
                msg['value'] = osp.join(
                    osp.join(ROOT, 'images', self.dataset_name), msg['value'])
        return msgs


class QSpatial(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {'QSpatial_plus': '', 'QSpatial_scannet': ''}

    # NOTE: To evaluate Q-Spatial-ScanNet, you need to get the permission from ScanNet website
    # Once you get the permission, you can use the helper code here to download and extract necessary images:
    # https://github.com/andrewliao11/Q-Spatial-Bench-code?tab=readme-ov-file#for-qspatial_scannet
    qspatial_root = "TO_BE_REPLACED_WITH_THE_PATH_TO_QSPATIAL_DATASET"
    url = "https://raw.githubusercontent.com/andrewliao11/Q-Spatial-Bench-code/refs/heads/main/prompt_templates/"

    def post_build(self, dataset):
        # Download the prompt templates from github

        links = [
            self.url + "system_prompt.txt",
            self.url + "spatial_prompt_single.txt",
            self.url + "spatial_prompt_steps.txt",
            self.url + "standard_prompt.txt", self.url + "zero_shot_prompt.txt"
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            for link in links:
                tgt_path = os.path.join(temp_dir, link.split("/")[-1])
                os.system(f"wget {link} -O {tgt_path}")

            self.system_prompt = open(
                os.path.join(temp_dir, "system_prompt.txt")).read()
            self._prompt_templates = dict(
                spatial_prompt_single=open(
                    os.path.join(temp_dir,
                                 "spatial_prompt_single.txt")).read(),
                spatial_prompt_steps=open(
                    os.path.join(temp_dir, "spatial_prompt_steps.txt")).read(),
                standard_prompt=open(
                    os.path.join(temp_dir, "standard_prompt.txt")).read(),
                zero_shot_prompt=open(
                    os.path.join(temp_dir, "zero_shot_prompt.txt")).read(),
            )

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        from jinja2.sandbox import SandboxedEnvironment
        text_prompt_template = self._prompt_templates["spatial_prompt_single"]
        env = SandboxedEnvironment()
        text_prompt = env.from_string(text_prompt_template).render(
            question=line["question"])
        tgt_path = self.dump_image(line)

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]

        msgs.append(
            dict(type='text', value=f"{self.system_prompt}\n{text_prompt}"))
        return msgs

    # Given the dataset name, return the dataset as a pandas dataframe, can override
    def load_data(self, dataset):
        import io
        import pandas as pd
        from datasets import load_dataset

        hf_dataset = load_dataset("andrewliao11/Q-Spatial-Bench",
                                  split=dataset)
        df = hf_dataset.to_pandas()

        df.reset_index(drop=True, inplace=True)
        df['index'] = df.index
        df['answer'] = list(zip(df['answer_value'], df['answer_unit']))
        df = df[['index'] + [col for col in df.columns if col != 'index']]

        if dataset == "QSpatial_scannet":
            df = df.drop(columns=["image"])
            df["image"] = [
                Image.open(os.path.join(self.qspatial_root, image_path))
                for image_path in df["image_path"]
            ]
        else:
            df["image"] = [
                Image.open(io.BytesIO(image_dict["bytes"]))
                for image_dict in df["image"]
            ]

        df["image"] = [encode_image_to_base64(image) for image in df["image"]]
        return df

    @classmethod
    def get_multiplier(self, unit):

        unit = unit.lower()
        if unit in ["meters", "meter", "m", "metre", "metres"]:
            multiplier = 100
        elif unit in ["centimeters", "centimeter", "cm"]:
            multiplier = 1
        elif unit in ["feet", "foot", "ft"]:
            multiplier = 30.48
        elif unit in ["inch", "inches", "in"]:
            multiplier = 2.54
        elif unit in ["mm"]:
            multiplier = 0.1
        else:
            print(f"Unknown unit: {unit}")
            multiplier = 0.

        return multiplier

    @classmethod
    def parse_string(self, input_str):
        # Regular expression to match the pattern (number or range, text)
        match = re.match(r'\(([\d.-]+), (.+)\)', input_str)
        if match:
            number_part = match.group(1)
            text = match.group(2)

            if '-' in number_part:
                start, end = map(float, number_part.split('-'))
                number = (start + end) / 2
            else:
                number = float(number_part)

            return number * self.get_multiplier(text)
        else:
            print(f"Unable to parse the input string {input_str}")
            return 0

    @classmethod
    def parse_prediction(self, vlm_response):
        # Value
        pattern = r'scalar{([^}]*)}'
        str_inside_scalar_boxes = re.findall(pattern, vlm_response)[-1]
        scalar_list = re.findall(r'\d+\.?\d*', str_inside_scalar_boxes)
        parsed_scalar = np.array(scalar_list).astype(float).mean()

        # Unit
        pattern = r'distance_unit{([^}]*)}'
        str_inside_unit_boxes = re.findall(pattern, vlm_response)
        parsed_unit = str_inside_unit_boxes[-1]

        pred_value_in_cms = parsed_scalar * self.get_multiplier(parsed_unit)
        return pred_value_in_cms

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):

        data = load(eval_file)
        if "model" in judge_kwargs:
            from .utils.qspatial import QSpatial_auxeval

            # extract using model
            model = judge_kwargs['model']
            suffix = eval_file.split('.')[-1]
            storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
            tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
            nproc = judge_kwargs.pop('nproc', 4)

            if not osp.exists(storage):
                model = build_judge(max_tokens=128, **judge_kwargs)

                assert model.working(), 'Evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
                lt = len(data)
                lines = [data.iloc[i] for i in range(lt)]
                tups = [(model, line) for line in lines]
                indices = [line['index'] for line in lines]

                ans = {}
                if osp.exists(tmp_file):
                    ans = load(tmp_file)
                tups = [x for x, i in zip(tups, indices) if i not in ans]
                indices = [i for i in indices if i not in ans]

                if len(indices):
                    new_results = track_progress_rich(
                        QSpatial_auxeval,
                        tups,
                        nproc=nproc,
                        chunksize=nproc,
                        keys=indices,
                        save=tmp_file,
                    )
                    ans = load(tmp_file)
                    for k, v in zip(indices, new_results):
                        assert k in ans
                        assert ans[k]['log'] == v['log'] and ans[k][
                            'res'] == v['res']

                data['res'] = [ans[idx]['res'] for idx in data['index']]
                data['log'] = [ans[idx]['log'] for idx in data['index']]
                dump(data, storage)

            data = load(storage)

            pred_value_in_cms = []
            for res in data["res"]:
                try:
                    pred_value_in_cms.append(self.parse_string(res))
                except ValueError:
                    pred_value_in_cms.append(0.)

            pred_value_in_cms = np.array(pred_value_in_cms) + 1e-8
        else:
            # regex parsing
            pred_value_in_cms = []
            n_errors_in_parsing = 0
            for pred in data["prediction"]:
                try:
                    parsed_value = self.parse_prediction(pred)
                except IndexError:
                    n_errors_in_parsing += 1
                    parsed_value = 1e-8

                pred_value_in_cms.append(parsed_value)

            print(f"Encounter {n_errors_in_parsing} errors in parsing")
            pred_value_in_cms = np.array(pred_value_in_cms) + 1e-8

        # Ground truth
        ground_truth_value_in_cms = []
        for answer in data["answer"]:
            value, unit = eval(answer)
            ground_truth_value_in_cms.append(value * self.get_multiplier(unit))
        ground_truth_value_in_cms = np.array(ground_truth_value_in_cms) + 1e-8

        # Calculate the score
        pred_gt = pred_value_in_cms / ground_truth_value_in_cms
        gt_pred = ground_truth_value_in_cms / pred_value_in_cms
        delta_2 = np.stack([pred_gt, gt_pred]).max(0) < 2.
        delta_1_point_5 = np.stack([pred_gt, gt_pred]).max(0) < 1.5

        data["eval_score_delta_2"] = delta_2
        data["eval_score_delta_1_point_5"] = delta_1_point_5

        final_score_dict = {
            "delta_2": delta_2.mean(),
            "delta_1_point_5": delta_1_point_5.mean()
        }
        for question_type in set(data["question_type"]):
            filtered_data = data[data["question_type"] == question_type]
            delta_2_per_question_type = filtered_data[
                "eval_score_delta_2"].mean()
            delta_1_point_5_per_question_type = filtered_data[
                "eval_score_delta_1_point_5"].mean()
            final_score_dict.update(
                {f"{question_type}_delta_2": delta_2_per_question_type})
            final_score_dict.update({
                f"{question_type}_delta_1_point_5":
                delta_1_point_5_per_question_type
            })

        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict


class MMNIAH(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MM_NIAH_VAL':
        'https://huggingface.co/datasets/petter12321/MM-NIAH-VLMEvalKit/resolve/main/MM_NIAH_VAL.tsv',
        'MM_NIAH_TEST': [
            'https://huggingface.co/datasets/petter12321/MM-NIAH-VLMEvalKit/resolve/main/part-aa',
            'https://huggingface.co/datasets/petter12321/MM-NIAH-VLMEvalKit/resolve/main/part-ab',
            'https://huggingface.co/datasets/petter12321/MM-NIAH-VLMEvalKit/resolve/main/part-ac',
            'https://huggingface.co/datasets/petter12321/MM-NIAH-VLMEvalKit/resolve/main/part-ad',
            'https://huggingface.co/datasets/petter12321/MM-NIAH-VLMEvalKit/resolve/main/part-ae'
        ]
    }
    DATASET_MD5 = {
        'MM_NIAH_VAL': '27e5a8c3cef7746cb38f89cd86c474c5',
        'MM_NIAH_TEST': 'f490eb2a43096307465fe9e7ef13497c'
    }

    def prepare_tsv(self, url, file_md5=None):
        import os
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        update_flag = False
        file_name = 'MM_NIAH_VAL.tsv' if 'MM_NIAH_VAL' in url else 'MM_NIAH_TEST.tsv'
        data_path = osp.join(data_root, file_name)
        if osp.exists(data_path) and (file_md5 is None
                                      or md5(data_path) == file_md5):
            pass
        elif file_name == 'MM_NIAH_TEST.tsv':
            warnings.warn('The dataset tsv is not downloaded')
            for i in range(len(url)):
                if osp.exists(osp.join(data_root,
                                       'part-a' + chr(ord('a') + i))):
                    print('part_a' + chr(ord('a') + i) + ' is existed')
                    continue
                download_file(url[i], data_path)
            file_prefix = 'part-'
            output_file = data_path
            split_files = sorted([
                f for f in os.listdir(data_root) if f.startswith(file_prefix)
            ])
            with open(output_file, 'wb') as outfile:
                # 逐个读取每个拆分文件并写入到输出文件
                for filename in split_files:
                    with open(osp.join(data_root, filename), 'rb') as infile:
                        outfile.write(infile.read())
            update_flag = True
        else:
            warnings.warn('The dataset tsv is not downloaded')
            download_file(url, data_path)
            update_flag = True

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get(
                    'FORCE_LOCAL', None) or update_flag:
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mmniah import is_correct
        # find-image, count-text, find-text,
        # infer-choose, count-image, visual-reasoning
        MMNIAH_score = {
            'count-text': 0,
            'find-image': 0,
            'find-text': 0,
            'infer-choose': 0,
            'count-image': 0,
            'visual-reasoning': 0,
            'total': 0,
        }
        MMNIAH_num = {
            'count-text': 0,
            'find-image': 0,
            'find-text': 0,
            'infer-choose': 0,
            'count-image': 0,
            'visual-reasoning': 0,
            'total': 0,
        }
        final_score_dict = {
            'count-text': 0,
            'find-image': 0,
            'find-text': 0,
            'infer-choose': 0,
            'count-image': 0,
            'visual-reasoning': 0,
            'total': 0,
        }
        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            predict = line['prediction']
            answers = line['answer']
            category = line['category']
            if category in ['visual-reasoning', 'find-image']:
                answers = int(answers)
            if is_correct(answers, predict):
                MMNIAH_score[category] += 1
                MMNIAH_score['total'] += 1
            MMNIAH_num[category] += 1
            MMNIAH_num['total'] += 1

        for category in [
                'find-image', 'count-text', 'find-text', 'infer-choose',
                'count-image', 'visual-reasoning', 'total'
        ]:
            if MMNIAH_num[category] != 0:
                final_score_dict[
                    category] = MMNIAH_score[category] / MMNIAH_num[category]
            else:
                final_score_dict[category] = None

        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        if isinstance(line, int):
            line = self.data.iloc[line]
        totalchoice = line['multi-choice options']
        totalchoice = eval(totalchoice)
        # find-image, count-text, find-text,
        # infer-choose, count-image, visual-reasoning
        context = msgs[-1]['value']
        context = eval(context)
        question = context[0] + '\n' + context[1]
        # tgt_path是所有图像地址列表
        tgt_path = []
        for i in range(len(msgs) - 1):
            tgt_path.append(msgs[i]['value'])
        choices = totalchoice[0]
        choices_image = totalchoice[1]
        if choices:
            for c_idx, c in enumerate(choices):
                question = f"{question}\n{chr(c_idx + ord('A'))}. {c}"
            question += "\nAnswer with the option's letter from the given choices directly."
        elif choices_image:
            for c_idx in range(len(choices_image)):
                question = f"{question}\n{chr(c_idx + ord('A'))}. <image>"
            question += "\nAnswer with the option's letter from the given choices directly."
        else:
            question += '\nAnswer the question using a single word or phrase.'
        question = '<start>' + question + '<end>'
        question = question.split('<image>')
        if choices_image:
            for i in range(len(question) - 5):
                question[i] = question[i] + '\n<image>'
            for i in range(len(question) - 5, len(question) - 1):
                question[i] = question[i] + '<image>'
        else:
            for i in range(len(question) - 1):
                question[i] = question[i] + '\n<image>'
        assert len(tgt_path) + 1 == len(question)
        context = []
        for i in range(len(tgt_path)):
            context.append(question[i])
            context.append(tgt_path[i])
        context.append(question[-1])
        context[0] = context[0][7:]
        context[-1] = context[-1][:-5]
        msgs = []
        for i in range(len(context)):
            if i % 2 == 0:
                msgs.append(dict(type='text', value=context[i]))
            else:
                ROOT = LMUDataRoot()
                msgs.append(
                    dict(type='image',
                         value=osp.join(
                             osp.join(ROOT, 'images', self.dataset_name),
                             context[i])))
        for element in msgs:
            if element['value'] == '':
                msgs.remove(element)
        return msgs


class MMSci_Captioning(ImageBaseDataset):

    TYPE = 'MMSci_Captioning'
    DATASET_URL = {
        'MMSci_DEV_Captioning_image_only':
        'https://opencompass.openxlab.space/utils/VLMEval/MMSci_DEV_Captioning_image_only.tsv',  # noqa: E501
        'MMSci_DEV_Captioning_with_abs':
        'https://opencompass.openxlab.space/utils/VLMEval/MMSci_DEV_Captioning_with_abs.tsv'  # noqa: E501
    }

    DATASET_MD5 = {
        'MMSci_DEV_Captioning_image_only': '0f5f0fd7ff383699fbd2203a4659d3e8',
        'MMSci_DEV_Captioning_with_abs': 'ae4a9b88166153efd74e28c989e4a484'
    }

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mmsci import (get_all_metrics_for_g_eval_score,
                                  get_all_metrics_for_reference_based_metrics,
                                  merge_rating, fact_score_generate)
        refer_based_metrics_output_file = eval_file.replace(
            '.xlsx', '_reference_based_metrics.xlsx')
        g_eval_metrics_output_file = eval_file.replace('.xlsx',
                                                       '_g_eval_metrics.xlsx')
        fact_score_metrics_output_file = eval_file.replace(
            '.xlsx', '_fact_score.xlsx')

        # calculate reference-based metrics
        if not osp.exists(refer_based_metrics_output_file):
            data = load(eval_file)
            old_candidates = {}
            old_references = {}
            for idx, item in data.iterrows():
                image_id = item["image_id"]
                old_candidates[image_id] = [item["prediction"]]
                old_references[image_id] = [item["caption"]]

            candidates = []
            references = []
            image_id_list = []
            image_ids = old_references.keys()
            for cid in image_ids:
                if cid in old_candidates:
                    candidates.append(old_candidates[cid][0])
                    references.append(old_references[cid])
                    image_id_list.append(cid)

            if isinstance(references[0], str):
                references = [[r] for r in references]

            reference_based_metrics_file = eval_file.replace(
                '.xlsx', '_reference_based_metrics.pkl')
            existing_data = get_all_metrics_for_reference_based_metrics(
                references, candidates, image_id_list,
                reference_based_metrics_file)
            for idx, item in data.iterrows():
                reference_based_metrics = str(existing_data[item["image_id"]])
                data.loc[idx,
                         'reference_based_metrics'] = reference_based_metrics
            dump(data, refer_based_metrics_output_file)

        # calculate g-eval metrics
        if not osp.exists(g_eval_metrics_output_file):

            data = load(eval_file)
            answers = {}
            for idx, item in data.iterrows():
                answers[item["abstract"]] = item["caption"]

            old_candidates = {}
            old_references = {}
            for idx, item in data.iterrows():
                caption = item['caption']
                if not caption:
                    caption = answers[item['abstract']]

                image_id = item['image_id']
                old_candidates[image_id] = [item["prediction"]]
                old_references[image_id] = [caption]

            candidates = []
            references = []
            image_id_list = []
            image_ids = old_references.keys()
            for cid in image_ids:
                if cid in old_candidates:
                    candidates.append(old_candidates[cid][0])
                    references.append(old_references[cid])
                    image_id_list.append(cid)

            if isinstance(references[0], str):
                references = [[r] for r in references]

            model = judge_kwargs.pop('model', 'gpt-4o-0806')
            nproc = judge_kwargs.pop('nproc', 4)
            # not supported gemini-1.5-pro-exp-0801 as judge model yet、
            assert model in ['gpt-4o-0806', 'gemini-1.5-pro-exp-0801']
            judge_model = build_judge(model=model, **judge_kwargs)

            assert judge_model.working(), (
                'Evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)
            suffix = '.' + eval_file.split('.')[-1]
            tmp_file = eval_file.replace(suffix, f'_{model}_G_eval.pkl')

            tmp_result = get_all_metrics_for_g_eval_score(
                references,
                candidates,
                evaluator=judge_model,
                tmp_file=tmp_file,
                nproc=nproc)

            indices = range(len(references))
            image_id_dict = {}
            for ind, img_id in zip(indices, image_id_list):
                image_id_dict[img_id] = ind

            for idx, item in data.iterrows():
                g_eval_metrics = tmp_result[image_id_dict[item["image_id"]]]
                data.loc[idx, 'g_eval_metrics'] = g_eval_metrics
            dump(data, g_eval_metrics_output_file)

        rating = merge_rating(refer_based_metrics_output_file,
                              g_eval_metrics_output_file,
                              fact_score_metrics_output_file)
        dump(rating, eval_file.replace('.xlsx', '_final_rating.xlsx'))
        return rating


class TDBenchGrounding(ImageVQADataset):
    DATASET_URL = {
        'tdbench_grounding_rot0':
        'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/tdbench_grounding_rot0.tsv',  # noqa: E501
        'tdbench_grounding_rot90':
        'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/tdbench_grounding_rot90.tsv',  # noqa: E501
        'tdbench_grounding_rot180':
        'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/tdbench_grounding_rot180.tsv',  # noqa: E501
        'tdbench_grounding_rot270':
        'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/tdbench_grounding_rot270.tsv',  # noqa: E501
    }

    DATASET_MD5 = {
        'tdbench_grounding_rot0': '4c2e16c797dfc27f23c06dbb82993a1e',
        'tdbench_grounding_rot90': '1c75f08b0db07a655333217bf32052c3',
        'tdbench_grounding_rot180': '794d15f88392a3b8ae5399e58b0f7327',
        'tdbench_grounding_rot270': '76005a412b49f54eedb6ef963cd0affb',
    }

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.tdbench import evaluate_bbox, extract_bbox_from_string, rotational_eval
        suffix = eval_file.split('.')[-1]
        method = judge_kwargs.get('model', 'centroid')
        assert method in ['centroid',
                          'iou'], '--judge should be either centroid or iou'

        data = load(eval_file).sort_values(by='index')
        predictions = [str(x) for x in data['prediction']]
        answers = [str(x) for x in data['answer']]
        indexes = [str(x) for x in data['index']]

        scores = []

        for idx, (pred, ans,
                  index) in enumerate(zip(predictions, answers, indexes)):
            try:
                pred_bbox = extract_bbox_from_string(pred)
                gt_bbox = extract_bbox_from_string(ans)

                score = evaluate_bbox(pred_bbox, gt_bbox, method)
                scores.append(score)
            except Exception as e:
                print(
                    f"Error calculating {method} for index {index}. Marking as incorrect: {e}"
                )
                scores.append(0.0)

        avg_score = sum(scores) / len(scores) if scores else 0

        data['hit'] = scores
        data['category'] = 'visual_grounding'
        result_file = eval_file.replace(f'.{suffix}', f'_{method}_result.xlsx')
        data.to_excel(result_file, index=False)

        metric_name = 'Average Centroid Containment' if method == 'centroid' else 'Average IoU'
        summary_scores = {metric_name: avg_score, 'Total Samples': len(scores)}

        score_df = pd.DataFrame(list(summary_scores.items()),
                                columns=['Metric', 'Score'])
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        score_df.to_csv(score_file, index=False)
        re_result = rotational_eval(result_file)
        if method == 'centroid' and re_result is not None and re_result is not False:
            file_addr = osp.abspath(
                result_file.split('_rot')[0] + '_REresult.csv')
            link_addr = osp.join(osp.dirname(osp.dirname(result_file)),
                                 osp.basename(file_addr))
            re_result.to_csv(file_addr, index=True)
            print(tabulate(re_result, headers="keys"))
            if osp.exists(link_addr) or osp.islink(link_addr):
                os.remove(link_addr)
            os.symlink(file_addr, link_addr)
        return summary_scores

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        obj = line['question']
        question = f'\nPlease output the coordinates of the {obj} in the image in the format [x1, y1, x2, y2]. Do not include any additional text. Respond with relative coordinates between 0 and 1, with top left corner (0, 0), top right (1, 0) and bottom right (1, 1).'  # noqa: E501
        tgt_path = self.dump_image(line)

        msgs = []
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
        msgs.append(dict(type='text', value=question))
        return msgs


class ZEROBench(ImageVQADataset):
    DATASET_URL = {'ZEROBench': 'https://opencompass.openxlab.space/utils/VLMEval/zerobench.tsv',
                   'ZEROBench_sub': 'https://opencompass.openxlab.space/utils/VLMEval/zerobench_sub.tsv'}
    DATASET_MD5 = {'ZEROBench': '600d5e89325f1dab5ad3fa2ea200cea6',
                   'ZEROBench_sub': '2d2131bffb7f09ca099fdd0f3ad0392b'}

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file).sort_values(by='index')
        predictions = [str(x) for x in data['prediction']]
        answers = [str(x) for x in data['answers']]
        indexes = [str(x) for x in data['index']]
        output_df = pd.DataFrame(columns=["Question_ID", "Ground_Truth", "Model_Output", "Correct?"])
        for idx, (pred, ans, index) in enumerate(zip(predictions, answers, indexes)):
            formatted_response = pred.strip()
            # convert to lowercase
            formatted_response = formatted_response.lower()
            # try to extract final answer from curly braces
            parsed_answer = ''
            try:
                pattern = r"\{(.*?)\}"
                parsed_answer = re.findall(pattern, formatted_response)[-1]
            except IndexError:
                pass

            # evaluate via exact matching
            correct = ans.strip().lower() in parsed_answer.lower()
            # store results
            results_row = {"Question_ID": idx,
                           "Ground_Truth": ans,
                           "Model_Output": pred,
                           "Correct?": correct}
            output_df = pd.concat([output_df, pd.DataFrame([results_row])],
                                  ignore_index=True)

        # compute accuracy
        accuracy = output_df["Correct?"].mean()
        return {"accuracy": accuracy}

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        obj = line['question']
        question = f"{obj} \n\n\nLet's think step by step and give the final answer in curly braces,  like this: {{final answer}}"   # noqa: E501
        tgt_path = self.dump_image(line)
        msgs = []
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
        msgs.append(dict(type='text', value=question))
        return msgs


class CountBenchQA(ImageVQADataset):
    TYPE = "VQA"
    DATASET_URL = {
        "CountBenchQA":
        "https://opencompass.openxlab.space/utils/VLMEval/CountBenchQA.tsv"
    }
    DATASET_MD5 = {"CountBenchQA": "fc73c8d4ffa665431448753f094d56ff"}

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)
        msgs = []
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
        ques = line['question']
        question = f'{ques} Note that: answer with a number directly e.g. 3. Do not include any additional text.'
        msgs.append(dict(type='text', value=question))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file).sort_values(by='index')
        predictions = [str(x) for x in data['prediction']]
        answers = [str(x) for x in data['answer']]
        correct_count = 0
        total_count = len(predictions)

        for pred, ans in zip(predictions, answers):
            if ans in pred:
                correct_count += 1
        accuracy = correct_count / total_count if total_count > 0 else 0
        return {'accuracy': accuracy}


class OCR_Reasoning(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'OCR_Reasoning':
        'https://opencompass.openxlab.space/utils/VLMEval/OCR_Reasoning.tsv'
    }
    DATASET_MD5 = {'OCR_Reasoning': 'cf95ace31742170cf669ef45e0dae267'}

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.ocr_reasoning import OcrR_auxeval, OcrR_acc

        model = judge_kwargs['model']
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        nproc = 1
        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=1024, **judge_kwargs)
            assert model.working(), 'OCRReasoning evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    OcrR_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v[
                        'res']

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            data['reason_score'] = [
                ans[idx]['reason_score'] for idx in data['index']
            ]
            dump(data, storage)
        score = OcrR_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        dump(score, score_pth)
        return score

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert sum([x['type'] == 'text' for x in msgs]) == 1
        for item in msgs:
            if item['type'] == 'text':
                if 'English' in line['language']:
                    item[
                        'value'] += f"\nSolve the complex problem through step-by-step reasoning. The composition of the final answer should be: '{line['format']}'."  # noqa e501
                else:
                    item[
                        'value'] += f"\n通过一步一步的推理解决这个复杂的问题。最后的回答的组成应该是: '{line['format']}'."
        return msgs


class PhyX(ImageBaseDataset):
    TYPE = 'VQA'

    def __init__(self, dataset='PhyX_mini', skip_noimg=True):
        if dataset != 'PhyX_mini_OE':
            import warnings
            warnings.warn(
                'To evaluate on PhyX, we would suggest `PhyX_mini_OE` for the default setting.'
            )
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

    DATASET_URL = {
        'PhyX_MC':
        'https://huggingface.co/datasets/Cloudriver/PhyX/resolve/main/data_tsv_vlmevalkit/PhyX_MC.tsv',  # noqa
        'PhyX_OE':
        'https://huggingface.co/datasets/Cloudriver/PhyX/resolve/main/data_tsv_vlmevalkit/PhyX_OE.tsv',  # noqa
        'PhyX_mini_MC':
        'https://huggingface.co/datasets/Cloudriver/PhyX/resolve/main/data_tsv_vlmevalkit/PhyX_mini_MC.tsv',  # noqa
        'PhyX_mini_OE':
        'https://huggingface.co/datasets/Cloudriver/PhyX/resolve/main/data_tsv_vlmevalkit/PhyX_mini_OE.tsv',  # noqa
    }

    DATASET_MD5 = {
        'PhyX_MC': '36e841a2c12e605cdc940e5e36d89213',  # noqa
        'PhyX_OE': '8abdd9802a78c688c3a38fe0c785ee7a',  # noqa
        'PhyX_mini_MC': '3587d8804b66a8e7678f3496716fa84f',  # noqa
        'PhyX_mini_OE': '1f646ccb086ef28453f8c260441734f3',  # noqa
    }

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        msgs = []
        if "TL" in self.dataset_name:
            # pure text, do not load image
            pass
        else:
            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image', value=p) for p in tgt_path])
            else:
                msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        valid_type = judge_kwargs["valid_type"]
        assert valid_type in ["STR", "LLM"], print(
            "To evaluate PhyX, you need to set valid_type in judge-args, STR for string level and LLM for LLM."
            " Please add: --judge-args '{\"valid_type\": \"STR\"}' or add: "
            "--judge deepseek --judge-args '{\"valid_type\": \"LLM\"}' in your command."
        )
        if valid_type == "STR":
            # Match at string level
            from .utils.phyx import PhyX_process_line, PhyX_process_line_MC
            data = load(eval_file)
            assert 'answer' in data and 'prediction' in data
            data['prediction'] = [str(x) for x in data['prediction']]
            data['answer'] = [str(x) for x in data['answer']]
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            pool = mp.Pool(1)
            if "_MC" in eval_file:
                # Multi choice
                res = pool.map(partial(PhyX_process_line_MC), lines)
            else:
                # Open ended mode
                res = pool.map(partial(PhyX_process_line), lines)

            suffix = eval_file.split('.')[-1]
            result_file = eval_file.replace(f'.{suffix}', '_predict.xlsx')
            df = pd.DataFrame(res)
            df.to_excel(result_file, index=False)

            hit = [x['match'] for x in res]
            ret = dict()
            ret['Overall'] = np.mean(hit)

            if 'category' in data:
                cates = list(set(data['category']))
                cates.sort()
                for c in cates:
                    sub = [r for l, r in zip(lines, res) if l['category'] == c]
                    hit = [x['match'] for x in sub]
                    ret[c] = np.mean(hit)
            ret = d2df(ret)
            ret.round(2)

            suffix = eval_file.split('.')[-1]
            result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
            dump(ret, result_file)
            return ret

        elif valid_type == "LLM":
            from .utils.phyx import PhyX_auxeval, PhyX_acc, PhyX_auxeval_MC

            model = judge_kwargs['model']
            suffix = eval_file.split('.')[-1]
            storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
            tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
            nproc = judge_kwargs.pop('nproc', 4)

            if not osp.exists(storage):
                data = load(eval_file)
                model = build_judge(max_tokens=128, **judge_kwargs)
                assert model.working(), (
                    "PhyX evaluation requires a working API. We use Deepseek-V3 provided by SiliconFlow. "
                    "please set SiliconFlow_API_KEY.\n")
                lt = len(data)
                lines = [data.iloc[i] for i in range(lt)]
                tups = [(model, line) for line in lines]
                indices = [line['index'] for line in lines]

                ans = {}
                if osp.exists(tmp_file):
                    ans = load(tmp_file)
                tups = [x for x, i in zip(tups, indices) if i not in ans]
                indices = [i for i in indices if i not in ans]

                if len(indices):
                    if "_MC" in eval_file:
                        new_results = track_progress_rich(
                            PhyX_auxeval_MC,
                            tups,
                            nproc=nproc,
                            chunksize=nproc,
                            keys=indices,
                            save=tmp_file,
                        )
                    else:
                        new_results = track_progress_rich(
                            PhyX_auxeval,
                            tups,
                            nproc=nproc,
                            chunksize=nproc,
                            keys=indices,
                            save=tmp_file,
                        )
                    ans = load(tmp_file)
                    for k, v in zip(indices, new_results):
                        assert k in ans
                        assert ans[k]['log'] == v['log'] and ans[k][
                            'res'] == v['res']

                data['res'] = [ans[idx]['res'] for idx in data['index']]
                data['log'] = [ans[idx]['log'] for idx in data['index']]
                data['extracted'] = [
                    ans[idx]['extracted'] for idx in data['index']
                ]
                dump(data, storage)

            score = PhyX_acc(storage)
            score_pth = storage.replace('.xlsx', '_score.csv')
            dump(score, score_pth)
            return score


class TallyQA(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'TallyQA': 'https://huggingface.co/datasets/moondream/TallyQA-VLMEvalKit/resolve/main/tallyqa_data.tsv'
    }
    DATASET_MD5 = {'TallyQA': '959df01cf1858e73a71efe5cd3b9bf19'}

    def evaluate(self, eval_file, **judge_kwargs):
        import pandas as pd
        from .utils.tallyqa import extract_count_from_prediction

        data = load(eval_file)

        pred_ints = data["prediction"].apply(extract_count_from_prediction)
        answer_ints = data["answer"].astype(int)

        correct = (pred_ints == answer_ints).sum()
        total = len(data)
        accuracy = correct / total

        result_df = pd.DataFrame([{"TallyQA": accuracy}])
        result_file = eval_file.replace(f".{eval_file.split('.')[-1]}", "_acc.csv")
        dump(result_df, result_file)
        return result_df

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        msgs[-1]['value'] += '\nAnswer the question using a single number.'
        return msgs


class Omni3DBench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'Omni3DBench':
        'https://huggingface.co/datasets/rohunagrawal/Omni3DBench-VLMEvalKit/resolve/main/omni3dbench.tsv'
    }
    DATASET_MD5 = {'Omni3DBench': 'ba1fa59c3897eb95aed445996ec9b690'}

    def build_prompt(self, line):
        from .utils.omni3dbench import OMNI3DBENCH_PROMPT
        question = line['question']
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += f'{OMNI3DBENCH_PROMPT}{question} (Answer type: {line["answer_type"]})'
        return msgs

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.omni3dbench import Omni3DBench_acc

        data = load(eval_file)
        result = Omni3DBench_acc(data)
        return result


class MMEReasoning(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {'MME-Reasoning': 'https://huggingface.co/datasets/U4R/MME-Reasoning/blob/main/MME_Reasoning.tsv'}
    DATASET_MD = {'MME-Reasoning': 'b243f44778782d3821523689f6b40a1e'}

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))

        return msgs

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mme_reasoning import MMEReasoning_extract, MMEReasoning_openeval, MMEReasoning_acc, FAIL_MSG, mme_reasoning_eval_functions  # noqa

        model = judge_kwargs.get('model', 'gpt-4o-mini')
        suffix = eval_file.split('.')[-1]
        storage_extract = eval_file.replace(f'.{suffix}', f'_{model}_extract.xlsx')
        tmp_file_extract = eval_file.replace(f'.{suffix}', f'_{model}_extract.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        # stage 1: extract answers using LLM
        if not osp.exists(storage_extract):
            data = load(eval_file)
            data = data.replace({float('nan'): None})
            model = build_judge(max_tokens=1024, **judge_kwargs)
            assert model.working(), ('MME-Reasoning evaluation requires a working OPENAI API\n')
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file_extract):
                ans = load(tmp_file_extract)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]
            if len(indices):
                new_results = track_progress_rich(
                    MMEReasoning_extract,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file_extract,
                )
                ans = load(tmp_file_extract)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v['res']

            res_list = []
            log_list = []
            for idx in data['index']:
                if not isinstance(ans[idx], dict):
                    res_list.append(ans[idx])
                    log_list.append('use previous answer')
                else:
                    res_list.append(ans[idx]['res'])
                    log_list.append(ans[idx]['log'])
            # data['res'] = [ans[int(idx)]['res'] for idx in data['index']]
            # data['log'] = [ans[idx]['log'] for idx in int(data['index'])]
            data['res'] = res_list
            data['log'] = log_list
            dump(data, storage_extract)

        storage_score = eval_file.replace(f'.{suffix}', f'_{model}_score.xlsx')
        tmp_file_score = eval_file.replace(f'.{suffix}', f'_{model}_score.pkl')

        # stage 2: evaluate score
        if not osp.exists(storage_score):
            data = load(storage_extract)
            data = data.replace({float('nan'): None})
            model = build_judge(max_tokens=1024, **judge_kwargs)
            assert model.working(), ('MME-Reasoning evaluation requires a working OPENAI API\n')
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            lines_scores_gpt = []
            lines_scores_other = []
            for line in lines:
                if (line['question_type'].lower() == 'open' and line.get('function_id', None) == None) or line.get('function_id', None) == 'open_function':  # noqa
                    lines_scores_gpt.append(line)
                else:
                    lines_scores_other.append(line)

            # for open question, use LLM
            tups_scores_gpt = [(model, line) for line in lines_scores_gpt]
            indices_scores_gpt = [line['index'] for line in lines_scores_gpt]
            if len(indices_scores_gpt):
                new_results_score = track_progress_rich(
                    MMEReasoning_openeval,
                    tups_scores_gpt,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices_scores_gpt,
                    save=tmp_file_score,
                )
                ans = load(tmp_file_score)
                for k, v in zip(indices_scores_gpt, new_results_score):
                    assert k in ans
                    assert ans[k]['log_score'] == v['log_score'] and ans[k]['score'] == v['score']

            # for other questions, use corresponding function
            res = {}
            indices_scores_other = [line['index'] for line in lines_scores_other]
            for k, line in zip(indices_scores_other, lines_scores_other):
                if line['res'] is None or FAIL_MSG in line['res']:
                    log_score = 'Failed to evaluate'
                    res.update({
                        k: {
                            "log_score": log_score,
                            "score": False
                        }
                    })
                    continue

                if line['function_id'] is None:
                    assert line['question_type'].lower() == 'choice'
                    function_id = 'choice_function'
                    function = mme_reasoning_eval_functions[function_id]

                else:
                    function_id = line['function_id']
                    function = mme_reasoning_eval_functions[function_id]

                if function_id not in ['open_function', 'choice_function']:
                    if function_id == "judge_24points_function":
                        response = line['res']
                    else:
                        response = json.loads(line['res'])
                    if line['answer'] is not None:
                        answer = eval(line['answer'])
                    else:
                        answer = None

                    if function_id in [
                        "calculate_answer_function_hashi",
                        "calculate_answer_function_skyscraper",
                        "calculate_answer_function_sudoku_4",
                        "calculate_answer_function_sudoku_6",
                        "calculate_answer_function_yinyang",
                        "judge_24points_function"
                    ]:
                        assert line["special_info"] is not None
                        special_info = eval(line['special_info'])
                    else:
                        special_info = None
                else:
                    response = line['res']
                    answer = line['answer']
                    special_info = None

                if special_info is None:
                    answer_judge = function(response, answer)
                else:
                    answer_judge = function(response, answer, special_info)

                if answer_judge not in [True, False]:
                    log_score = 'Failed to evaluate'
                    score = False
                else:
                    log_score = 'Succeed'
                    score = answer_judge
                res.update({
                    k: {
                        "log_score": log_score,
                        "score": score
                    }
                })

            ans.update(res)

            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log_score'] = [ans[idx]['log_score'] for idx in data['index']]
            dump(data, storage_score)

        score = MMEReasoning_acc(storage_score)
        score_pth = storage_score.replace('.xlsx', '.csv')
        dump(score, score_pth)
        return score


class MMVMBench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MMVMBench':
        'https://opencompass.openxlab.space/utils/VLMEval/MMVMBench.tsv'
    }
    DATASET_MD5 = {'MMVMBench': '168823a449bc323bec5d309227943757'}

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)

        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))

        return msgs

    def report_acc_mmatch(scores, match_types_int):
        res = defaultdict(list)

        match_type = ['CL', 'SP', 'TM', 'SZ', 'RP', 'OO', 'BR', 'OM']
        match_type_dict = {i + 1: v for i, v in enumerate(match_type)}
        match_type_hit = {i + 1: [] for i, _ in enumerate(match_type)}
        overall_hits = []
        for match_type_i, score_i in zip(match_types_int, scores):
            try:
                match_type_i_list = eval(match_type_i)
            except:
                match_type_i_list = [1, 5]
            hit_i = 0
            for tag in ['[YES]', '[Yes]', '[yes]', 'YES', 'Yes', 'yes']:
                if tag in score_i:
                    hit_i = 1

            for mt in match_type_i_list:
                if mt not in match_type_hit.keys():
                    mt = 2
                match_type_hit[mt].append(hit_i)

            overall_hits.append(hit_i)

        res['Overall'] = [np.mean(overall_hits),]
        for k, v in match_type_dict.items():
            res[v] = np.mean(match_type_hit[k])

        return pd.DataFrame(res)

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'
        judge = judge_kwargs['model']
        nproc = judge_kwargs.pop('nproc', 4)

        tmp_file = eval_file.replace('.xlsx', f'_{judge}_tmp.pkl')
        score_file = eval_file.replace('.xlsx', f'_{judge}_score.xlsx')
        acc_file = eval_file.replace('.xlsx', f'_{judge}_acc.xlsx')

        judge_kwargs['temperature'] = 0.0
        model = build_judge(**judge_kwargs)

        prompt = "I asked a model a question: {QUESTION}. The model's response was: {RESPONSE}. "\
            "The correct answer to this question is: {GT}. Please determine whether the model's "\
            "response is correct. If it is, conclude with [YES]. If it is not, conclude with [NO]."

        def prepare_score_prompt(line):
            vq = "{" + line['question'] + "}"
            va = "{" + line['prediction'] + "}"
            gt = "{" + line['answer'] + "}"
            question = prompt.format(QUESTION=vq, RESPONSE=va, GT=gt)
            return question

        if not osp.exists(score_file):
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if model.fail_msg not in v}

            data = load(eval_file)
            data_un = data[~data['index'].isin(res)]
            data_un = data_un[~pd.isna(data_un['prediction'])]
            lt = len(data_un)

            prompts = [prepare_score_prompt(data_un.iloc[i]) for i in range(lt)]
            indices = [data_un.iloc[i]['index'] for i in range(lt)]

            if len(prompts):
                _ = track_progress_rich(
                    model.generate,
                    prompts,
                    keys=indices,
                    save=tmp_file,
                    nproc=nproc,
                    chunksize=nproc
                )
                score_map = load(tmp_file)
                data['score'] = [score_map[idx] if idx in score_map else '[NO]' for idx in data['index']]
                FAIL_MSG = 'Failed to obtain answer via API.'
                rejected = [x for x in score_map.values() if FAIL_MSG in x]
                print(
                    f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(score_map)} '
                    f'questions, failed to obtain the score for another {len(rejected)} questions. '
                    'Those questions will be counted as 0 score in ALL rating, '
                    'and will not be counted in VALID rating.'
                )

                dump(data, score_file)

            scores = data['score']
            match_types_int = data['match_type']
            acc = self.report_acc_mmatch(scores, match_types_int)
            dump(acc, acc_file)

            return acc
