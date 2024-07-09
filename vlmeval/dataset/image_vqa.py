import multiprocessing
from functools import partial

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import track_progress_rich


class ImageVQADataset(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'OCRVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TEST.tsv',
        'OCRVQA_TESTCORE': 'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TESTCORE.tsv',
        'TextVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv',
        'DocVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv',
        'DocVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_TEST.tsv',
        'InfoVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv',
        'InfoVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_TEST.tsv',
        'ChartQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv',
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
    }

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += '\nAnswer the question using a single word or phrase.'
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
            res = pool.map(partial(process_line, method='relaxed_accuracy'), lines)
        elif listinstr(['OCRVQA'], dataset):
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


class OCRBench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'OCRBench': 'https://opencompass.openxlab.space/utils/VLMEval/OCRBench.tsv'
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
                    answer = answers[j].strip().replace('\n', ' ').replace(' ', '')
                    predict = predict.strip().replace('\n', ' ').replace(' ', '')
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
        final_score_dict['Scene Text-centric VQA'] = OCRBench_score['Scene Text-centric VQA']
        final_score_dict['Doc-oriented VQA'] = OCRBench_score['Doc-oriented VQA']
        final_score_dict['Key Information Extraction'] = OCRBench_score['Key Information Extraction']
        final_score_dict['Handwritten Mathematical Expression Recognition'] = \
            (OCRBench_score['Handwritten Mathematical Expression Recognition'])
        final_score_dict['Final Score'] = \
            (final_score_dict['Text Recognition'] + final_score_dict['Scene Text-centric VQA']
             + final_score_dict['Doc-oriented VQA'] + final_score_dict['Key Information Extraction']
             + final_score_dict['Handwritten Mathematical Expression Recognition'])
        final_score_dict['Final Score Norm'] = (float(final_score_dict['Final Score']) / 10)
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(final_score_dict, score_pth)
        return final_score_dict


class MathVista(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MathVista_MINI': 'https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv'
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
            assert model.working(), ('MathVista evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)
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
                    assert ans[k]['log'] == v['log'] and ans[k]['res'] == v['res']

            data['res'] = [ans[idx]['res'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score = MathVista_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        dump(score, score_pth)
        return score


class LLaVABench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {'LLaVABench': 'https://opencompass.openxlab.space/utils/VLMEval/LLaVABench.tsv'}
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
            model = build_judge(temperature=0.2, system_prompt=system_prompt, **judge_kwargs)
            assert model.working(), ('LLaVABench evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

            prompts = [build_prompt(line) for line in lines]
            tups = [(model, prompt) for prompt in prompts]
            scores = track_progress_rich(LLaVABench_atomeval, tups, nproc=nproc, chunksize=nproc)
            data['gpt4_score'] = [x[0] for x in scores]
            data['score'] = [x[1] for x in scores]
            dump(data, record_file)

        data = load(record_file)
        ret = LLaVABench_score(data).round(1)
        dump(ret, score_file)
        return ret


class MMVet(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MMVet': 'https://opencompass.openxlab.space/utils/VLMEval/MMVet.tsv'
    }
    DATASET_MD5 = {'MMVet': '748aa6d4aa9d4de798306a63718455e3'}

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
            assert model.working(), ('MMVet evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

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
                    assert ans[k]['log'] == v['log'] and ans[k]['score'] == v['score']
            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score, score_fine = MMVet_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        score_fine_pth = storage.replace('.xlsx', '_score_fine.csv')
        dump(score, score_pth)
        dump(score_fine, score_fine_pth)
        return score


class VCRDataset(ImageBaseDataset):
    TYPE = 'VQA'

    URL_PREFIX = 'https://huggingface.co/datasets/vcr-org'

    DATASET_URL = {
        'VCR_EN_EASY_500': f'{URL_PREFIX}/VCR-wiki-en-easy-test-500/resolve/main/VCR-wiki-en-easy-test-500.tsv',
        'VCR_EN_EASY_100': f'{URL_PREFIX}/VCR-wiki-en-easy-test-100/resolve/main/VCR-wiki-en-easy-test-100.tsv',
        'VCR_EN_EASY_ALL': f'{URL_PREFIX}/VCR-wiki-en-easy-test/resolve/main/VCR-wiki-en-easy-test.tsv',
        'VCR_EN_HARD_500': f'{URL_PREFIX}/VCR-wiki-en-hard-test-500/resolve/main/VCR-wiki-en-hard-test-500.tsv',
        'VCR_EN_HARD_100': f'{URL_PREFIX}/VCR-wiki-en-hard-test-100/resolve/main/VCR-wiki-en-hard-test-100.tsv',
        'VCR_EN_HARD_ALL': f'{URL_PREFIX}/VCR-wiki-en-hard-test/resolve/main/VCR-wiki-en-hard-test.tsv',
        'VCR_ZH_EASY_500': f'{URL_PREFIX}/VCR-wiki-zh-easy-test-500/resolve/main/VCR-wiki-zh-easy-test-500.tsv',
        'VCR_ZH_EASY_100': f'{URL_PREFIX}/VCR-wiki-zh-easy-test-100/resolve/main/VCR-wiki-zh-easy-test-100.tsv',
        'VCR_ZH_EASY_ALL': f'{URL_PREFIX}/VCR-wiki-zh-easy-test/resolve/main/VCR-wiki-zh-easy-test.tsv',
        'VCR_ZH_HARD_500': f'{URL_PREFIX}/VCR-wiki-zh-hard-test-500/resolve/main/VCR-wiki-zh-hard-test-500.tsv',
        'VCR_ZH_HARD_100': f'{URL_PREFIX}/VCR-wiki-zh-hard-test-100/resolve/main/VCR-wiki-zh-hard-test-100.tsv',
        'VCR_ZH_HARD_ALL': f'{URL_PREFIX}/VCR-wiki-zh-hard-test/resolve/main/VCR-wiki-zh-hard-test.tsv',
    }

    DATASET_MD5 = {
        'VCR_EN_EASY_500': 'fd9258db52f8685dc710619a0ea0a261',
        'VCR_EN_EASY_100': '9df5d7266683458621ecbe122beb72f0',
        'VCR_EN_EASY_ALL': '8a9b96885f251d1c85f42f84073327f1',
        'VCR_EN_HARD_500': '0a22a85080b6a1f52b1f95e302d43df4',
        'VCR_EN_HARD_100': '1b20f5cbcbeae0b0bec77f7a36143958',
        'VCR_EN_HARD_ALL': '2d8b8b1ee0eba0e0b618fd3aa7d9710e',
        'VCR_ZH_EASY_500': 'beca5fd54176adf44cf94bd9b50cf048',
        'VCR_ZH_EASY_100': '4a86a5678a79844d6d22ab0629c51cd5',
        'VCR_ZH_EASY_ALL': '5050fe7f0027ad2068fd4c7f220edaea',
        'VCR_ZH_HARD_500': '617e3360f75c54455625cb0a8da5c1e7',
        'VCR_ZH_HARD_100': 'b0e38c85f5d5e63894a3b881c372a62b',
        'VCR_ZH_HARD_ALL': '54bbfef448206518b03127ef8b61404c',
    }

    def __init__(self, dataset='VCR_EN_EASY_500', skip_noimg=True):
        super().__init__(dataset, skip_noimg)

        self.language = 'en' if 'EN' in dataset else 'zh'
        self.difficulty = 'easy' if 'EASY' in dataset else 'hard'

    # def build_prompt(self, line):
    #     msgs = super().build_prompt(line)
    #     assert msgs[-1]['type'] == 'text'
    #     if self.language == 'zh':
    #         msgs[-1]['value'] += '图像中被覆盖的文本是什么？请在不输出解释的情况下还原被覆盖的文本。'
    #     else:
    #         msgs[-1]['value'] += ('What is the covered texts in the image? '
    #                               'Please restore the covered texts without outputting the explanations.')
    #     return msgs

    def evaluate(self, eval_file, **judge_kwargs):

        from .utils.vcr import process_match_single_new
        vcr_score_list = {'Exact_Match': [], 'Jaccard': []}
        vcr_score = {'Exact_Match': 0, 'Jaccard': 0}
        logger = get_logger('Evaluation')
        data = load(eval_file)

        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]

        pool = multiprocessing.Pool()
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        results = []

        overall_results = {str(image_id): {} for image_id in range(len(lines))}

        for instance_id, instance in enumerate(lines):
            results.append(
                pool.apply_async(
                    process_match_single_new,
                    args=(
                        str(instance_id),
                        instance['prediction'],
                        instance['answer'],
                        self.language,
                        progress_queue,
                    ),
                )
            )
        pool.close()

        # Display progress bar
        for _ in tqdm(range(len(results))):
            progress_queue.get()

        pool.join()

        # Merging results into overall_result
        for result in results:
            image_id, result_per_id = result.get()
            overall_results[str(image_id)].update(result_per_id[image_id])
            for blank_id_str in result_per_id[image_id].keys():
                vcr_score_list['Exact_Match'].append(
                    result_per_id[image_id][blank_id_str]['exact_match']
                )
                vcr_score_list['Jaccard'].append(
                    result_per_id[image_id][blank_id_str]['jaccard']
                )
            vcr_score['Exact_Match'] = np.mean(vcr_score_list['Exact_Match'])
            vcr_score['Jaccard'] = np.mean(vcr_score_list['Jaccard'])
        results_out = {
            k: v for i in range(len(results)) for k, v in results[i].get()[1].items()
        }
        results_with_metrics = {
            'Exact_Match': vcr_score['Exact_Match'],
            'Jaccard': vcr_score['Jaccard'],
            'Predictions': results_out,
        }
        score_pth = eval_file.replace(
            '.xlsx', f'{self.language}_{self.difficulty}_score.json'
        )
        dump(results_with_metrics, score_pth)
        logger.info(
            f'VCR successfully finished evaluating {eval_file}, results saved in {score_pth}'
        )
        logger.info('Score: ')
        for key, value in vcr_score.items():
            logger.info('{}:{}'.format(key, value))


class CustomVQADataset(ImageBaseDataset):
    TYPE = 'VQA'

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE

                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def evaluate(self, eval_file, **judge_kwargs):
        raise NotImplementedError
