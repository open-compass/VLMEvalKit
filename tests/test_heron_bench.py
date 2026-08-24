import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


class FakeImageBaseDataset:

    def dump_image(self, line):
        return [line['image_path']]


class FakeJudge:

    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def working(self):
        return True

    def generate(self, prompt):
        self.prompts.append(prompt)
        return next(self.responses)


def _load_module():
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = ['vlmeval/dataset']
    image_base = types.ModuleType('vlmeval.dataset.image_base')
    image_base.ImageBaseDataset = FakeImageBaseDataset
    dataset_utils = types.ModuleType('vlmeval.dataset.utils')
    dataset_utils.DEBUG_MESSAGE = 'debug'
    dataset_utils.build_judge = mock.Mock()
    llavabench = types.ModuleType('vlmeval.dataset.utils.llavabench')
    llavabench.build_prompt = lambda line: f"BASE:{line['question']}\n"
    smp = types.ModuleType('vlmeval.smp')
    smp.dump = mock.Mock()
    smp.get_intermediate_file_path = mock.Mock()
    smp.load = mock.Mock()
    utils = types.ModuleType('vlmeval.utils')
    utils.track_progress_rich = mock.Mock()
    huggingface_hub = types.ModuleType('huggingface_hub')
    huggingface_hub.snapshot_download = mock.Mock()

    modules = {
        'vlmeval': vlmeval,
        'vlmeval.dataset': dataset,
        'vlmeval.dataset.image_base': image_base,
        'vlmeval.dataset.utils': dataset_utils,
        'vlmeval.dataset.utils.llavabench': llavabench,
        'vlmeval.smp': smp,
        'vlmeval.utils': utils,
        'huggingface_hub': huggingface_hub,
    }
    name = 'vlmeval.dataset.heron_bench'
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(name, 'vlmeval/dataset/heron_bench.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        sys.modules.pop(name, None)
        return module


class TestHeronBench(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_parse_score_accepts_official_formats(self):
        parse = self.module.parse_heron_score
        self.assertEqual(parse('8 9\nexplanation'), [8.0, 9.0])
        self.assertEqual(parse('8.5, 9\nexplanation'), [8.5, 9.0])
        self.assertEqual(parse('8   9'), [8.0, 9.0])

    def test_parse_score_rejects_malformed_or_out_of_range_values(self):
        parse = self.module.parse_heron_score
        self.assertEqual(parse('scores: 8 9'), [-1.0, -1.0])
        self.assertEqual(parse('0 9'), [-1.0, -1.0])
        self.assertEqual(parse('8 11'), [-1.0, -1.0])

    def test_score_uses_unweighted_mean_of_official_categories(self):
        data = pd.DataFrame([
            {'category': 'conv', 'gpt4_score': 10, 'score': 5},
            {'category': 'detail', 'gpt4_score': 5, 'score': 5},
            {'category': 'detail', 'gpt4_score': 5, 'score': 5},
            {'category': 'complex', 'gpt4_score': 4, 'score': 2},
            {'category': 'complex', 'gpt4_score': -1, 'score': -1},
        ])
        result = self.module.heron_bench_score(data)
        by_split = result.set_index('split')

        self.assertAlmostEqual(by_split.loc['conv', 'Relative Score (main)'], 50.0)
        self.assertAlmostEqual(by_split.loc['detail', 'Relative Score (main)'], 100.0)
        self.assertAlmostEqual(by_split.loc['complex', 'Relative Score (main)'], 50.0)
        self.assertAlmostEqual(by_split.loc['overall', 'Relative Score (main)'], 200 / 3)
        self.assertEqual(by_split.loc['overall', 'Valid Samples'], 4)
        self.assertEqual(by_split.loc['overall', 'Parse Errors'], 1)

    def test_load_data_joins_official_questions_answers_and_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'images').mkdir()
            for image_id in range(1, 22):
                (root / 'images' / f'{image_id:03}.jpg').write_bytes(b'image')

            questions = []
            answers = []
            for question_id in range(self.module.HERON_BENCH_SAMPLES):
                category = self.module.HERON_CATEGORIES[question_id % 3]
                questions.append({
                    'question_id': question_id,
                    'image': f'{question_id % 21 + 1:03}.jpg',
                    'category': category,
                    'image_category': 'landscape',
                    'context': f'context {question_id}',
                    'text': f'question {question_id}',
                })
                answers.append({
                    'question_id': question_id,
                    'text': f'answer {question_id}',
                })

            for name, records in (
                ('questions_ja.jsonl', questions),
                ('answers_gpt4.jsonl', answers),
            ):
                with (root / name).open('w', encoding='utf-8') as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')

            with mock.patch.object(self.module, 'snapshot_download', return_value=str(root)):
                benchmark = self.module.HeronBench.__new__(self.module.HeronBench)
                data = benchmark.load_data(self.module.HERON_BENCH_DATASET)

            self.assertEqual(len(data), self.module.HERON_BENCH_SAMPLES)
            self.assertEqual(data.iloc[0]['question'], 'question 0')
            self.assertEqual(data.iloc[0]['gpt4_ans'], 'answer 0')
            self.assertTrue(Path(data.iloc[-1]['image_path']).is_file())

    def test_build_prompt_keeps_japanese_question_unmodified(self):
        benchmark = self.module.HeronBench.__new__(self.module.HeronBench)
        line = {'image_path': '/tmp/example.jpg', 'question': '何が見えますか？'}
        self.assertEqual(benchmark.build_prompt(line), [
            {'type': 'image', 'value': '/tmp/example.jpg'},
            {'type': 'text', 'value': '何が見えますか？'},
        ])

    def test_judge_prompt_includes_official_low_score_instruction(self):
        prompt = self.module.build_heron_judge_prompt({'question': '質問'})
        self.assertIn('BASE:質問', prompt)
        self.assertIn('give it a low score', prompt)

    def test_evaluate_runs_pairwise_judge_and_aggregates_scores(self):
        data = pd.DataFrame([
            {
                'index': 0, 'caption': 'context', 'question': 'conv question',
                'gpt4_ans': 'reference', 'prediction': 'candidate', 'category': 'conv',
            },
            {
                'index': 1, 'caption': 'context', 'question': 'detail question',
                'gpt4_ans': 'reference', 'prediction': 'candidate', 'category': 'detail',
            },
            {
                'index': 2, 'caption': 'context', 'question': 'complex question',
                'gpt4_ans': 'reference', 'prediction': 'candidate', 'category': 'complex',
            },
        ])
        judge = FakeJudge(['10 5\nreason', '5 5\nreason', '4 2\nreason'])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_file = str(root / 'predictions.xlsx')
            Path(eval_file).touch()
            storage = {eval_file: data}

            def intermediate(_, suffix, extension='xlsx'):
                return str(root / f'{suffix}.{extension}')

            def load(path):
                value = storage[str(path)]
                return value.copy() if hasattr(value, 'copy') else value

            def dump(value, path):
                storage[str(path)] = value.copy() if hasattr(value, 'copy') else value
                Path(path).touch()

            def run_jobs(func, jobs, **kwargs):
                return [func(*job) for job in jobs]

            with (
                mock.patch.object(self.module, 'get_intermediate_file_path', side_effect=intermediate),
                mock.patch.object(self.module, 'load', side_effect=load),
                mock.patch.object(self.module, 'dump', side_effect=dump),
                mock.patch.object(self.module, 'build_judge', return_value=judge) as build_judge,
                mock.patch.object(self.module, 'track_progress_rich', side_effect=run_jobs),
            ):
                result = self.module.HeronBench.evaluate(
                    eval_file, model='fake-judge', nproc=2
                )

        overall = result.set_index('split').loc['overall']
        self.assertAlmostEqual(overall['Relative Score (main)'], 66.67, places=2)
        self.assertEqual(len(judge.prompts), 3)
        self.assertTrue(all('give it a low score' in prompt for prompt in judge.prompts))
        build_judge.assert_called_once_with(
            model='fake-judge',
            system_prompt=self.module.HERON_SYSTEM_PROMPT,
            temperature=0,
        )


if __name__ == '__main__':
    unittest.main()
