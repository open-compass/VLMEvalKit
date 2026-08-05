import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from vlmeval.inference import infer_data_job
from vlmeval.smp import status_report
from vlmeval.smp.dataset_alias import resolve_dataset_alias

os.environ.setdefault('LMUData', '/tmp/vlmevalkit-test-lmudata')
os.makedirs(os.environ['LMUData'], exist_ok=True)


class DummyReporter:

    @staticmethod
    def report_infer_err(pred_path):
        return {'failed': 0, 'total': 1 if pred_path else 0}

    @staticmethod
    def report_judge_err(pred_path, total_samples=None, judge_model=None, error_message=None):
        return {'failed': 0, 'total': total_samples or 0}

    @staticmethod
    def report_primary_metric(metrics):
        return {}


class FakeDataset:
    dataset_name = 'LogicalDS'
    MODALITY = 'IMAGE'
    TYPE = 'VQA'

    def __init__(self):
        self.data = pd.DataFrame([{'index': 0, 'question': 'question'}])

    def __len__(self):
        return len(self.data)

    def build_prompt(self, line):
        return [{'type': 'text', 'value': line['question']}]

    def dump_image(self, line):
        return []


class FakeEntryDataset(FakeDataset):
    dataset_name = 'PlainAlias'

    def evaluate(self, eval_file, **judge_kwargs):
        return {'acc': 1.0}


class FakeModel:
    is_api = False

    def __init__(self):
        self.custom_prompt_datasets = []
        self.generate_datasets = []

    def set_dump_image(self, dump_image):
        self.dump_image = dump_image

    def use_custom_prompt(self, dataset):
        self.custom_prompt_datasets.append(dataset)
        return False

    def generate(self, message, dataset=None):
        self.generate_datasets.append(dataset)
        return 'ok'


class TestDatasetAlias(unittest.TestCase):

    def test_resolve_dataset_alias_uses_config_dataset_as_logical_name(self):
        ctx = resolve_dataset_alias(
            'AnyAlias',
            {'AnyAlias': {'class': 'ImageMCQDataset', 'dataset': 'MMBench_DEV_EN_V11'}},
        )

        self.assertEqual(ctx.dataset_alias_name, 'AnyAlias')
        self.assertEqual(ctx.dataset_name, 'MMBench_DEV_EN_V11')

    def test_status_keeps_alias_key_and_logical_dataset_name_for_reporter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            pred_file = run_dir / 'MockModel_AnyAlias.tsv'
            pred_file.write_text('index\tprediction\n0\tok\n', encoding='utf-8')

            status_report.upsert_dataset_status(
                run_dir=run_dir,
                model_name='MockModel',
                dataset_name='AnyAlias',
                logical_dataset_name='LogicalDS',
                prediction_file=pred_file,
                status='done',
            )

            status = status_report.load_run_status(run_dir)
            self.assertIn('AnyAlias', status['datasets'])
            self.assertEqual(status['datasets']['AnyAlias']['logical_dataset_name'], 'LogicalDS')

            resolved_names = []

            def fake_resolve(name, dataset_obj=None):
                resolved_names.append(name)
                return DummyReporter

            with mock.patch.object(
                    status_report, '_resolve_dataset_reporter', side_effect=fake_resolve):
                rows = status_report.collect_run_benchmark_report(run_dir)

            self.assertEqual(resolved_names, ['LogicalDS'])
            self.assertEqual(rows[0]['benchmark'], 'AnyAlias')

    def test_infer_data_job_writes_alias_result_but_uses_logical_dataset_for_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            result_file = tmpdir / 'MockModel_AnyAlias.tsv'
            logical_result_file = tmpdir / 'MockModel_LogicalDS.tsv'
            model = FakeModel()

            infer_data_job(
                model=model,
                work_dir=str(tmpdir),
                model_name='MockModel',
                dataset=FakeDataset(),
                result_file=str(result_file),
                dataset_alias_name='AnyAlias',
            )

            self.assertTrue(result_file.exists())
            self.assertFalse(logical_result_file.exists())
            self.assertEqual(model.custom_prompt_datasets, ['LogicalDS'])
            self.assertEqual(model.generate_datasets, ['LogicalDS'])

    def test_run_local_mode_non_config_data_allows_missing_data_config(self):
        import run as runner

        with tempfile.TemporaryDirectory() as tmpdir:
            args = types.SimpleNamespace(
                config=None,
                data=['PlainAlias'],
                model=['MockModel'],
                data_config={},
                work_dir=tmpdir,
                mode='infer',
                reuse=False,
                reuse_aux='all',
                base_url=None,
                api_nproc=1,
                retry=1,
                verbose=False,
                keep_failed=False,
                use_vllm=False,
                judge_api_nproc=None,
                judge_retry=None,
                judge_timeout=600,
                judge_args=None,
                judge_base_url=None,
                judge_key=None,
                judge=None,
            )

            pred_file = Path(tmpdir) / 'MockModel' / 'TENTRY' / 'MockModel_PlainAlias.tsv'
            build_calls = []

            def fake_build_dataset_from_cli(dataset_name, data_config, dataset_kwargs):
                build_calls.append((dataset_name, data_config, dataset_kwargs))
                return FakeEntryDataset()

            judge_kwargs = {'model': 'mock-judge'}
            reuse_ctx = {'source_eval_id': None, 'prediction_complete': False}

            with mock.patch.object(runner, 'RANK', 0), \
                    mock.patch.object(runner, 'WORLD_SIZE', 1), \
                    mock.patch.object(runner, 'build_eval_id', return_value='TENTRY'), \
                    mock.patch.object(runner, 'githash', return_value='deadbeef'), \
                    mock.patch.object(runner, 'setup_logger'), \
                    mock.patch.object(runner, 'apply_supported_vlm_cli_overrides'), \
                    mock.patch.object(runner, 'upsert_run_status'), \
                    mock.patch.object(runner, 'upsert_dataset_status'), \
                    mock.patch.object(runner, 'get_pred_file_path', return_value=str(pred_file)), \
                    mock.patch.object(runner, 'build_dataset_from_cli',
                                      side_effect=fake_build_dataset_from_cli), \
                    mock.patch.object(runner, 'get_judge_kwargs', return_value=judge_kwargs), \
                    mock.patch.object(runner, 'prepare_reuse_files', return_value=reuse_ctx), \
                    mock.patch.object(runner, 'is_prediction_complete', return_value=False), \
                    mock.patch.object(runner, 'infer_data_job', return_value='MockModel'), \
                    mock.patch.object(runner, 'log_run_benchmark_report'):
                runner.run_local_mode(args)

            self.assertEqual(build_calls, [('PlainAlias', {}, {})])

    def test_run_api_mode_non_config_data_allows_missing_data_config(self):
        import run as runner
        import vlmeval.inference_api as inference_api

        captured_configs = []

        class FakePipeline:

            def __init__(self, dataset_configs, **kwargs):
                captured_configs.extend(dataset_configs)
                self.kwargs = kwargs

            async def run(self):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            args = types.SimpleNamespace(
                data=['PlainAlias'],
                model='MockModel',
                data_config={},
                work_dir=tmpdir,
                mode='infer',
                reuse=False,
                reuse_aux='all',
                base_url=None,
                api_nproc=1,
                monitor_interval=30,
                debug=False,
                retry=1,
                verbose=False,
                keep_failed=False,
                custom_prompt=None,
                judge_api_nproc=None,
                judge_retry=None,
                judge_timeout=600,
                judge_args=None,
                judge_base_url=None,
                judge_key=None,
                judge=None,
            )

            build_calls = []

            def fake_build_dataset_from_cli(dataset_name, data_config, dataset_kwargs):
                build_calls.append((dataset_name, data_config, dataset_kwargs))
                return FakeEntryDataset()

            def fake_pred_file(work_dir, model_name, dataset_name, use_env_format=True):
                return str(Path(work_dir) / f'{model_name}_{dataset_name}.tsv')

            judge_kwargs = {'model': 'mock-judge'}
            reuse_ctx = {'source_eval_id': None, 'prediction_complete': False}

            with mock.patch.object(runner, 'build_eval_id', return_value='TENTRY'), \
                    mock.patch.object(runner, 'githash', return_value='deadbeef'), \
                    mock.patch.object(runner, 'setup_logger'), \
                    mock.patch.object(runner, 'apply_supported_vlm_cli_overrides'), \
                    mock.patch.object(runner, 'supported_VLM', {'MockModel': lambda: object()}), \
                    mock.patch.object(runner, 'upsert_run_status'), \
                    mock.patch.object(runner, 'upsert_dataset_status'), \
                    mock.patch.object(runner, 'get_pred_file_path', side_effect=fake_pred_file), \
                    mock.patch.object(runner, 'build_dataset_from_cli',
                                      side_effect=fake_build_dataset_from_cli), \
                    mock.patch.object(runner, 'get_judge_kwargs', return_value=judge_kwargs), \
                    mock.patch.object(runner, 'prepare_reuse_files', return_value=reuse_ctx), \
                    mock.patch.object(runner, 'log_run_benchmark_report'), \
                    mock.patch.object(inference_api, 'APIEvalPipeline', FakePipeline):
                runner.run_api_mode(args)

            self.assertEqual(build_calls, [('PlainAlias', {}, {})])
            self.assertEqual(len(captured_configs), 1)
            self.assertEqual(captured_configs[0].dataset_name, 'PlainAlias')
            self.assertEqual(captured_configs[0].dataset_alias_name, 'PlainAlias')


if __name__ == '__main__':
    unittest.main()
