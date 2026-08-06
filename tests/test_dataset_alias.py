import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from vlmeval.inference import infer_data_job
from vlmeval.smp import dump, get_composite_child_eval_file, load, status_report
from vlmeval.smp.dataset_alias import (DatasetSpec, get_predefined_dataset_spec,
                                       resolve_dataset_alias, resolve_dataset_spec)

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


class RecordingChildEvaluator:

    def __init__(self, result):
        self.result = result
        self.calls = []
        self.frames = []

    def evaluate(self, eval_file, **judge_kwargs):
        self.calls.append(eval_file)
        self.frames.append(load(eval_file))
        return self.result


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
        self.assertEqual(ctx.dataset_class_name, 'ImageMCQDataset')

    def test_resolve_predefined_shortcut(self):
        spec = resolve_dataset_spec('MMBench_Video_8frame_nopack')

        self.assertEqual(spec.dataset_alias_name, 'MMBench_Video_8frame_nopack')
        self.assertEqual(spec.dataset_name, 'MMBench-Video')
        self.assertEqual(spec.dataset_class_name, 'MMBenchVideo')
        self.assertEqual(spec.build_config['class'], 'MMBenchVideo')
        self.assertEqual(spec.build_config['dataset'], 'MMBench-Video')
        self.assertEqual(spec.build_config['nframe'], 8)
        self.assertFalse(spec.build_config['pack'])

    def test_resolve_preset_alias(self):
        spec = resolve_dataset_spec(
            'AliasVideo',
            {'AliasVideo': {'preset': 'MMBench_Video_8frame_nopack', 'nframe': 16}},
        )

        self.assertEqual(spec.dataset_alias_name, 'AliasVideo')
        self.assertEqual(spec.dataset_name, 'MMBench-Video')
        self.assertEqual(spec.dataset_class_name, 'MMBenchVideo')
        self.assertEqual(spec.build_config['nframe'], 16)
        self.assertFalse(spec.build_config['pack'])

    def test_empty_dict_config_is_not_predefined_shortcut(self):
        with self.assertRaisesRegex(ValueError, 'Empty dataset config'):
            resolve_dataset_spec(
                'MMBench_Video_8frame_nopack',
                {'MMBench_Video_8frame_nopack': {}},
            )

    def test_explicit_config_requires_non_empty_class_and_dataset(self):
        bad_configs = [
            {'dataset': 'Video-MME'},
            {'class': 'VideoMME'},
            {'class': '', 'dataset': 'Video-MME'},
            {'class': '   ', 'dataset': 'Video-MME'},
            {'class': 'VideoMME', 'dataset': ''},
            {'class': 'VideoMME', 'dataset': '   '},
            {'class': None, 'dataset': 'Video-MME'},
            {'class': 'VideoMME', 'dataset': None},
        ]
        for cfg in bad_configs:
            with self.subTest(cfg=cfg):
                with self.assertRaises(ValueError):
                    resolve_dataset_spec('AliasVideo', {'AliasVideo': cfg})

    def test_get_predefined_dataset_spec_returns_deepcopy(self):
        import vlmeval.dataset.video_dataset_config as video_config

        video_config.PREDEFINED_DATASET_SPECS['NestedSpecForTest'] = DatasetSpec(
            dataset_alias_name='NestedSpecForTest',
            dataset_name='NestedDS',
            dataset_class_name='NestedClass',
            build_config={
                'class': 'NestedClass',
                'dataset': 'NestedDS',
                'nested': {'values': [1]},
            },
            source='predefined_shortcut',
        )
        try:
            first = get_predefined_dataset_spec('NestedSpecForTest')
            second = get_predefined_dataset_spec('NestedSpecForTest')
            first.build_config['nested']['values'].append(2)

            self.assertEqual(second.build_config['nested']['values'], [1])
            registry_values = video_config.PREDEFINED_DATASET_SPECS[
                'NestedSpecForTest'
            ].build_config['nested']['values']
            self.assertEqual(
                registry_values,
                [1],
            )
        finally:
            video_config.PREDEFINED_DATASET_SPECS.pop('NestedSpecForTest', None)

    def test_status_keeps_alias_key_and_dataset_class_name_for_reporter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            pred_file = run_dir / 'MockModel_AnyAlias.tsv'
            pred_file.write_text('index\tprediction\n0\tok\n', encoding='utf-8')

            status_report.upsert_dataset_status(
                run_dir=run_dir,
                model_name='MockModel',
                dataset_name='AnyAlias',
                resolved_dataset_name='LogicalDS',
                dataset_alias_name='AnyAlias',
                dataset_class_name='DummyReporter',
                prediction_file=pred_file,
                status='done',
            )

            status = status_report.load_run_status(run_dir)
            self.assertIn('AnyAlias', status['datasets'])
            self.assertEqual(status['datasets']['AnyAlias']['dataset_name'], 'LogicalDS')
            self.assertEqual(status['datasets']['AnyAlias']['dataset_alias_name'], 'AnyAlias')
            self.assertEqual(status['datasets']['AnyAlias']['dataset_class_name'], 'DummyReporter')
            self.assertNotIn('logical_dataset_name', status['datasets']['AnyAlias'])

            resolved_args = []

            def fake_resolve(
                name,
                dataset_class_name=None,
                dataset_obj=None,
                dataset_alias_name=None,
            ):
                resolved_args.append((name, dataset_class_name, dataset_alias_name))
                return DummyReporter

            with mock.patch.object(
                    status_report, '_resolve_dataset_reporter', side_effect=fake_resolve):
                rows = status_report.collect_run_benchmark_report(run_dir)

            self.assertEqual(resolved_args, [('LogicalDS', 'DummyReporter', 'AnyAlias')])
            self.assertEqual(rows[0]['benchmark'], 'AnyAlias')

    def test_status_report_resolves_reporter_from_dataset_class_name(self):
        reporter = status_report._resolve_dataset_reporter(
            'CG-AV-Counting',
            dataset_class_name='CGAVCounting',
            dataset_alias_name='AnyAlias',
        )

        self.assertEqual(reporter.__name__, 'CGAVCounting')

    def test_predefined_video_manifest_matches_registry(self):
        import json

        import vlmeval.dataset.video_dataset_config as video_config

        manifest_path = (
            Path(__file__).parent / 'fixtures' / 'predefined_video_shortcuts_manifest.json'
        )
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        registry = video_config.PREDEFINED_DATASET_SPECS

        self.assertEqual(set(registry), set(manifest))
        for key, expected in manifest.items():
            with self.subTest(key=key):
                spec = registry[key]
                self.assertEqual(spec.dataset_alias_name, expected['dataset_alias_name'])
                self.assertEqual(spec.dataset_name, expected['dataset_name'])
                self.assertEqual(spec.dataset_class_name, expected['dataset_class_name'])
                self.assertEqual(spec.build_config, expected['build_config'])
                self.assertEqual(spec.dataset_alias_name, key)
                self.assertEqual(spec.dataset_name, spec.build_config['dataset'])
                self.assertEqual(spec.dataset_class_name, spec.build_config['class'])
                self.assertTrue(spec.build_config['class'].strip())
                self.assertTrue(spec.build_config['dataset'].strip())

    def test_video_build_config_validation_uses_class_defaults_and_rejects_none(self):
        from vlmeval.dataset import DREAM, MMBenchVideo, MVUEval, VideoMMEv2

        with self.assertRaisesRegex(ValueError, 'fps and nframe'):
            VideoMMEv2.validate_build_config({'dataset': 'Video-MME-v2', 'fps': 1.0})

        MMBenchVideo.validate_build_config({
            'dataset': 'MMBench-Video',
            'nframe': 8,
            'fps': 0,
        })

        VideoMMEv2.validate_build_config({
            'dataset': 'Video-MME-v2',
            'nframe': 0,
            'fps': 1.0,
        })

        for dataset_cls, dataset_name in [
            (DREAM, 'DREAM-1K'),
            (MVUEval, 'MVU-Eval'),
        ]:
            with self.subTest(dataset_name=dataset_name):
                dataset_cls.validate_build_config({'dataset': dataset_name})
                dataset_cls.validate_build_config({
                    'dataset': dataset_name,
                    'nframe': 0,
                    'fps': -1,
                })
                with self.assertRaisesRegex(ValueError, 'fps and nframe'):
                    dataset_cls.validate_build_config({
                        'dataset': dataset_name,
                        'fps': 0,
                    })

        with self.assertRaisesRegex(ValueError, 'fps should not be None'):
            MMBenchVideo.validate_build_config({
                'dataset': 'MMBench-Video',
                'nframe': 8,
                'fps': None,
            })
        with self.assertRaisesRegex(ValueError, 'nframe should not be None'):
            MMBenchVideo.validate_build_config({
                'dataset': 'MMBench-Video',
                'nframe': None,
                'fps': 1.0,
            })

    def test_sitebench_image_strict_config_uses_explicit_init_signature(self):
        import run as runner
        from vlmeval.dataset import SiteBenchImage

        fake_data = pd.DataFrame([{
            'index': 0,
            'question': 'question',
            'options': '["A", "B"]',
            'image_path': '/tmp/sitebench.jpg',
        }])

        with mock.patch.object(SiteBenchImage, 'load_data', return_value=fake_data):
            dataset = runner.build_dataset_from_config_dict(
                {
                    'class': 'SiteBenchImage',
                    'dataset': 'SiteBenchImage',
                    'skip_noimg': False,
                },
                display_name='SiteBenchImageAlias',
            )

        self.assertIsInstance(dataset, SiteBenchImage)
        self.assertEqual(dataset.dataset_name, 'SiteBenchImage')
        self.assertFalse(dataset.skip_noimg)
        self.assertEqual(dataset.repo_id, 'franky-veteran/SITE-Bench')

        with self.assertRaisesRegex(ValueError, 'Unsupported parameter'):
            runner.build_dataset_from_config_dict(
                {
                    'class': 'SiteBenchImage',
                    'dataset': 'SiteBenchImage',
                    'unknown': 1,
                },
                display_name='SiteBenchImageAlias',
            )

    def test_videommev2_1fps_presets_clear_default_nframe(self):
        from vlmeval.dataset import VideoMMEv2

        shortcut_names = [
            'Video-MME-v2_1fps',
            'Video-MME-v2_1fps_subs',
            'Video-MME-v2_1fps_subs_interleave',
            'Video-MME-v2_1fps_resize',
        ]
        for name in shortcut_names:
            with self.subTest(name=name):
                spec = resolve_dataset_spec(name)
                self.assertEqual(spec.build_config['nframe'], 0)
                self.assertEqual(spec.build_config['fps'], 1.0)
                VideoMMEv2.validate_build_config(spec.build_config)

    def test_video_sampling_branches_treat_fps_zero_as_unset(self):
        repo_root = Path(__file__).parents[1]
        dataset_dir = repo_root / 'vlmeval' / 'dataset'
        offenders = []
        legacy_patterns = [
            'self.nframe > 0 and self.fps < 0',
            'num_frames > 0 and fps < 0',
        ]
        for path in dataset_dir.rglob('*.py'):
            if path.relative_to(repo_root).as_posix() == 'vlmeval/dataset/utils/cgbench.py':
                continue
            text = path.read_text(encoding='utf-8')
            for pattern in legacy_patterns:
                if pattern in text:
                    offenders.append(f'{path.relative_to(repo_root)}: {pattern}')

        mvbench_text = (dataset_dir / 'mvbench.py').read_text(encoding='utf-8')
        if 'if self.fps < 0:' in mvbench_text:
            offenders.append('vlmeval/dataset/mvbench.py')

        self.assertEqual(offenders, [])

    def test_cgbench_local_fps_zero_sampling_uses_nframe_branch(self):
        import numpy as np
        from PIL import Image

        from vlmeval.dataset.cgbench import CGBench_MCQ_Grounding_Mini

        class FakeFrame:

            def asnumpy(self):
                return np.zeros((2, 2, 3), dtype=np.uint8)

        class FakeVideoReader:

            def __init__(self, path):
                self.path = path

            def __len__(self):
                return 9

            def __getitem__(self, idx):
                return FakeFrame()

            def get_avg_fps(self):
                return 30

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset = CGBench_MCQ_Grounding_Mini.__new__(CGBench_MCQ_Grounding_Mini)
            dataset.data_root = str(tmpdir)
            dataset.frame_root = str(tmpdir / 'frames')
            dataset.frame_tmpl = 'frame-{}-of-{}.jpg'
            dataset.frame_tmpl_fps = 'frame-{}-of-{}-{}fps.jpg'
            dataset.nframe = 2

            frame_dir = tmpdir / 'frames' / 'sample'
            frame_dir.mkdir(parents=True)
            for idx in range(1, 3):
                Image.new('RGB', (2, 2)).save(frame_dir / f'frame-{idx}-of-2.jpg')

            fake_decord = types.SimpleNamespace(VideoReader=FakeVideoReader)
            with mock.patch.dict('sys.modules', {'decord': fake_decord}):
                paths, indices, vid_fps = dataset.save_video_frames(
                    'video.mp4',
                    uid='sample',
                    num_frames=2,
                    fps=0,
                )

        self.assertEqual(indices, [3, 6])
        self.assertEqual(vid_fps, 30)
        self.assertEqual(len(paths), 2)

    def test_supported_video_datasets_symbol_removed(self):
        import vlmeval.dataset.video_dataset_config as video_config

        self.assertFalse(hasattr(video_config, 'supported_video_datasets'))

    def test_build_dataset_does_not_parse_predefined_shortcut(self):
        from vlmeval.dataset import build_dataset

        self.assertIsNone(build_dataset('MMBench_Video_8frame_nopack'))

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

    def test_composite_child_eval_file_uses_sub_marker_without_parent_replace(self):
        direct = get_composite_child_eval_file('/tmp/MockModel_SIUO.tsv', 'SIUO_GEN')
        alias = get_composite_child_eval_file('/tmp/MockModel_OutputAlias.tsv', 'SIUO_GEN')
        alias_with_parent = get_composite_child_eval_file(
            '/tmp/MockModel_CustomSIUOAlias.tsv',
            'SIUO_GEN',
        )

        self.assertEqual(direct, '/tmp/_MockModel_SIUO_SUB_SIUO_GEN.tsv')
        self.assertEqual(alias, '/tmp/_MockModel_OutputAlias_SUB_SIUO_GEN.tsv')
        self.assertEqual(alias_with_parent, '/tmp/_MockModel_CustomSIUOAlias_SUB_SIUO_GEN.tsv')

    def test_siuo_composite_evaluate_uses_alias_safe_child_files(self):
        from vlmeval.dataset.siuo import SIUODataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            eval_file = tmpdir / 'MockModel_OutputAlias.tsv'
            dump(
                pd.DataFrame([
                    {
                        'index': 0,
                        'original_index': 11,
                        'SUB_DATASET': 'SIUO_GEN',
                        'prediction': 'gen',
                    },
                    {
                        'index': 1,
                        'original_index': 22,
                        'SUB_DATASET': 'SIUO_MCQ',
                        'prediction': 'mcq',
                    },
                ]),
                str(eval_file),
            )

            dataset = SIUODataset.__new__(SIUODataset)
            dataset.dataset_name = 'SIUO'
            dataset.dataset_map = {
                'SIUO_GEN': RecordingChildEvaluator({'overall_avg_combined': 40}),
                'SIUO_MCQ': RecordingChildEvaluator({'Overall': 80}),
            }

            score = dataset.evaluate(str(eval_file))
            gen_file = tmpdir / '_MockModel_OutputAlias_SUB_SIUO_GEN.tsv'
            mcq_file = tmpdir / '_MockModel_OutputAlias_SUB_SIUO_MCQ.tsv'

            self.assertTrue(eval_file.exists())
            self.assertTrue(gen_file.exists())
            self.assertTrue(mcq_file.exists())
            self.assertEqual(dataset.dataset_map['SIUO_GEN'].calls, [str(gen_file)])
            self.assertEqual(dataset.dataset_map['SIUO_MCQ'].calls, [str(mcq_file)])
            self.assertEqual(list(dataset.dataset_map['SIUO_GEN'].frames[0]['index']), [11])
            self.assertEqual(list(dataset.dataset_map['SIUO_MCQ'].frames[0]['index']), [22])
            self.assertNotIn('SUB_DATASET', dataset.dataset_map['SIUO_GEN'].frames[0])
            self.assertNotIn('original_index', dataset.dataset_map['SIUO_MCQ'].frames[0])
            self.assertEqual(float(score['SIUO'].iloc[0]), 60.0)

    def test_concat_dataset_evaluate_uses_alias_safe_child_files(self):
        from vlmeval.dataset import ConcatDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            eval_file = tmpdir / 'MockModel_ImageAlias.tsv'
            dump(
                pd.DataFrame([
                    {
                        'index': 0,
                        'original_index': 101,
                        'SUB_DATASET': 'ChildImageA',
                        'prediction': 'a',
                    },
                    {
                        'index': 1,
                        'original_index': 202,
                        'SUB_DATASET': 'ChildImageB',
                        'prediction': 'b',
                    },
                ]),
                str(eval_file),
            )

            dataset = ConcatDataset.__new__(ConcatDataset)
            dataset.dataset_name = 'ImageParent'
            dataset.datasets = ['ChildImageA', 'ChildImageB']
            dataset.dataset_map = {
                'ChildImageA': RecordingChildEvaluator({'acc': 1}),
                'ChildImageB': RecordingChildEvaluator({'acc': 2}),
            }

            result = dataset.evaluate(str(eval_file))
            child_a_file = tmpdir / '_MockModel_ImageAlias_SUB_ChildImageA.tsv'
            child_b_file = tmpdir / '_MockModel_ImageAlias_SUB_ChildImageB.tsv'

            self.assertEqual(result, {'ChildImageA:acc': 1, 'ChildImageB:acc': 2})
            self.assertEqual(dataset.dataset_map['ChildImageA'].calls, [str(child_a_file)])
            self.assertEqual(dataset.dataset_map['ChildImageB'].calls, [str(child_b_file)])
            self.assertEqual(list(dataset.dataset_map['ChildImageA'].frames[0]['index']), [101])
            self.assertEqual(list(dataset.dataset_map['ChildImageB'].frames[0]['index']), [202])
            self.assertNotIn('SUB_DATASET', dataset.dataset_map['ChildImageA'].frames[0])
            self.assertNotIn('original_index', dataset.dataset_map['ChildImageB'].frames[0])

    def test_concat_video_dataset_evaluate_uses_alias_safe_child_files(self):
        from vlmeval.dataset.video_concat_dataset import ConcatVideoDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            eval_file = tmpdir / 'MockModel_VideoAlias.tsv'
            dump(
                pd.DataFrame([
                    {
                        'index': 0,
                        'original_index': 7,
                        'SUB_DATASET': 'ChildVideoA',
                        'prediction': 'a',
                    },
                    {
                        'index': 1,
                        'original_index': 8,
                        'SUB_DATASET': 'ChildVideoB',
                        'prediction': 'b',
                    },
                ]),
                str(eval_file),
            )

            dataset = ConcatVideoDataset.__new__(ConcatVideoDataset)
            dataset.dataset_name = 'VideoParent'
            dataset.datasets = ['ChildVideoA', 'ChildVideoB']
            dataset.dataset_map = {
                'ChildVideoA': RecordingChildEvaluator({'ChildVideoA': {'success': 1, 'overall': 2}}),
                'ChildVideoB': RecordingChildEvaluator({'ChildVideoB': {'success': 3, 'overall': 4}}),
            }

            result = dataset.evaluate(str(eval_file))
            child_a_file = tmpdir / '_MockModel_VideoAlias_SUB_ChildVideoA.tsv'
            child_b_file = tmpdir / '_MockModel_VideoAlias_SUB_ChildVideoB.tsv'

            self.assertEqual(dataset.dataset_map['ChildVideoA'].calls, [str(child_a_file)])
            self.assertEqual(dataset.dataset_map['ChildVideoB'].calls, [str(child_b_file)])
            self.assertEqual(list(dataset.dataset_map['ChildVideoA'].frames[0]['index']), [7])
            self.assertEqual(list(dataset.dataset_map['ChildVideoB'].frames[0]['index']), [8])
            self.assertNotIn('SUB_DATASET', dataset.dataset_map['ChildVideoA'].frames[0])
            self.assertNotIn('original_index', dataset.dataset_map['ChildVideoB'].frames[0])
            self.assertEqual(float(result.loc['ChildVideoA', 'acc']), 50.0)
            self.assertEqual(float(result.loc['ChildVideoB', 'acc']), 75.0)

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

            def fake_build_dataset_from_spec(spec, extra_kwargs=None):
                build_calls.append((spec.dataset_alias_name, spec.dataset_name, extra_kwargs))
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
                    mock.patch.object(runner, 'build_dataset_from_spec',
                                      side_effect=fake_build_dataset_from_spec), \
                    mock.patch.object(runner, 'get_judge_kwargs', return_value=judge_kwargs), \
                    mock.patch.object(runner, 'prepare_reuse_files', return_value=reuse_ctx), \
                    mock.patch.object(runner, 'is_prediction_complete', return_value=False), \
                    mock.patch.object(runner, 'infer_data_job', return_value='MockModel'), \
                    mock.patch.object(runner, 'log_run_benchmark_report'):
                runner.run_local_mode(args)

            self.assertEqual(build_calls, [('PlainAlias', 'PlainAlias', {})])

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

            def fake_build_dataset_from_spec(spec, extra_kwargs=None):
                build_calls.append((spec.dataset_alias_name, spec.dataset_name, extra_kwargs))
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
                    mock.patch.object(runner, 'build_dataset_from_spec',
                                      side_effect=fake_build_dataset_from_spec), \
                    mock.patch.object(runner, 'get_judge_kwargs', return_value=judge_kwargs), \
                    mock.patch.object(runner, 'prepare_reuse_files', return_value=reuse_ctx), \
                    mock.patch.object(runner, 'log_run_benchmark_report'), \
                    mock.patch.object(inference_api, 'APIEvalPipeline', FakePipeline):
                runner.run_api_mode(args)

            self.assertEqual(build_calls, [('PlainAlias', 'PlainAlias', {})])
            self.assertEqual(len(captured_configs), 1)
            self.assertEqual(captured_configs[0].dataset_name, 'PlainAlias')
            self.assertEqual(captured_configs[0].dataset_alias_name, 'PlainAlias')
            self.assertIsNone(captured_configs[0].dataset_class_name)


if __name__ == '__main__':
    unittest.main()
