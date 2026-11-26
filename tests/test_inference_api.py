import asyncio
import importlib.util
import logging
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


def _dump(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_inference_api():
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']

    smp = types.ModuleType('vlmeval.smp')
    smp.dump = _dump
    smp.get_logger = lambda name: logging.getLogger(name)
    smp.load = _load
    smp.upsert_dataset_status = lambda *args, **kwargs: None

    smp_log = types.ModuleType('vlmeval.smp.log')
    smp_log.setup_subprocess_logger = lambda *args, **kwargs: None

    utils = types.ModuleType('vlmeval.utils')
    utils.__path__ = ['vlmeval/utils']

    modules = {
        'vlmeval': vlmeval,
        'vlmeval.smp': smp,
        'vlmeval.smp.log': smp_log,
        'vlmeval.utils': utils,
    }
    with mock.patch.dict(sys.modules, modules):
        mp_spec = importlib.util.spec_from_file_location(
            'vlmeval.utils.mp_util',
            'vlmeval/utils/mp_util.py',
        )
        mp_util = importlib.util.module_from_spec(mp_spec)
        sys.modules['vlmeval.utils.mp_util'] = mp_util
        mp_spec.loader.exec_module(mp_util)

        spec = importlib.util.spec_from_file_location(
            'vlmeval.inference_api',
            'vlmeval/inference_api.py',
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules['vlmeval.inference_api'] = module
        spec.loader.exec_module(module)
        sys.modules.pop('vlmeval.inference_api', None)
        sys.modules.pop('vlmeval.utils.mp_util', None)
        return module


def _exit_abruptly():
    os._exit(1)


class FakePromptDataset:

    def build_prompt(self, index, video_llm=False):
        return {'index': index, 'video_llm': video_llm}


class FakeEvalDataset:
    dataset_name = 'FakeEval'
    data = [0]

    def evaluate(self, result_file, **judge_kwargs):
        return {'result_file': result_file, 'judge_kwargs': judge_kwargs}


class FakeCrashEvalDataset:
    dataset_name = 'CrashEval'
    data = [0]

    def evaluate(self, result_file, **judge_kwargs):
        os._exit(1)


class TestInferenceApiProcessHelpers(unittest.TestCase):

    def test_async_wait_process_observes_abrupt_exit(self):
        inference_api = _load_inference_api()
        process = mp.Process(target=_exit_abruptly)
        process.start()

        try:
            exitcode = asyncio.run(
                inference_api.async_wait_process(process, poll_interval=0.01)
            )
            self.assertEqual(exitcode, 1)
        finally:
            inference_api.terminate_processes([process])

    def test_async_recv_process_message_raises_when_process_exits(self):
        inference_api = _load_inference_api()
        parent_conn, child_conn = mp.Pipe(duplex=True)
        process = mp.Process(target=_exit_abruptly)
        process.start()
        child_conn.close()

        try:
            with self.assertRaisesRegex(RuntimeError, 'exited unexpectedly'):
                asyncio.run(
                    inference_api.async_recv_process_message(
                        parent_conn,
                        process,
                        'test process',
                        poll_interval=0.01,
                    )
                )
        finally:
            parent_conn.close()
            inference_api.terminate_processes([process])

    def test_prompt_process_builds_prompt_and_exits(self):
        inference_api = _load_inference_api()
        parent_conn, child_conn = mp.Pipe(duplex=True)
        process = mp.Process(
            target=inference_api._prompt_subprocess_target,
            args=(FakePromptDataset(), True, child_conn, parent_conn),
        )
        process.start()
        child_conn.close()

        try:
            parent_conn.send(7)
            reply = asyncio.run(
                inference_api.async_recv_process_message(
                    parent_conn,
                    process,
                    'prompt process',
                    poll_interval=0.01,
                )
            )
            self.assertEqual(reply, {
                'success': True,
                'result': {'index': 7, 'video_llm': True},
                'error': None,
            })

            parent_conn.send(None)
            process.join(timeout=5)
            self.assertEqual(process.exitcode, 0)
        finally:
            parent_conn.close()
            inference_api.terminate_processes([process])

    def test_eval_process_sends_result(self):
        inference_api = _load_inference_api()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            log_file = tmpdir / 'eval.log'
            parent_conn, child_conn = mp.Pipe(duplex=False)
            process = mp.Process(
                target=inference_api._eval_process_entry,
                args=(
                    FakeEvalDataset(),
                    'pred.pkl',
                    {'judge': 'mock'},
                    str(log_file),
                    child_conn,
                    parent_conn,
                ),
            )
            process.start()
            child_conn.close()

            try:
                eval_result = asyncio.run(
                    inference_api.async_recv_process_message(
                        parent_conn,
                        process,
                        'eval process',
                        poll_interval=0.01,
                    )
                )
                asyncio.run(
                    inference_api.async_wait_process(process, poll_interval=0.01)
                )
                self.assertEqual(process.exitcode, 0)
                self.assertEqual(eval_result, {
                    'success': True,
                    'result': {
                        'result_file': 'pred.pkl',
                        'judge_kwargs': {'judge': 'mock'},
                    },
                    'error': None,
                })
            finally:
                parent_conn.close()
                inference_api.terminate_processes([process])

    def test_run_eval_in_subprocess_handles_abrupt_exit(self):
        inference_api = _load_inference_api()

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = inference_api.DatasetConfig(
                dataset_name='CrashEval',
                dataset_obj=FakeCrashEvalDataset(),
                model_name='mock',
                model_obj=None,
                work_dir=tmpdir,
                result_file='pred.pkl',
                judge_kwargs={},
            )
            pipeline = inference_api.APIEvalPipeline([cfg], concurrency=1)

            try:
                log_file = str(Path(tmpdir) / 'eval.log')
                logging.disable(logging.CRITICAL)
                asyncio.run(pipeline._run_eval_in_subprocess('CrashEval', log_file))

                self.assertEqual(cfg.eval_status, inference_api.EvalStatus.Error)
                self.assertEqual(pipeline.eval_processes, {})
            finally:
                logging.disable(logging.NOTSET)
                pipeline._shutdown_executors()


if __name__ == '__main__':
    unittest.main()
