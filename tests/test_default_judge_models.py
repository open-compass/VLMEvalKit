import json
import types
import unittest
from unittest import mock

from vlmeval.api import OpenAIWrapper
from vlmeval.dataset.image_vqa import LLaVABench, MMVet
from vlmeval.dataset.mmbench_video import MMBenchVideo
from vlmeval.dataset.SGI_Bench_1_0.deep_research import SGI_Bench_Deep_Research
from vlmeval.dataset.SGI_Bench_1_0.dry_experiment import SGI_Bench_Dry_Experiment
from vlmeval.dataset.SGI_Bench_1_0.experimental_reasoning import SGI_Bench_Experimental_Reasoning

# OpenAI shut these down on 2026-10-23.
RETIRED = {'gpt-4-turbo', 'o4-mini'}


class TestDefaultJudgeModels(unittest.TestCase):

    def test_default_judges_are_not_retired_models(self):
        for cls in (LLaVABench, MMVet, MMBenchVideo, SGI_Bench_Deep_Research, SGI_Bench_Dry_Experiment,
                    SGI_Bench_Experimental_Reasoning):
            self.assertNotIn(cls.DEFAULT_JUDGE_MODEL, RETIRED, cls.__name__)
            self.assertTrue(cls.DEFAULT_JUDGE_MODEL.startswith('gpt-5.6-'), cls.__name__)

    def test_replacement_judge_is_sent_without_temperature_or_max_tokens(self):
        judge = OpenAIWrapper(LLaVABench.DEFAULT_JUDGE_MODEL, key='sk-test', temperature=0.2, max_tokens=128)
        response = types.SimpleNamespace(
            status_code=200, headers={'Content-Type': 'application/json; charset=utf-8'},
            text=json.dumps({'choices': [{'message': {'content': 'ok'}}]}))
        with mock.patch('vlmeval.api.gpt.requests.post', return_value=response) as post:
            judge.generate_inner([dict(type='text', value='hi')])

        payload = json.loads(post.call_args.kwargs['data'])
        self.assertEqual(payload['max_completion_tokens'], 128)
        self.assertNotIn('temperature', payload)
        self.assertNotIn('max_tokens', payload)
