import hashlib
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'mock_chat_api', Path(__file__).parents[1] / '.github/scripts/mock_chat_api.py',
)
mock_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mock_api)


@pytest.fixture
def server():
    with mock_api.MockAPIServer(('127.0.0.1', 0)) as instance:
        thread = threading.Thread(target=instance.serve_forever, daemon=True)
        thread.start()
        try:
            yield f'http://127.0.0.1:{instance.server_port}'
        finally:
            instance.shutdown()
            thread.join()


def request_json(server, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        server + path, data=data, headers={'Content-Type': 'application/json'},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=5) as response:
        return json.load(response)


def completion(server, dataset, stage, content='hello'):
    return request_json(server, '/v1/chat/completions', {
        'model': 'mock-model',
        'mock_dataset': dataset,
        'mock_stage': stage,
        'messages': [{'role': 'user', 'content': content}],
    })


class HTTPJudge:
    def __init__(self, server, dataset):
        self.server = server
        self.dataset = dataset
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        return completion(self.server, self.dataset, 'judge', prompt)['choices'][0]['message']['content']


def test_concurrent_dataset_and_stage_routing_preserves_baseline_responses(server):
    expected = {
        'XLRS-Bench-lite': 'A',
        'MMMU_DEV_VAL': 'A',
        'MathVista_MINI': 'A',
        'AI2D_TEST': 'A',
        'MMStar': 'A',
        'HallusionBench': 'yes',
        'MMVet': '0.6',
    }
    cases = [(dataset, stage) for dataset in expected for stage in ('infer', 'judge')]
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda case: completion(server, *case), cases))
    for (dataset, stage), response in zip(cases, responses):
        assert response['model'] == 'mock-model'
        assert response['choices'][0]['message'] == {
            'role': 'assistant', 'content': expected[dataset],
        }
        assert response['choices'][0]['finish_reason'] == 'stop'
    health = request_json(server, '/health')
    assert health['status'] == 'ok'
    assert health['requests'] == {dataset: {'infer': 1, 'judge': 1} for dataset in expected}


def test_ocrbench_response_matches_previous_ci_prediction(server):
    content = completion(server, 'OCRBench', 'infer')['choices'][0]['message']['content']
    # SHA256 of the unique OCRBench prediction in CI run 41567725070.
    assert hashlib.sha256(content.encode()).hexdigest() == (
        '56f5e36218d0062f67ecda00364056d059fa38c1d5de33fd9384ceea978a8450'
    )


@pytest.mark.parametrize('dataset', ['MMMU_Pro_10c', 'MicroVQA'])
def test_new_mcq_inference_and_judge_extraction(server, dataset):
    from vlmeval.dataset.utils.multiple_choice import extract_answer_from_item

    assert completion(server, dataset, 'infer')['choices'][0]['message']['content'] == 'A'
    judge = HTTPJudge(server, dataset)
    result = extract_answer_from_item(judge, {
        'question': 'Choose a shape.', 'A': 'circle', 'B': 'square', 'prediction': 'uncertain',
    }, dataset_name=dataset)
    assert result['opt'] == 'A'
    assert judge.calls == 1


def test_hle_and_sfe_judge_formats_with_actual_evaluators(server):
    from vlmeval.dataset.image_shortqa import ShortQA_auxeval
    from vlmeval.dataset.sfebench import SFE_auxeval

    item = {'question': 'What is the value?', 'answer': '42', 'prediction': 'A'}
    result = ShortQA_auxeval(HTTPJudge(server, 'hle_verified'), item)
    assert result == {'hit': 0, 'log': 'Mock judgment.'}
    result = SFE_auxeval(HTTPJudge(server, 'SFE'), item)
    assert result == {'score': 4, 'log': 'Success to Judge'}


@pytest.mark.parametrize(('question_type', 'expected'), [
    ('Factoid', '0'), ('Conversational', '0'), ('Hypothetical', '0'),
    ('Multi Choice', 'a'), ('Fact Checking', 'false'),
])
def test_chartqapro_prompt_types_with_multimodal_content(server, question_type, expected):
    from vlmeval.dataset.utils.chartqapro import prompt_context

    prompt = prompt_context(['Example question?'], ['0'], question_type, 'Direct')
    content = [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,unused'}},
        {'type': 'text', 'text': 'Chart context.'},
        {'type': 'text', 'text': prompt},
    ]
    assert completion(server, 'ChartQAPro', 'infer', content)['choices'][0]['message']['content'] == expected


@pytest.mark.parametrize(('prompt', 'expected'), [
    ('The question is a multiple-choice question. Select an option.', 'A'),
    ('The question is an open-ended question.', '0'),
    ('The question is an exact-match question.', '0'),
])
def test_sfe_inference_types(server, prompt, expected):
    assert completion(server, 'SFE', 'infer', prompt)['choices'][0]['message']['content'] == expected


def test_scidocbench_answer_and_reasoning_judges(server):
    from vlmeval.dataset.scidocbench import eval_judge, eval_reasoning

    judge = HTTPJudge(server, 'SciDocBench')
    # An answer rubric can mention reasoning keys without selecting the reasoning route.
    score, note = eval_judge(judge, '{}', '{}', 'Explain internal_consistency and no_hallucination.')
    assert (score, note) == (0.5, 'Mock answer judgment.')
    score, note = eval_reasoning(judge, '{}', 'Explain the figure.')
    assert score == 0.5
    assert 'Mock reasoning judgment.' in note
    assert judge.calls == 2


def test_scidocbench_code_inference_executes_through_evaluator(server, tmp_path):
    from PIL import Image

    from vlmeval.dataset.scidocbench import eval_exec_match

    prompt = 'Required interface:\nUsage: python solution.py input.png output.png'
    prediction = completion(server, 'SciDocBench', 'infer', prompt)['choices'][0]['message']['content']
    image_path = tmp_path / 'input.png'
    Image.new('RGB', (3, 3), 'red').save(image_path)
    reference = ('from PIL import Image\nimport sys\n'
                 'Image.open(sys.argv[1]).save(sys.argv[2])\n')
    assert eval_exec_match(prediction, {
        'input_path': str(image_path), 'reference_script': reference,
    }) == (1.0, 'Pixel-perfect match')


@pytest.mark.parametrize('point_count', [0, 1, 4])
def test_omnimat_qa_precision_and_recall_with_variable_key_points(server, point_count):
    from vlmeval.dataset.omnimat import omnimat_qa_precision, omnimat_qa_recall

    judge = HTTPJudge(server, 'OmniMat_QA')
    item = {
        'gt_answer': 'Example reference.', 'llm_answer': 'Example prediction.',
        'key_points': [{'id': f'point_{i}', 'description': 'A fact.\nWith extra detail.'}
                       for i in range(point_count)],
    }
    precision = omnimat_qa_precision(judge, item)
    recall = omnimat_qa_recall(judge, item)
    assert 'error' not in precision and 'error' not in recall
    assert precision['score'] == 0.5
    assert recall['weighted_score'] == (0.5 if point_count else 0.0)
    assert recall['eval_details']['met'] == [1] * point_count
    assert recall['eval_details']['quality_score'] == [0.5] * point_count


@pytest.mark.parametrize(('structure', 'expected'), [
    ([''], ['0']), (['', ''], ['0', '0']), ([['', ''], ['']], [['0', '0'], ['0']]),
])
def test_omnimat_cal_preserves_requested_answer_structure(server, structure, expected):
    from vlmeval.dataset.omnimat import extract_answer_text, flatten_answer, normalize_prediction

    prompt = ('Calculate.\n\n--- Final Answer Format Constraint ---\n'
              f'final_answer_format: {json.dumps(structure)}\n'
              'Inside <answer>, output only the filled answer structure.')
    prediction = completion(server, 'OmniMat_CAL', 'infer', prompt)['choices'][0]['message']['content']
    answer = json.loads(extract_answer_text(prediction))
    assert answer == expected
    assert normalize_prediction(prediction) == ['0'] * len(flatten_answer(structure))


@pytest.mark.parametrize('updates', [
    {'mock_dataset': 'unknown'},
    {'mock_dataset': None},
    {'mock_stage': 'unknown'},
    {'mock_stage': None},
    {'messages': []},
])
def test_invalid_routing_is_rejected(server, updates):
    payload = {
        'mock_dataset': 'MMVet',
        'mock_stage': 'judge',
        'messages': [{'role': 'user', 'content': 'hello'}],
        **updates,
    }
    with pytest.raises(urllib.error.HTTPError) as error:
        request_json(server, '/v1/chat/completions', payload)
    assert error.value.code == 400
    assert json.load(error.value)['error']
    assert request_json(server, '/health')['requests'] == {}
