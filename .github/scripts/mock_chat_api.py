"""Dataset-aware OpenAI-compatible mock server for CI API evaluations.

Requests select a handler with mock_dataset and mock_stage (infer or judge).
Responses are deterministic fixtures for exercising evaluation, not model scores.
The launcher accepts dataset names, e.g. run_mock_api_tests.sh MMMU_Pro_10c SFE.
"""

import argparse
import json
import re
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Preserve the response used by the existing OCRBench CI baseline.
OCRBENCH_RESPONSE = r"""Calvin Klein
27.0g
Kent B. Mills
21.1
0.0g
TISPPip
(10-k^{2})x^{2}+2kx-2=0
independent ice and cold storage co.
exciting
soul
33779
avenue
invest
music
7696
HIGH FIVE ENTERPRISE
oCwrttsde
312mg
(x-1)+\frac{2}{x-1}=\frac{a^{2}-2a+1}{a-1}+\frac{2}{a-1}
New York
5017
Asbestos
Biobus
0.0388
medium
stop
Classic British Cars 2016 Calendar
Cingular.
1998
carroll
1.8g
Vitebsk Oblast
MRS. K. A. SPARROW
1992
Draw the cap-height
20.4g
1624
NGTGTIE
Stan Uris
Y809
Aerobell
andy law
No
line"""


def choice_response(request):
    return 'A'


def hallusion_response(request):
    return 'yes'


def mmvet_response(request):
    return '0.6'


def ocrbench_response(request):
    return OCRBENCH_RESPONSE


def prompt_text(request):
    texts = []
    for message in request['messages']:
        content = message.get('content', '')
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(part['text'] for part in content if part.get('type') == 'text')
    return '\n'.join(texts)


def hle_response(request):
    if request['mock_stage'] == 'infer':
        return 'A'
    return ('[Begin Correctness]no[End Correctness]\n'
            '[Begin Reason]Mock judgment.[End Reason]')


def chartqapro_response(request):
    prompt = prompt_text(request)
    if 'Your answer should be either true or false' in prompt:
        return 'false'
    if 'Your answer should be one of the options letters only' in prompt:
        return 'a'
    return '0'


def sfe_response(request):
    if request['mock_stage'] == 'judge':
        return '4'
    return 'A' if 'The question is a multiple-choice question.' in prompt_text(request) else '0'


def scidocbench_response(request):
    prompt = prompt_text(request)
    if request['mock_stage'] == 'judge':
        if prompt.lstrip().startswith("You are auditing a model's reasoning"):
            return json.dumps({
                'internal_consistency': 0.5, 'no_hallucination': 0.5,
                'reasoning_note': 'Mock reasoning judgment.',
            })
        return json.dumps({'score': 0.5, 'eval_note': 'Mock answer judgment.'})
    if 'Usage: python solution.py input.png output.png' in prompt:
        return ('{"evidence": "Mock image copy.", "reasoning": "Copy the input image."}\n'
                '```python\nfrom PIL import Image\nimport sys\n'
                'with Image.open(sys.argv[1]) as image:\n'
                '    image.save(sys.argv[2])\n```')
    return '{}'


def omnimat_qa_response(request):
    if request['mock_stage'] == 'infer':
        return 'Mock reasoning.\n<answer>Mock materials science answer.</answer>'
    prompt = prompt_text(request)
    if prompt.lstrip().startswith('You are a strict and meticulous grader'):
        points = prompt.split('Key Scoring Points:\n', 1)[1].split('\n\nGround Truth Answer:', 1)[0]
        count = len(re.findall(r'^\d+\. \[', points, flags=re.MULTILINE))
        return json.dumps({
            'met': [1] * count, 'quality_score': [0.5] * count,
            'reasoning': 'Mock recall judgment.',
        })
    return json.dumps({'tp_string': 'Mock matching fact.', 'fp_string': 'Mock extra fact.'})


def omnimat_cal_response(request):
    def fill_slots(value):
        return [fill_slots(item) for item in value] if isinstance(value, list) else '0'

    prompt = prompt_text(request)
    marker = '\n--- Final Answer Format Constraint ---\nfinal_answer_format: '
    if marker in prompt:
        value, _ = json.JSONDecoder().raw_decode(prompt.rsplit(marker, 1)[1])
        answer = json.dumps(fill_slots(value))
    else:
        answer = '0'
    return f'Mock calculation.\n<answer>{answer}</answer>'


# Handlers receive the full request, including messages and mock_stage, so
# datasets can implement different inference and judge prompt handling here.
DATASET_HANDLERS = {
    'XLRS-Bench-lite': choice_response,
    'MMMU_DEV_VAL': choice_response,
    'MathVista_MINI': choice_response,
    'AI2D_TEST': choice_response,
    'MMStar': choice_response,
    'HallusionBench': hallusion_response,
    'MMVet': mmvet_response,
    'OCRBench': ocrbench_response,
    'hle_verified': hle_response,
    'MMMU_Pro_10c': choice_response,
    'ChartQAPro': chartqapro_response,
    'SciDocBench': scidocbench_response,
    'MicroVQA': choice_response,
    'SFE': sfe_response,
    'OmniMat_QA': omnimat_qa_response,
    'OmniMat_CAL': omnimat_cal_response,
}


def get_answer(request):
    if not isinstance(request, dict):
        raise ValueError('Request body must be a JSON object.')
    dataset = request.get('mock_dataset')
    if not isinstance(dataset, str) or dataset not in DATASET_HANDLERS:
        raise ValueError(f'Unsupported mock_dataset: {dataset!r}')
    if request.get('mock_stage') not in ('infer', 'judge'):
        raise ValueError('mock_stage must be infer or judge.')
    if not isinstance(request.get('messages'), list) or not request['messages']:
        raise ValueError('messages must be a nonempty list.')
    return DATASET_HANDLERS[dataset](request)


class MockAPIServer(ThreadingHTTPServer):
    request_queue_size = 512

    def __init__(self, server_address):
        super().__init__(server_address, MockAPIHandler)
        self.request_counts = Counter()
        self.counts_lock = threading.Lock()

    def record_request(self, dataset, stage):
        with self.counts_lock:
            self.request_counts[(dataset, stage)] += 1

    def health(self):
        with self.counts_lock:
            counts = {}
            for (dataset, stage), count in sorted(self.request_counts.items()):
                counts.setdefault(dataset, {})[stage] = count
        return {'status': 'ok', 'requests': counts}


class MockAPIHandler(BaseHTTPRequestHandler):
    def send_json(self, status, body):
        content = json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path == '/health':
            self.send_json(200, self.server.health())
        else:
            self.send_json(404, {'error': 'Unknown endpoint.'})

    def do_POST(self):
        if self.path != '/v1/chat/completions':
            self.send_json(404, {'error': 'Unknown endpoint.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0:
                raise ValueError('Request body must not be empty.')
            request = json.loads(self.rfile.read(length))
            answer = get_answer(request)
        except (ValueError, UnicodeDecodeError) as exc:
            self.send_json(400, {'error': str(exc)})
            return

        self.server.record_request(request['mock_dataset'], request['mock_stage'])
        self.send_json(200, {
            'id': 'chatcmpl-mock',
            'object': 'chat.completion',
            'created': 0,
            'model': request.get('model', 'mock/mock-api'),
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': answer},
                'finish_reason': 'stop',
            }],
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=24334)
    args = parser.parse_args()
    with MockAPIServer((args.host, args.port)) as server:
        print(f'Mock API listening on http://{args.host}:{server.server_port}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
