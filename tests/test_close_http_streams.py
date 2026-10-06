import importlib.util
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _restore_modules():
    original = {name: module for name, module in sys.modules.items() if name.startswith(('vlmeval', 'torch'))}
    yield
    for name in list(sys.modules):
        if name.startswith(('vlmeval', 'torch')):
            del sys.modules[name]
    sys.modules.update(original)


def _load_modules(*names):
    root = Path(__file__).resolve().parents[1]
    modules = {}
    for name in ('vlmeval', 'vlmeval.smp', 'vlmeval.api'):
        package = types.ModuleType(name)
        package.__path__ = [str(root / name.replace('.', '/'))]
        modules[name] = package
    misc = types.ModuleType('vlmeval.smp.misc')
    misc.toliststr = lambda value: value if isinstance(value, list) else [value]
    modules[misc.__name__] = misc
    sys.modules.update(modules)
    for name in ('smp.log', 'smp.vlm', 'smp.file', 'api.base', *names):
        full_name = 'vlmeval.' + name
        spec = importlib.util.spec_from_file_location(full_name, root / (full_name.replace('.', '/') + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        modules[full_name] = module
        spec.loader.exec_module(module)
        if name.startswith('smp.'):
            for key, value in vars(module).items():
                if not key.startswith('_'):
                    setattr(modules['vlmeval.smp'], key, value)
    return modules


@pytest.mark.parametrize('name', ['gpt', 'openai_sdk'])
def test_http_stream_is_closed_after_done(tmp_path, name):
    modules = _load_modules('api.openai_sdk', 'api.gpt')
    received = []
    body = (b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
            b'data: [DONE]\n\n' + b': trailing server heartbeat\n' * 1000)

    class Handler(BaseHTTPRequestHandler):

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    response = None
    try:
        api_base = f'http://127.0.0.1:{server.server_port}/chat/completions'
        if name == 'gpt':
            model = modules['vlmeval.api.gpt'].OpenAIWrapper(key='test-key', api_base=api_base, stream=True)
        else:

            class SDKWrapper(modules['vlmeval.api.openai_sdk'].OpenAISDKWrapper):

                def prepare_inputs(self, inputs, system_prompt):
                    return [{'role': 'user', 'content': inputs[0]['value']}]

            model = SDKWrapper(stream=True, verbose=False)
            model.key, model.model, model.api_base, model.timeout = 'test-key', 'mock', api_base, 5
        code, answer, response = model.generate_inner([{'type': 'text', 'value': 'Question'}])
        assert (code, answer) == (0, 'answer')
        assert received[0]['stream'] is True
        assert response.raw.closed
    finally:
        if response is not None:
            response.close()
        server.shutdown()
        server.server_close()
        thread.join()
