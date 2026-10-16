import functools
import http.server
import importlib.util
import logging
import sys
import threading
import types
from unittest import mock

from PIL import Image


@functools.lru_cache(maxsize=1)
def _load_file_module():
    importlib.import_module('pandas')
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']

    smp = types.ModuleType('vlmeval.smp')
    smp.__path__ = ['vlmeval/smp']

    log = types.ModuleType('vlmeval.smp.log')
    log.get_logger = logging.getLogger

    misc = types.ModuleType('vlmeval.smp.misc')
    misc.toliststr = lambda value: value if isinstance(value, list) else [value]

    vlm = types.ModuleType('vlmeval.smp.vlm')
    vlm.decode_base64_to_image_file = mock.MagicMock()

    validators = types.ModuleType('validators')
    validators.url = lambda value: False

    modules = {
        'validators': validators,
        'vlmeval': vlmeval,
        'vlmeval.smp': smp,
        'vlmeval.smp.log': log,
        'vlmeval.smp.misc': misc,
        'vlmeval.smp.vlm': vlm,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            'vlmeval.smp.file',
            'vlmeval/smp/file.py',
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules['vlmeval.smp.file'] = module
        spec.loader.exec_module(module)
        sys.modules.pop('vlmeval.smp.file', None)
        return module


def test_parse_file_detects_extensionless_image(tmp_path):
    module = _load_file_module()
    image_path = tmp_path / 'sitebench-image'
    Image.new('RGB', (4, 4), color='red').save(image_path, format='PNG')

    assert module.parse_file(str(image_path)) == ('image/png', str(image_path))


def test_parse_file_keeps_unknown_for_non_image(tmp_path):
    module = _load_file_module()
    file_path = tmp_path / 'extensionless-text'
    file_path.write_text('not an image')

    assert module.parse_file(str(file_path)) == ('unknown', str(file_path))


def test_parse_file_keeps_unknown_when_pillow_rejects_image(tmp_path):
    module = _load_file_module()
    image_path = tmp_path / 'extensionless-image'
    image_path.write_bytes(b'image')

    error = Image.DecompressionBombError('image exceeds Pillow safety limit')
    with mock.patch.object(Image, 'open', side_effect=error):
        assert module.parse_file(str(image_path)) == ('unknown', str(image_path))


def test_remote_query_urls_keep_table_format_and_distinct_cached_results(tmp_path, monkeypatch):
    module = _load_file_module()
    (tmp_path / 'files').mkdir()
    monkeypatch.setattr(module, 'LMUDataRoot', lambda: str(tmp_path))

    class Handler(http.server.BaseHTTPRequestHandler):

        def do_GET(self):
            answer = 'first' if self.path.endswith('variant=first') else 'second'
            payload = f'index,prediction\n1,{answer}\n'.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    with http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        root = f'http://127.0.0.1:{server.server_port}/samples.csv'
        first, second = f'{root}?variant=first', f'{root}?variant=second'
        try:
            assert module.load(first)['prediction'].tolist() == ['first']
            assert module.load(second)['prediction'].tolist() == ['second']
        finally:
            server.shutdown()
            thread.join()
    cached = list((tmp_path / 'files').iterdir())
    assert len(cached) == 2
    assert all(path.suffix == '.csv' for path in cached)
    assert module.load(first)['prediction'].tolist() == ['first']
    assert module.load(second)['prediction'].tolist() == ['second']
