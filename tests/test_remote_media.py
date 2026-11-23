import functools
import importlib.util
import sys
import threading
import types
import wave
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image


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


@pytest.mark.parametrize('filename', ['image.webp', 'image.png', 'audio.wav'])
@pytest.mark.parametrize('query', ['', '?token=example#fragment'])
def test_parse_remote_media_returns_downloaded_file(tmp_path, monkeypatch, filename, query):
    module = _load_modules()['vlmeval.smp.file']
    media = tmp_path / 'media'
    media.mkdir()
    if filename.endswith('.wav'):
        with wave.open(str(media / filename), 'wb') as audio:
            audio.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
            audio.writeframes(b'\x00\x00' * 16)
        expected_mime = 'audio/wav'
    else:
        Image.new('RGB', (12, 8), 'red').save(media / filename)
        expected_mime = 'image/webp' if filename.endswith('webp') else 'image/png'
    cache = tmp_path / 'cache'
    cache.mkdir()
    monkeypatch.setenv('LMUData', str(cache))
    server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(SimpleHTTPRequestHandler, directory=str(media)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}/{filename}' + query
        mime, path = module.parse_file(url)
        assert mime == expected_mime
        assert Path(path).is_file()
        assert cache in Path(path).parents
        assert Path(path).read_bytes() == (media / filename).read_bytes()
        base = _load_modules()['vlmeval.api.base']
        wrapper = base.BaseAPI()
        wrapper.allowed_types = [*wrapper.allowed_types, 'audio']
        assert wrapper.preproc_content([url])[0]['type'] == expected_mime.split('/')[0]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
