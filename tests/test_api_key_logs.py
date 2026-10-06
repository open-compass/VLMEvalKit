import importlib.util
import logging
import sys
import types
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


def _load_provider_modules(*names):
    modules = _load_modules('api.openai_sdk', 'api.gpt')
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.DATASET_TYPE = lambda name: None
    dataset.img_root_map = lambda name: name
    sys.modules['vlmeval.dataset'] = dataset
    smp = modules['vlmeval.smp']
    smp.cn_string = lambda text: False
    smp.listinstr = lambda choices, text: any(choice in text for choice in choices)
    smp.toliststr = lambda value: value if isinstance(value, list) else [value]
    root = Path(__file__).resolve().parents[1]
    for name in names:
        full_name = 'vlmeval.api.' + name
        spec = importlib.util.spec_from_file_location(full_name, root / ('vlmeval/api/' + name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        modules[full_name] = module
    return modules


@pytest.mark.parametrize('provider,class_name', [('gpt', 'OpenAIWrapper'), ('kimivl_api', 'KimiVLAPIWrapper'),
                                                 ('taiyi', 'TaiyiWrapper'), ('doubao_vl_api', 'DoubaoVLWrapper')])
def test_constructor_logs_exclude_api_key(provider, class_name, monkeypatch, caplog):
    modules = _load_provider_modules(provider)
    module = modules['vlmeval.api.' + provider]
    marker = 'test-credential-must-not-be-logged'
    monkeypatch.setenv('DOUBAO_VL_KEY', marker)
    module.logger.propagate = True
    with caplog.at_level(logging.INFO, logger=module.logger.name):
        wrapper = getattr(module, class_name)(**({} if provider == 'doubao_vl_api' else {'key': marker}))
    assert wrapper.key == marker
    assert marker not in caplog.text
    assert 'Using' in caplog.text
    if hasattr(wrapper, 'client'):
        wrapper.client.close()


def test_sensechat_v2_constructs_without_logging_credentials(caplog):
    module = _load_provider_modules('sensechat_vision')['vlmeval.api.sensechat_vision']
    module.logger.propagate = True
    with caplog.at_level(logging.INFO, logger=module.logger.name):
        wrapper = module.SenseChatVisionV2API(key='test-credential-must-not-be-logged')
    assert 'test-credential-must-not-be-logged' not in caplog.text
    assert wrapper.prepare_itlist([{'type': 'text', 'value': 'Hello'}]) == [{'type': 'text', 'text': 'Hello'}]
