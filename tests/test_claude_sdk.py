import base64
import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from vlmeval.api.claude_sdk import ClaudeSDKAPI, ClaudeSDKWrapper


def make_response(text='answer', stop_reason='end_turn'):
    return SimpleNamespace(
        content=[
            SimpleNamespace(type='thinking', thinking='private reasoning'),
            SimpleNamespace(type='text', text=text),
        ],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        model_dump_json=lambda: '{"id":"test-response"}',
    )


@pytest.fixture
def client_factory(monkeypatch):
    import anthropic

    created = {}
    client = SimpleNamespace(messages=SimpleNamespace(create=Mock(return_value=make_response())))

    def create_client(**kwargs):
        created.update(kwargs)
        return client

    monkeypatch.setattr(anthropic, 'Anthropic', create_client)
    return created, client


def test_init_thinking_and_effort_are_sent_to_anthropic_sdk(client_factory):
    created, client = client_factory
    model = ClaudeSDKWrapper(
        model='claude-sonnet-5-5',
        thinking={'type': 'adaptive'},
        effort='high',
        max_tokens=2048,
        key='test-key',
        api_base='https://tensoris.invalid/v1',
        verbose=False,
    )

    code, answer, _ = model.generate_inner([{'type': 'text', 'value': 'Question'}])

    assert code == 0
    assert answer == 'answer'
    assert created['base_url'] == 'https://tensoris.invalid'
    assert created['api_key'] == 'test-key'
    payload = client.messages.create.call_args.kwargs
    assert payload['model'] == 'claude-sonnet-5-5'
    assert payload['max_tokens'] == 2048
    assert payload['thinking'] == {'type': 'adaptive'}
    assert payload['output_config'] == {'effort': 'high'}
    assert payload['messages'] == [{
        'role': 'user',
        'content': [{'type': 'text', 'text': 'Question'}],
    }]


def test_generate_overrides_init_thinking_and_effort_and_extracts_text(client_factory):
    _, client = client_factory
    model = ClaudeSDKAPI(
        model='claude-sonnet-5-5',
        thinking={'type': 'disabled'},
        effort='low',
        key='test-key',
        api_base='https://tensoris.invalid/v1',
        verbose=False,
    )
    messages = [
        {'role': 'system', 'type': 'text', 'value': 'Be precise.'},
        {'role': 'user', 'type': 'text', 'value': 'Say hello.'},
    ]

    assert model.generate(
        messages,
        thinking={'type': 'adaptive'},
        effort='xhigh',
    ) == 'answer'

    payload = client.messages.create.call_args.kwargs
    assert payload['system'] == 'Be precise.'
    assert payload['thinking'] == {'type': 'adaptive'}
    assert payload['output_config'] == {'effort': 'xhigh'}


def test_streaming_returns_text_from_final_message(client_factory):
    _, client = client_factory
    final_message = make_response(text='stream answer')
    stream = Mock()
    stream.__enter__ = Mock(return_value=stream)
    stream.__exit__ = Mock(return_value=False)
    stream.get_final_message = Mock(return_value=final_message)
    client.messages.stream = Mock(return_value=stream)

    model = ClaudeSDKWrapper(
        model='claude-sonnet-5-5',
        stream=True,
        key='test-key',
        api_base='https://tensoris.invalid/v1',
        verbose=False,
    )
    code, answer, _ = model.generate_inner([{'type': 'text', 'value': 'Question'}])

    assert code == 0
    assert answer == 'stream answer'
    client.messages.stream.assert_called_once()
    client.messages.create.assert_not_called()


@pytest.mark.parametrize('mode,suffix', [
    ('RGB', '.jpg'), ('RGB', '.png'), ('RGBA', '.png'), ('LA', '.png'), ('P', '.png'), ('L', '.png'),
    ('1', '.png'), ('CMYK', '.jpg'),
])
@pytest.mark.parametrize('convert_rgb', [False, True])
def test_generate_encodes_images_with_matching_mime(client_factory, tmp_path, mode, suffix, convert_rgb):
    _, client = client_factory
    path = tmp_path / ('image' + suffix)
    source = Image.new(mode, (240, 160))
    if mode == 'P':
        source.putpalette([0, 0, 0] + [0, 0, 0] * 255)
        source.info['transparency'] = 0
    source.save(path)
    model = ClaudeSDKAPI(key='test-key', verbose=False, convert_rgb=convert_rgb)

    assert model.generate([
        {'type': 'text', 'value': 'Describe this image.'},
        {'type': 'image', 'value': str(path)},
    ]) == 'answer'

    content = client.messages.create.call_args.kwargs['messages'][0]['content']
    assert [part['type'] for part in content] == ['text', 'image']
    source = content[1]['source']
    assert source['type'] == 'base64'
    fmt = 'JPEG' if convert_rgb or mode == 'CMYK' else 'PNG'
    assert source['media_type'] == Image.MIME[fmt]
    with Image.open(io.BytesIO(base64.b64decode(source['data'], validate=True))) as encoded:
        assert encoded.format == fmt
        assert encoded.size == (240, 160)
    with Image.open(path) as original:
        assert original.mode == mode
        assert original.size == (240, 160)


def test_multiple_images_preserve_order_and_total_size_budget(client_factory, tmp_path):
    _, client = client_factory
    paths = [tmp_path / 'red.png', tmp_path / 'blue.png']
    for path, color in zip(paths, ('red', 'blue')):
        Image.new('RGB', (800, 400), color).save(path)
    model = ClaudeSDKAPI(
        key='test-key', img_size=400, total_img_size=400, verbose=False,
    )

    assert model.generate([
        {'type': 'text', 'value': 'First'},
        {'type': 'image', 'value': str(paths[0])},
        {'type': 'text', 'value': 'Second'},
        {'type': 'image', 'value': str(paths[1])},
    ]) == 'answer'

    content = client.messages.create.call_args.kwargs['messages'][0]['content']
    assert [part['type'] for part in content] == ['text', 'image', 'text', 'image']
    assert content[0]['text'] == 'First'
    assert content[2]['text'] == 'Second'
    for part, channel in zip((content[1], content[3]), (0, 2)):
        with Image.open(io.BytesIO(base64.b64decode(part['source']['data']))) as encoded:
            assert encoded.size == (282, 141)
            assert encoded.getpixel((100, 100))[channel] > 240


@pytest.mark.parametrize('image_count,multi_turn,expected_edge', [
    (1, False, 8000), (21, False, 3000), (21, True, 3000),
])
def test_image_dimension_limits_follow_anthropic_request_rules(
        client_factory, tmp_path, image_count, multi_turn, expected_edge):
    _, client = client_factory
    path = tmp_path / 'wide.png'
    Image.new('RGB', (8100, 500), 'green').save(path)
    model = ClaudeSDKAPI(key='test-key', verbose=False)
    inputs = [{'type': 'image', 'value': str(path)} for _ in range(image_count)]
    if multi_turn:
        inputs = [
            {'role': 'user', 'content': inputs[:11]},
            {'role': 'assistant', 'content': [{'type': 'text', 'value': 'Continue.'}]},
            {'role': 'user', 'content': inputs[11:]},
        ]

    messages, _ = model.prepare_inputs(inputs)

    images = [
        part for message in messages for part in message['content'] if part['type'] == 'image'
    ]
    assert len(images) == image_count
    for image in images:
        with Image.open(io.BytesIO(base64.b64decode(image['source']['data']))) as encoded:
            assert max(encoded.size) == expected_edge

    client.messages.create.assert_not_called()


def test_image_encoding_respects_vlmeval_environment_limit(client_factory, monkeypatch, tmp_path):
    monkeypatch.setenv('VLMEVAL_MAX_IMAGE_SIZE', '10000')
    path = tmp_path / 'noise.png'
    Image.effect_noise((256, 256), 100).convert('RGB').save(path, format='PNG')
    model = ClaudeSDKAPI(key='test-key', verbose=False)

    content = model.prepare_itlist([{'type': 'image', 'value': str(path)}])

    assert len(content[0]['source']['data']) <= 10000


def test_image_encoding_respects_file_size_limit(client_factory, monkeypatch, tmp_path):
    monkeypatch.delenv('VLMEVAL_MAX_IMAGE_SIZE', raising=False)
    path = tmp_path / 'noise.png'
    Image.effect_noise((256, 256), 100).convert('RGB').save(path, format='PNG')
    model = ClaudeSDKAPI(key='test-key', max_file_size=10000, verbose=False)

    content = model.prepare_itlist([{'type': 'image', 'value': str(path)}])

    assert len(content[0]['source']['data']) <= 10000


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('thinking_tokens', [0, 17, None])
@pytest.mark.parametrize('verbose', [False, True])
def test_logs_reported_reasoning_tokens(
        client_factory, monkeypatch, stream, thinking_tokens, verbose):
    _, client = client_factory
    response = make_response()
    if thinking_tokens is not None:
        response.usage.output_tokens = 20
        response.usage.output_tokens_details = SimpleNamespace(thinking_tokens=thinking_tokens)
    client.messages.create.return_value = response
    message_stream = Mock()
    message_stream.__enter__ = Mock(return_value=message_stream)
    message_stream.__exit__ = Mock(return_value=False)
    message_stream.get_final_message.return_value = response
    client.messages.stream = Mock(return_value=message_stream)
    log = Mock()
    monkeypatch.setattr('vlmeval.api.claude_sdk.logger.info', log)
    model = ClaudeSDKAPI(key='test-key', stream=stream, verbose=verbose)

    assert model.generate('Question') == 'answer'

    expected = 'unavailable (not reported by API)' if thinking_tokens is None else thinking_tokens
    assert f'reasoning_tokens={expected}' in log.call_args.args[0]
