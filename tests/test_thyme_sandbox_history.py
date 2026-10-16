"""Regression tests for Thyme's iterative sandbox conversation history."""

import copy
import importlib.util
import sys
import types


class FakeInputs(dict):

    def __init__(self):
        super().__init__(input_ids=[[1]])
        self.input_ids = self["input_ids"]

    def to(self, device):
        assert device == "cuda"
        return self


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self):
        self.outputs = iter(
            [
                "<code>make_crop()</code>",
                "<answer>done</answer>",
            ]
        )

    def batch_decode(self, *args, **kwargs):
        return [next(self.outputs)]


class FakeProcessor:

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((copy.deepcopy(messages[0]), kwargs))
        return ["rendered prompt"]

    def __call__(self, **kwargs):
        return FakeInputs()


class FakeModel:

    def __init__(self):
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [[1, len(self.generate_calls)]]


def _load_thyme_module(monkeypatch, sandbox_result, vision_histories):
    packages = {
        "vlmeval": "vlmeval",
        "vlmeval.vlm": "vlmeval/vlm",
        "vlmeval.vlm.thyme": "vlmeval/vlm/thyme",
    }
    for name, path in packages.items():
        package = types.ModuleType(name)
        package.__path__ = [path]
        monkeypatch.setitem(sys.modules, name, package)

    torch = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", torch)

    base = types.ModuleType("vlmeval.vlm.base")
    base.BaseModel = type("BaseModel", (), {})
    monkeypatch.setitem(sys.modules, "vlmeval.vlm.base", base)

    prompt = types.ModuleType("vlmeval.vlm.thyme.prompt")
    prompt.ThymePromptMixin = type("ThymePromptMixin", (), {})
    monkeypatch.setitem(sys.modules, "vlmeval.vlm.thyme.prompt", prompt)

    sandbox = types.ModuleType("vlmeval.vlm.thyme.sandbox")
    sandbox.execute_code_in_sandbox = lambda *args, **kwargs: sandbox_result
    monkeypatch.setitem(sys.modules, "vlmeval.vlm.thyme.sandbox", sandbox)

    utils = types.ModuleType("vlmeval.vlm.thyme.utils")
    utils.REASONING_SYS_PROMPT = "reason"
    utils.SIMPLE_SYS_PROMPT = "simple"
    utils.SPECIAL_STRING_LIST = ["</code>", "</answer>"]
    utils.generate_prompt_final_qa = lambda question, image: question
    utils.generate_prompt_simple_qa = lambda question: question
    monkeypatch.setitem(sys.modules, "vlmeval.vlm.thyme.utils", utils)

    qwen_vl_utils = types.ModuleType("qwen_vl_utils")

    def process_vision_info(messages):
        vision_histories.append(copy.deepcopy(messages[0]))
        return [], []

    qwen_vl_utils.process_vision_info = process_vision_info
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qwen_vl_utils)

    spec = importlib.util.spec_from_file_location(
        "vlmeval.vlm.thyme.model", "vlmeval/vlm/thyme/model.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "vlmeval.vlm.thyme.model", module)
    spec.loader.exec_module(module)
    return module


def test_sandbox_image_is_a_user_observation_in_reencoded_history(
        monkeypatch, tmp_path):
    original = tmp_path / "original.png"
    crop = tmp_path / "crop.png"
    original.touch()
    crop.touch()
    vision_histories = []
    sandbox_result = ([str(crop)], "", "", {})
    module = _load_thyme_module(monkeypatch, sandbox_result, vision_histories)

    thyme = module.Thyme.__new__(module.Thyme)
    thyme.min_pixels = None
    thyme.max_pixels = None
    thyme.fps = 2.0
    thyme.nframe = 64
    thyme.FRAME_FACTOR = 2
    thyme.verbose = False
    thyme.max_retry = 1
    thyme.max_iterations = 2
    thyme.post_process = True
    thyme.temperature = 0.01
    thyme.generate_kwargs = {
        "temperature": thyme.temperature,
        "stop_strings": ["</code>", "</answer>"],
    }
    thyme.processor = FakeProcessor()
    thyme.model = FakeModel()

    answer = thyme.generate_inner_transformers(
        [
            {"type": "image", "value": str(original)},
            {"type": "text", "value": "What is shown?"},
        ],
        temp_output_dir=str(tmp_path),
    )

    assert answer == "done"
    assert len(thyme.processor.template_calls) == 2
    assert all(
        kwargs["add_generation_prompt"]
        for _, kwargs in thyme.processor.template_calls
    )
    assert all("past_key_values" not in call for call in thyme.model.generate_calls)

    second_history = thyme.processor.template_calls[1][0]
    assert [message["role"] for message in second_history] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assistant_items = second_history[2]["content"]
    assert assistant_items == [
        {"type": "text", "text": "<code>make_crop()</code>"}
    ]
    observation_items = second_history[3]["content"]
    assert observation_items == [
        {"type": "text", "text": "<sandbox_output>"},
        {"type": "image", "image": str(crop)},
        {"type": "text", "text": "</sandbox_output>"},
    ]
    assert vision_histories == [call[0] for call in thyme.processor.template_calls]
