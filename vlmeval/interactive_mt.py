"""Common policy-driven runner for interactive multi-turn datasets."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

FAIL_MSG = 'Failed to obtain answer via API.'


def has_failure(value):
    """Return whether a response or finalized prediction contains a failure."""
    if isinstance(value, str):
        return any(marker in value for marker in (
            FAIL_MSG,
            'Failed to obtain answer',
            'Fail to obtain answer',
            'Chat Mode: Failed with all possible conversation turns.',
        ))
    if isinstance(value, dict):
        return any(has_failure(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_failure(item) for item in value)
    return False


@dataclass
class PolicyStep:
    user_message: dict | None
    done: bool


class TurnPolicy(ABC):
    """Single-use policy whose state belongs to one benchmark case."""

    max_turns: int

    def __init__(self):
        self.state = {}
        self.next_turn = 0

    def on_failure(self, assistant_response: str, history: list[dict], turn: int):
        """Record a failed response before the runner finalizes this case."""
        self.state['failure'] = {
            'message': assistant_response,
            'history': deepcopy(history),
            'turn': turn,
        }

    @abstractmethod
    def step(self, assistant_response: str, history: list[dict]) -> PolicyStep:
        """Record a response and choose the next user message or finish."""
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> Any:
        """Return the prediction expected by this dataset's evaluator."""
        raise NotImplementedError


class StaticTurnPolicy(TurnPolicy):
    """Replay fixed user turns after a complete initial model context."""

    def __init__(self, initial_messages, user_messages):
        super().__init__()
        self.initial_messages = deepcopy(initial_messages)
        self.user_turns = deepcopy(user_messages)
        if not self.initial_messages or self.initial_messages[-1].get('role') != 'user':
            raise ValueError('Initial MT messages must end with a user message.')
        if any(message.get('role') != 'user' for message in self.user_turns):
            raise ValueError('Static MT follow-up messages must be user turns.')
        self.max_turns = 1 + len(self.user_turns)
        self.state['responses'] = []

    def on_failure(self, assistant_response: str, history: list[dict], turn: int):
        self.state['responses'].append(assistant_response)
        super().on_failure(assistant_response, history, turn)

    def step(self, assistant_response: str, history: list[dict]) -> PolicyStep:
        self.state['responses'].append(assistant_response)
        self.next_turn += 1
        if self.next_turn >= self.max_turns:
            return PolicyStep(None, True)
        return PolicyStep(deepcopy(self.user_turns[self.next_turn - 1]), False)

    def finalize(self):
        return self.state['responses']


def _chat_with_policy(model, policy, initial_messages, dataset_name):
    if policy.max_turns <= 0:
        raise ValueError('An MT policy must allow at least one turn.')

    history = deepcopy(initial_messages)
    if not history or history[-1].get('role') != 'user':
        raise ValueError('Initial MT messages must end with a user message.')

    for turn in range(policy.max_turns):
        try:
            response = model.chat(history, dataset=dataset_name)
        except Exception as error:
            response = f'{FAIL_MSG} {type(error).__name__}: {error}'

        model_fail_msg = getattr(model, 'fail_msg', None)
        response_failed = has_failure(response) or (
            isinstance(response, str) and model_fail_msg and model_fail_msg in response
        )
        if response_failed:
            policy.on_failure(response, history, turn)
            break

        history.append({'role': 'assistant', 'content': response})
        step = policy.step(response, history)
        if step.done or turn + 1 >= policy.max_turns:
            break
        if step.user_message is None or step.user_message.get('role') != 'user':
            raise ValueError('An unfinished MT policy must return a user message.')
        history.append(step.user_message)

    return policy.finalize()


def build_mt_prompt(model, dataset, sample, dataset_name):
    if hasattr(model, 'set_dump_image') and hasattr(dataset, 'dump_image'):
        model.set_dump_image(dataset.dump_image)

    if getattr(dataset, 'force_use_dataset_prompt', False):
        return dataset.build_prompt(sample)
    if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
        return model.build_prompt(sample, dataset=dataset_name)
    return dataset.build_prompt(sample)


def chat_interactive_mt(model, dataset, sample, dataset_name, *, messages=None):
    """Run one interactive MT case and return its finalized prediction."""
    if messages is None:
        messages = build_mt_prompt(model, dataset, sample, dataset_name)
    policy = dataset.build_policy(sample, messages=messages)
    return _chat_with_policy(model, policy, messages, dataset_name)
