"""Prompt preparation and case policies for interactive multi-turn datasets."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from inspect import signature
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


class ModelCallError(RuntimeError):
    """A model call failed (exception or a failure marker in the response)."""


@dataclass
class Step:
    """What the runner should do after a model response.

    Only ``messages`` is needed by turn-based datasets. The other fields let a
    policy rebase the visible history, override generation parameters for the
    next call, or ask for several samples from the same prefix without taking
    over the loop.
    """

    # User messages appended to the history before the next model call.
    messages: list = field(default_factory=list)
    # Explicit replacement for the next call's history (reference history,
    # context reset, streaming visibility). None keeps the current history.
    history: list | None = None
    # Extra keyword arguments for the next model call only.
    chat_kwargs: dict = field(default_factory=dict)
    # Number of candidates to sample from the same prefix.
    num_samples: int = 1
    done: bool = False


class TurnPolicy(ABC):
    """Single-use policy whose state belongs to one benchmark case."""

    max_turns: int

    def __init__(self):
        self.state = {}
        self.next_turn = 0

    def on_failure(self, message: str, history: list, turn: int, reason: str = 'model'):
        """Record a failed step before the runner finalizes this case.

        ``reason`` is 'model' for a failed model call and 'policy' for an
        exception raised by :meth:`step` or :meth:`on_samples`.
        """
        self.state['failure'] = {
            'message': message,
            'history': deepcopy(history),
            'turn': turn,
            'reason': reason,
        }

    @abstractmethod
    def step(self, assistant_response: str, history: list) -> Step:
        """Record a response and choose the next user message or finish."""
        raise NotImplementedError

    def on_samples(self, candidates: list, history: list) -> Step:
        """Choose how to continue after sampling several candidates."""
        raise NotImplementedError('This policy does not accept num_samples > 1.')

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

    def on_failure(self, message, history, turn, reason='model'):
        # Keep the failed answer in its turn slot so the evaluator still sees
        # one entry per turn.
        self.state['responses'].append(message)
        super().on_failure(message, history, turn, reason)

    def step(self, assistant_response: str, history: list) -> Step:
        self.state['responses'].append(assistant_response)
        self.next_turn += 1
        if self.next_turn >= self.max_turns:
            return Step(done=True)
        return Step(messages=[deepcopy(self.user_turns[self.next_turn - 1])])

    def finalize(self):
        return self.state['responses']


class ModelCaller:
    """The only way a case may reach the model under test.

    Failure detection lives here so every policy reports model failures the
    same way.
    """

    def __init__(self, model, dataset_name):
        if not callable(getattr(model, 'chat', None)):
            raise TypeError('InteractiveMTRunner requires a model with a callable chat() method.')
        self.model = model
        self.dataset_name = dataset_name
        self._fail_msg = getattr(model, 'fail_msg', None)

    def chat(self, history, **kwargs):
        try:
            if kwargs:
                response = self.model.chat(history, dataset=self.dataset_name, **kwargs)
            else:
                response = self.model.chat(history, dataset=self.dataset_name)
        except Exception as error:
            raise ModelCallError(f'{FAIL_MSG} {type(error).__name__}: {error}') from error

        if has_failure(response) or (
            isinstance(response, str) and self._fail_msg and self._fail_msg in response
        ):
            raise ModelCallError(str(response))
        return response

    def chat_n(self, history, n, **kwargs):
        return [self.chat(deepcopy(history), **kwargs) for _ in range(n)]


def validate_interactive_dataset(dataset):
    """Validate required hooks, and optional hooks when explicitly provided."""
    for name, args, kwargs in (
        ('build_prompt', (None,), {}),
        ('build_policy', (None,), {'messages': []}),
    ):
        hook = getattr(dataset, name, None)
        if not callable(hook):
            raise TypeError(
                f'{type(dataset).__name__} uses InteractiveMTRunner and requires {name}().')
        try:
            signature(hook).bind(*args, **kwargs)
        except TypeError as error:
            raise TypeError(
                f'{type(dataset).__name__}.{name} has an invalid hook signature.') from error
    for name in ('build_cases', 'build_policy_llm'):
        hook = getattr(dataset, name, None)
        if hook is not None and not callable(hook):
            raise TypeError(f'{type(dataset).__name__}.{name} must be callable when provided.')


@dataclass
class InteractiveCase:
    sample: Any
    messages: list


def build_interactive_prompt(dataset, sample):
    """Build case inputs with dataset hooks before running inference.

    Interactive datasets own their prompts and media handling. ``build_cases``
    is optional; an omitted hook means one case containing the original sample.
    """
    validate_interactive_dataset(dataset)
    build_cases = getattr(dataset, 'build_cases', None)
    cases = [sample] if build_cases is None else list(build_cases(sample))
    if not cases:
        raise ValueError('An MT sample must produce at least one case.')
    prepared = []
    for case in cases:
        messages = dataset.build_prompt(case)
        if not messages or messages[-1].get('role') != 'user':
            raise ValueError('Initial MT messages must end with a user message.')
        prepared.append(InteractiveCase(case, messages))
    return prepared


def _turn_kwargs(step):
    kwargs = dict(step.chat_kwargs or {})
    return kwargs


def run_policy(policy, messages, caller):
    """Standard case-level loop: one policy, one history, at most max_turns calls.

    Prompts are already prepared; all mutable case state belongs to the policy.
    """
    if not isinstance(policy, TurnPolicy):
        raise TypeError('build_policy() must return a TurnPolicy instance.')
    if not isinstance(getattr(policy, 'max_turns', None), int) or policy.max_turns <= 0:
        raise ValueError('An MT policy must allow at least one turn.')

    history = deepcopy(messages)
    pending = Step()
    for turn in range(policy.max_turns):
        if pending.history is not None:
            history = deepcopy(pending.history)
        for message in pending.messages:
            if message.get('role') != 'user':
                raise ValueError('MT policies may only append user messages.')
            history.append(deepcopy(message))

        try:
            if pending.num_samples > 1:
                candidates = caller.chat_n(history, pending.num_samples, **_turn_kwargs(pending))
            else:
                response = caller.chat(history, **_turn_kwargs(pending))
        except ModelCallError as error:
            policy.on_failure(str(error), history, turn, reason='model')
            break

        try:
            if pending.num_samples > 1:
                step = policy.on_samples(candidates, deepcopy(history))
            else:
                history.append({'role': 'assistant', 'content': response})
                step = policy.step(response, history)
        except Exception as error:  # policy-side failure must not kill the run
            policy.on_failure(
                f'{FAIL_MSG} {type(error).__name__}: {error}', history, turn, reason='policy')
            break

        if step.done or turn + 1 >= policy.max_turns:
            break
        if not step.messages and step.history is None:
            raise ValueError('An unfinished MT policy must return a user message.')
        pending = step

    return policy.finalize()
