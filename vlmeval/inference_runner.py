"""Per-sample execution strategies, independent of backend orchestration.

Backends call prepare_prompt() before inference; dataset and model builders
construct the actual prompts. run() only consumes prepared inputs. Runners own
no mutable sample state. A dataset opts in with
``INFERENCE_RUNNER = InteractiveMTRunner``.
"""

from abc import ABC, abstractmethod
from inspect import Parameter, signature

from vlmeval.interactive_mt import (FAIL_MSG, ModelCaller, build_interactive_prompt, has_failure,
                                    run_policy, validate_interactive_dataset)


class InferenceRunner(ABC):
    """Interface for a sample strategy reused by backend workers.

    Implementations must keep mutable case state in per-case policies so a
    runner can serve multiple samples concurrently.
    """

    @classmethod
    @abstractmethod
    def validate_dataset(cls, dataset):
        """Check a dataset's strategy-specific contract before inference.

        Args:
            dataset: Dataset instance selecting this runner class.

        Returns:
            None when the required hooks and their signatures are valid.

        Raises:
            TypeError: A required hook is missing, is not callable, or has an
                incompatible signature.

        Strategies with no additional dataset requirements implement a no-op.
        Validation must not call the model or create mutable case state.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_prompt(self, model, dataset, sample, dataset_name, *, video_llm=None):
        """Delegate prompt preparation before executing a sample.

        Args:
            model: Model whose prompt builder may be selected by the strategy.
            dataset: Dataset providing prompt and optional case-building hooks.
            sample: Backend-supplied input, forwarded unchanged to its builder.
                The builder interprets it; no row or index type is required.
            dataset_name: Logical dataset name used for model prompt selection.
            video_llm: Optional backend override for video prompt mode. None
                lets the concrete strategy use the model's default.

        Returns:
            Prepared input accepted by this strategy's run() method, such as
            a single-turn message list or a list of prepared interactive cases.
            A builder's None result is passed back to the calling backend.

        This stage selects and calls builders without performing inference.
        Prompt content and media handling belong to those builders.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, model, dataset, sample, dataset_name, *, message, policy_llm=None):
        """Execute one sample using its prepared input.

        Args:
            model: Model under test, called through generate() or chat().
            dataset: Dataset providing any policies required by the strategy.
            sample: Original backend input, with the same meaning as in
                prepare_prompt(); the runner imposes no additional input type.
            dataset_name: Logical dataset name passed to model inference calls.
            message: Output of this strategy's prepare_prompt() for the sample.
            policy_llm: Optional shared LLM client for case policies, independent
                of the model under test and with no mutable case state.

        Returns:
            Prediction in the format expected by the dataset's evaluator.

        Consume the prepared input without rebuilding prompts. Scheduling,
        checkpoints, retries and backend error handling belong to the caller;
        interactive case state belongs to each newly created TurnPolicy.
        """
        raise NotImplementedError

    @abstractmethod
    def is_failed(self, prediction) -> bool:
        """Decide whether a sample prediction failed and should be retried.

        Args:
            prediction: Finalized output from run() or a prediction loaded
                from a checkpoint or result file.

        Returns:
            True if the prediction contains a failure under this strategy's
            rules. Backends discard it only when retry_failed is enabled and
            retry the entire sample, including all its cases.

        Interpret the strategy's prediction format without mutating it or
        calling a model. Backends use this interface for result reuse and
        need no knowledge of the concrete runner or its failure markers.
        """
        raise NotImplementedError


class SingleTurnRunner(InferenceRunner):
    @classmethod
    def validate_dataset(cls, dataset):
        """Single-turn inference adds no dataset hook requirements."""

    def prepare_prompt(self, model, dataset, sample, dataset_name, *, video_llm=None):
        """Select dataset or model prompts consistently across all backends."""
        video = getattr(dataset, 'MODALITY', None) == 'VIDEO'
        prompt_kwargs = {}
        if video:
            prompt_kwargs['video_llm'] = (
                getattr(model, 'VIDEO_LLM', False) if video_llm is None else video_llm)
        if (not getattr(dataset, 'force_use_dataset_prompt', False)
                and hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name)):
            custom_kwargs = {'dataset': dataset_name}
            if video:
                # Existing video-aware builders take the dataset object;
                # standard model builders take its name and no video kwargs.
                parameters = signature(model.build_prompt).parameters
                if 'video_llm' in parameters or any(
                    p.kind == Parameter.VAR_KEYWORD for p in parameters.values()
                ):
                    if dataset.nframe == 0:
                        raise ValueError(
                            'nframe must be set for custom prompt, fps is not suitable.')
                    custom_kwargs = {'dataset': dataset, **prompt_kwargs}
            return model.build_prompt(sample, **custom_kwargs)
        return dataset.build_prompt(sample, **prompt_kwargs)

    def run(self, model, dataset, sample, dataset_name, *, message, policy_llm=None):
        return model.generate(message=message, dataset=dataset_name)

    def is_failed(self, prediction) -> bool:
        """Keep the existing single-turn API failure marker check."""
        return FAIL_MSG in str(prediction)


class InteractiveMTRunner(InferenceRunner):
    """Execute the prepared cases with a fresh TurnPolicy for each case."""

    @classmethod
    def validate_dataset(cls, dataset):
        validate_interactive_dataset(dataset)

    def prepare_prompt(self, model, dataset, sample, dataset_name, *, video_llm=None):
        return build_interactive_prompt(dataset, sample)

    def run(self, model, dataset, sample, dataset_name, *, message, policy_llm=None):
        caller = ModelCaller(model, dataset_name)
        policy_kwargs = {} if policy_llm is None else {'policy_llm': policy_llm}
        predictions = []
        for case in message:
            policy = dataset.build_policy(case.sample, messages=case.messages, **policy_kwargs)
            predictions.append(run_policy(policy, case.messages, caller))
        return predictions[0] if len(predictions) == 1 else predictions

    def is_failed(self, prediction) -> bool:
        """Detect model or policy failures in any finalized case or turn."""
        return has_failure(prediction)


def get_inference_runner(dataset):
    """Resolve a class-valued strategy and fail clearly for invalid contracts."""
    runner_class = getattr(dataset, 'INFERENCE_RUNNER', SingleTurnRunner)
    if not isinstance(runner_class, type) or not issubclass(runner_class, InferenceRunner):
        raise TypeError(
            f'{type(dataset).__name__}.INFERENCE_RUNNER must be an InferenceRunner class.'
        )
    runner_class.validate_dataset(dataset)
    return runner_class()


def dispatch_sample(
    model, dataset, sample, dataset_name, *, message, policy_llm=None, inference_runner=None,
):
    """The shared execution contract for local, direct API and API pipeline calls."""
    runner = inference_runner if inference_runner is not None else get_inference_runner(dataset)
    return runner.run(
        model, dataset, sample, dataset_name, message=message, policy_llm=policy_llm)
