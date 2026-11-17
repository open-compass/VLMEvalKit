"""Modality-agnostic multi-turn capability.

MT is a property of *how* a dataset is run, not of its media type, so it lives
in a mixin rather than in a dataset base class. A dataset opts in with::

    class XDataset(InteractiveMTMixin, ImageBaseDataset):
        ...

The mixin only declares the contract and a default case-level loop. It never
touches images, videos or any other media; datasets implement ``build_policy``
and override the base class ``build_prompt`` with whatever media handling they
need.

``build_prompt`` keeps its meaning from the base classes: it returns the input
of the first model call, and every runner builds it exactly as for a
single-turn dataset (same call, same arguments, ``video_llm`` for video data)
before handing it to the MT loop instead of to ``model.generate``. For an MT
dataset that input is the whole first-call context as role messages,
``[dict(role=..., content=[...]), ...]``, ending with the user turn to answer;
later turns come from the policy. The mixin does not declare ``build_prompt``
itself, so that the base class implementation is not shadowed; a dataset that
forgets to override it gets the base single-turn prompt, which has no roles and
is rejected by the MT loop on the first case.

One dataset row is one *case*: the largest unit whose model calls may depend on
each other (dependency graphs, probes asked in a context built from the
dialogue, ...), all handled inside one policy. Different cases never depend on
each other. A benchmark with several independent conditions per item (golden vs
self-predicted history, several contexts per image, ...) expands them into
separate rows in ``load_data``, or exposes them as separate dataset names, and
regroups them in ``evaluate``.
"""

from abc import ABC, abstractmethod

from vlmeval.interactive_mt import run_default_loop


class InteractiveMTMixin(ABC):
    """Capability marker plus the minimal MT contract."""

    # Marks the dataset as policy-driven multi-turn. Runners read this flag
    # instead of isinstance(), so a dataset may opt out by overriding it.
    interactive_mt = True

    DEFAULT_POLICY_MODEL = None

    # An MT script is multi-turn, so a model's single-turn custom prompt must
    # not replace it. The runners already honor this flag when they build the
    # first prompt; subclasses that really want the model prompt can override.
    force_use_dataset_prompt = True

    def build_policy_llm(self, policy_kwargs):
        """Create the model(s) that policies call internally, or ``None``.

        The runner calls this once per dataset, before inference, with the
        ``--policy*`` options as a private dict. Those options are consumed
        here and go no further: ``build_policy`` never sees them.

        The return value is opaque to the framework (an API client, a local
        model, or a dict of roles). It is shared by every case and every API
        worker thread and is never copied, so it must hold no per-case state
        and be safe to call concurrently.

        One role is the default: ``--policy``, ``--policy-key``,
        ``--policy-base-url``, ``--policy-retry`` and ``--policy-timeout``
        configure it, and ``DEFAULT_POLICY_MODEL`` names its default model.

        A dataset whose policy needs several LLMs (examiner + context
        generator, ...) returns a dict keyed by role and reads per-role
        settings from ``--policy-args``, e.g.
        ``{"model": "m", "roles": {"examiner": {"model": "a"}, "context":
        {"model": "b"}}}``. Top-level fields are shared defaults and each role
        overrides them; role names and per-role default models are the
        dataset's own convention. Roles that run before inference (data
        preparation) or after it (evaluation) do not belong here.
        """
        return None

    @abstractmethod
    def build_policy(self, case, messages, *, policy_llm=None):
        """Build a single-use policy for this case.

        When ``build_policy_llm`` returned something, it arrives as the keyword
        ``policy_llm`` (shared, not copied); otherwise the keyword is absent.
        """
        raise NotImplementedError

    def get_default_policy_model(self, policy_kwargs=None):
        return self.DEFAULT_POLICY_MODEL

    def run_case(self, case, messages, caller, policy_llm=None):
        """Run one case and return its finalized prediction.

        ``messages`` is the first-call context: the runner builds it with
        ``build_prompt`` exactly as for a single-turn dataset (so video
        datasets receive ``video_llm`` there) and passes it in.

        Override only when a benchmark needs to control the number or order of
        model calls (dependency graphs, abort-and-restart on judge failure).
        Inside an override, reach the model exclusively through ``caller`` so
        failure detection stays uniform.
        """
        return run_default_loop(self, case, messages, caller, policy_llm)
