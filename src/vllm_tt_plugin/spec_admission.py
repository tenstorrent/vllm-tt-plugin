# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Config-time admission policy for speculative decoding.

Separate from ``spec_decode``, which is the value contract a tt-metal model
class imports to build a ``SpecPlan``. Policy lives here so that contract module
stays a leaf with no vLLM dependency: only this module reads
``model_capabilities`` keys, resolves a vLLM speculative method name, and
decides whether a launch may proceed.

Every refusal raises. Speculation is never disabled silently, because a server
that quietly serves without it reports speedups it did not achieve.

The model-side contract this admits against, which a tt-metal model class must
implement, is documented in ``docs/SPEC_DECODE_CONTRACT.md``.
"""

from typing import TYPE_CHECKING, get_args

from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    ACCEPT_MODE_LOGITS,
    DRAFTER_STATE_PAGED,
    HIDDEN_HANDOFFS,
    SPEC_REQUIREMENT_DEVICE_PROPOSE,
    SPEC_REQUIREMENT_HIDDEN_FEED,
    SPEC_REQUIREMENT_PAGED_DRAFTER_CACHE,
    SPEC_REQUIREMENTS,
    SpecPlan,
    SpecReject,
    normalize_declared_values,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Accept modes the runner can drive today. A plan offering none of these is
# refused; a plan offering more keeps its extra modes and is still admitted,
# so declaring a real capability never makes a model less admissible.
_RUNNABLE_ACCEPT_MODES = (ACCEPT_MODE_LOGITS, ACCEPT_MODE_ARGMAX_IDS)

_DEVICE_DRAFTER = (SPEC_REQUIREMENT_DEVICE_PROPOSE, SPEC_REQUIREMENT_HIDDEN_FEED)


def _build_method_requirements() -> dict[str, tuple[str, ...]]:
    """One table mapping every servable vLLM method name to its requirements.

    Built once from vLLM's own literals rather than hand-copied, so an upstream
    rename drops a name out of this table instead of leaving the plugin mapping
    a name vLLM no longer knows. Adding support for a method means adding one
    entry here.
    """
    # Imported lazily: vllm.config pulls in a module that resolves
    # current_platform at import time, which loads this plugin, so importing it
    # at module scope from a module on the plugin's own bootstrap path is a
    # cycle. The same reasoning is spelled out at platform.py's vllm.config
    # import.
    from vllm.config.speculative import EagleModelTypes

    table: dict[str, tuple[str, ...]] = {
        # Host drafters ask nothing of the model. The model still needs a
        # spec_plan, because the plugin needs its feasible (max_num_seqs, K)
        # points either way.
        "ngram": (),
        "suffix": (),
        # A separate draft model carries its own growing KV cache, which the
        # scheduler must own.
        "draft_model": (
            SPEC_REQUIREMENT_DEVICE_PROPOSE,
            SPEC_REQUIREMENT_PAGED_DRAFTER_CACHE,
        ),
        # Device drafters that read the target's hidden state but are not in
        # vLLM's EagleModelTypes grouping.
        "medusa": _DEVICE_DRAFTER,
        "mlp_speculator": _DEVICE_DRAFTER,
    }
    # EagleModelTypes flattens to EAGLE, every MTP variant and dFlash. All of
    # them draft on device from the target's hidden state.
    for name in get_args(EagleModelTypes):
        table.setdefault(name, _DEVICE_DRAFTER)
    return table


def method_requirements(method: str) -> tuple[str, ...]:
    """What a vLLM speculative method requires of the model.

    The plugin owns this mapping, so a new upstream method name is a plugin
    change rather than a model release, and a model declares what it can serve
    without ever enumerating method names.
    """
    table = _build_method_requirements()
    if method in table:
        return table[method]
    raise ValueError(
        f"speculative method {method!r} is not supported by the TT backend. "
        f"Set --spec-method, or the 'method' key inside --speculative-config, "
        f"to one of {sorted(table)}"
    )


def refuse_unimplemented_execution(model_class: type) -> None:
    """Refuse a resolved plan, because nothing can execute it yet.

    Called after resolution, the writeback and the store, so a model author
    sees their own declaration error rather than this blanket refusal, and so
    whoever implements the execution path deletes exactly one call.
    """
    raise ValueError(
        f"{model_class.__name__} is admissible for speculative decoding, but "
        "the TT backend cannot execute it yet: TTWorker implements no "
        "take_draft_token_ids, which vLLM's EngineCore calls on every step of "
        "a speculative run, and TTModelRunner drives no verify-then-propose "
        "loop. Refused rather than started, because a server that accepts the "
        "flags and serves no speculation reports a speedup it did not achieve. "
        "Drop the speculative flags to serve this model"
    )


def resolve_speculative_plan(
    vllm_config: "VllmConfig",
    model_class: type,
    model_capabilities: dict | None,
    max_num_seqs: int,
) -> SpecPlan | None:
    """Resolve a speculative configuration at config time, or refuse it.

    Returns the model's plan when speculation is admitted, and ``None`` when
    the configuration asks for no speculation. Every refusal raises with the
    offending values.

    ``max_num_seqs`` is the concurrency the plan is dimensioned against, passed
    in rather than read off ``vllm_config`` because the platform rewrites
    ``scheduler_config.max_num_seqs`` when it folds data parallelism into lanes.
    Pass the value that is final for the launch.

    The model's ``spec_plan`` classmethod is called with a ``vllm_config`` on
    which no Tenstorrent platform state is guaranteed to be stored, so a
    ``spec_plan`` implementation must not read ``get_tt_*`` helpers off it.
    """
    speculative_config = vllm_config.speculative_config
    if not speculative_config:
        return None

    capabilities = model_capabilities or {}
    if not capabilities.get("supports_spec_decode", False):
        raise ValueError(
            f"{model_class.__name__} does not declare "
            "model_capabilities['supports_spec_decode'], so it cannot serve a "
            "speculative_config. Drop the speculative flags"
        )

    method = str(speculative_config.method)
    required = method_requirements(method)
    declared = normalize_declared_values(
        capabilities.get("spec_requirements"),
        SPEC_REQUIREMENTS,
        f"{model_class.__name__} model_capabilities['spec_requirements']",
    )
    missing = [name for name in required if name not in declared]
    if missing:
        raise ValueError(
            f"speculative method {method!r} requires {list(required)}, but "
            f"{model_class.__name__} declares spec_requirements "
            f"{list(declared)}; missing {missing}. Change --spec-method, or "
            "drop the speculative flags"
        )

    # Validated only when the method needs it, so a typo in an unused
    # declaration does not refuse an ngram launch. Which handoff the model
    # declared is a real behavioural difference, on device or a host round
    # trip, but nothing consumes it until the runner holds a HiddenHandle
    # between propose and verify, so SpecPlan carries no field for it yet.
    if SPEC_REQUIREMENT_HIDDEN_FEED in required:
        handoff = normalize_declared_values(
            capabilities.get("spec_hidden_handoff"),
            HIDDEN_HANDOFFS,
            f"{model_class.__name__} model_capabilities['spec_hidden_handoff']",
        )
        if not handoff:
            raise ValueError(
                f"speculative method {method!r} feeds the target hidden state "
                f"to its drafter, but {model_class.__name__} declares no "
                f"spec_hidden_handoff; expected one of "
                f"{sorted(HIDDEN_HANDOFFS)}"
            )

    # Refused on the requirement, not only on the returned plan, so a model
    # whose two declarations contradict each other is reported rather than
    # admitted through the gap between them.
    if SPEC_REQUIREMENT_PAGED_DRAFTER_CACHE in required:
        raise ValueError(
            f"speculative method {method!r} needs a scheduler-owned drafter "
            "cache, which the TT backend does not yet allocate. Change "
            "--spec-method, or drop the speculative flags"
        )

    spec_plan = getattr(model_class, "spec_plan", None)
    if not callable(spec_plan):
        raise ValueError(
            f"{model_class.__name__} declares supports_spec_decode but its "
            f"spec_plan is {spec_plan!r}, not a callable classmethod, so its "
            "feasible (max_num_seqs, K) points cannot be resolved"
        )

    requested_k = int(speculative_config.num_speculative_tokens)
    outcome = spec_plan(vllm_config, max_num_seqs, requested_k)

    if isinstance(outcome, SpecReject):
        supported = (
            f"; supported draft lengths at max_num_seqs={max_num_seqs}: "
            f"{list(outcome.supported_k)}"
            if outcome.supported_k
            else f"; no draft length is supported at max_num_seqs={max_num_seqs}"
        )
        raise ValueError(
            f"{model_class.__name__} refused speculation at "
            f"max_num_seqs={max_num_seqs}, num_speculative_tokens="
            f"{requested_k}: {outcome.reason}{supported}. Change --spec-tokens "
            "or --max-num-seqs, or drop the speculative flags"
        )
    if not isinstance(outcome, SpecPlan):
        raise ValueError(
            f"{model_class.__name__}.spec_plan must return SpecPlan or "
            f"SpecReject, got {type(outcome).__name__}"
        )
    if outcome.effective_k > requested_k:
        raise ValueError(
            f"{model_class.__name__}.spec_plan returned effective_k="
            f"{outcome.effective_k}, above the requested "
            f"num_speculative_tokens={requested_k}"
        )
    if outcome.drafter_state == DRAFTER_STATE_PAGED:
        raise ValueError(
            f"{model_class.__name__}.spec_plan declares drafter_state "
            f"{DRAFTER_STATE_PAGED!r}, which needs a scheduler-owned drafter "
            "cache the TT backend does not yet allocate"
        )
    if not any(mode in outcome.accept_modes for mode in _RUNNABLE_ACCEPT_MODES):
        raise ValueError(
            f"{model_class.__name__}.spec_plan offers accept modes "
            f"{list(outcome.accept_modes)}, none of which the runner can "
            f"drive; it drives {list(_RUNNABLE_ACCEPT_MODES)}"
        )

    return outcome


__all__ = [
    "method_requirements",
    "refuse_unimplemented_execution",
    "resolve_speculative_plan",
]
