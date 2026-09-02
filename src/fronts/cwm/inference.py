"""Inference as code: the synthesised sampler ISMCTS determinizes with.

ISMCTS needs to sample from the belief over hidden states, and exact
posterior inference is exponential in the worst case. The paper's move --
and ours -- is to have the LLM synthesise an approximate sampler instead,
and to establish its correctness structurally rather than probabilistically:
replaying what it produces through the world model must reproduce every
observation actually seen.

The guarantee, stated precisely because it is easy to oversell. What
``validate_history`` buys is SUPPORT MEMBERSHIP, not density. As
``docs/PAPER.md`` s6 quotes the paper:

    "Although this does not guarantee that s_t is correctly distributed, the
     correct support is already very informative, given the extremely sparse
     support of state posteriors in games."

In distribution terms: a validated sample gives you ranking weights that
are not CONTRADICTED by your own results -- weights you can plan against,
which is strictly more than an operator reasoning by feel has. It does not
give you the right distribution over ranking weights, and a planner that
treats one sample as the truth is overconfidence with extra steps.

Because we are closed deck, ``resample_state`` is the primary path. The
paper is candid that it is the weaker variant -- it "cannot guarantee that
the produced sample belongs to the support of the posterior, nor that it
constitutes a valid CWM hidden state, because it ignores the dependency
between consecutive states." We use it anyway because the alternative does
not exist here: there is no offline record of hidden state to learn from.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from fronts.cwm.llm import LLMClient, extract_code
from fronts.cwm.sandbox import (
    Sandbox,
    SandboxConfig,
    call_with_timeout,
    check_protocol_methods,
    guard_methods,
)
from fronts.cwm.synth import SynthConfig
from fronts.cwm.tests_from_traj import Tolerance, _draw_recorded_chance
from fronts.game.types import (
    CHANCE_PLAYER,
    OPERATOR,
    TERMINAL_PLAYER,
    ActionKey,
    Observation,
    State,
    Trajectory,
)
from fronts.protocols import CodeWorldModel, HistoryInference, StateInference

__all__ = [
    "STATE_INFERENCE_CLASS",
    "HISTORY_INFERENCE_CLASS",
    "FallbackInference",
    "synthesise_state_inference",
    "synthesise_state_inference_source",
    "load_state_inference",
    "synthesise_history_inference",
    "validate_history",
    "inference_accuracy",
]


STATE_INFERENCE_CLASS = "StateInferenceSampler"
HISTORY_INFERENCE_CLASS = "HistoryInferenceSampler"

_STATE_METHODS = {"resample_state": 2}
_HISTORY_METHODS = {"resample_history": 2}

# Validation runs LOOSER than the generated tests' Tolerance defaults
# (counts 0.05 / rates 0.02), and the gap is deliberate. A unit test replays
# a recorded branch, so a prediction outside test precision is model error;
# validation replays an INFERRED sample whose claim is support membership,
# not density (module docstring), and demanding test-grade precision from a
# sampler that is approximate by construction would reject every honest
# sample. Stricter than THIS rejects honest approximations; looser admits
# samples that contradict the data.
_VALIDATION_TOLERANCE = Tolerance(counts=0.25, rates=0.10)


@dataclass(slots=True)
class FallbackInference:
    """The degenerate sampler: return the model's initial state.

    Used when synthesis fails, so a planner degrades to open-loop planning
    against an uninformative prior instead of crashing -- the reference
    paper's own fallback behaviour. A planner paired with this is a weaker
    planner, not a broken one, and the difference is visible in its results
    rather than in a stack trace.

    ``model`` is optional: the caller who owns the synthesised CWM passes it
    so the fallback returns that model's true opening state; without one the
    sampler returns an empty state, which claims nothing and contradicts
    nothing."""

    model: CodeWorldModel | None = None

    def resample_state(
        self,
        obs_action_history: list[tuple[Observation | None, ActionKey | None]],
        player_id: int,
    ) -> State:
        if self.model is None:
            return State({})
        return self.model.initial_state()

    def resample_history(
        self,
        obs_action_history: list[tuple[Observation | None, ActionKey | None]],
        player_id: int,
    ) -> list[ActionKey]:
        # An empty history is the do-nothing sample: no claimed events, so
        # nothing to replay. The planner that consumes it plans from the
        # prior, which is exactly the degradation intended.
        return []


def _inference_prompt(
    kind: str,
    class_name: str,
    method_name: str,
    rules: str,
    trajectories: list[Trajectory],
) -> tuple[str, str]:
    from fronts.cwm.synth import _serialise_trajectory

    system = (
        "You are a program synthesiser. You write approximate posterior "
        "samplers for the hidden state of a content-distribution environment, "
        "as Python classes. You write code, not explanations."
    )
    lines = [
        f"Write a Python module defining a class named {class_name} (no "
        "constructor arguments) implementing the required method below. The "
        "sampler must be deterministic given its arguments.",
        "",
        "== GAME RULES ==",
        rules.strip(),
        "",
        f"== REQUIRED API ({kind}) ==",
        f"    def {method_name}(self, obs_action_history: "
        "list[tuple[Observation | None, ActionKey | None]], player_id: int)"
        " -> ...: ...",
        "obs_action_history is the player's own observation/action sequence, "
        "oldest first; each element is (what the player saw after the previous "
        "move, the move it then made). Return ONE plausible sample consistent "
        "with every observation in it.",
        "",
        "== OBSERVED TRAJECTORIES ==",
    ]
    for index, trajectory in enumerate(trajectories):
        lines.append(f"trajectory[{index}]: {_serialise_trajectory(trajectory)}")
    lines += ["", "Respond with a single fenced Python code block."]
    return system, "\n".join(lines)


def _verified_instance(
    source: str, class_name: str, methods: dict[str, int]
) -> object | None:
    """Sandbox-load ``source`` and return a verified ``class_name`` instance,
    or None on any failure -- callers fall back rather than propagate, per
    the module contract."""
    try:
        namespace = Sandbox().load(source, SandboxConfig())
        cls = getattr(namespace, class_name, None)
        if cls is None:
            return None
        # The constructor is untrusted code too: a spinning __init__ would
        # hang the caller before any method guard could apply.
        instance: object = call_with_timeout(
            cls, (), SandboxConfig().timeout_seconds
        )
        if check_protocol_methods(instance, methods, class_name):
            return None
        return instance
    except Exception:
        return None


def _load_inference_class(
    client: LLMClient,
    config: SynthConfig,
    system: str,
    user: str,
    class_name: str,
    methods: dict[str, int],
) -> tuple[object, str] | None:
    """Complete, extract, sandbox, verify. Returns (instance, source) so a
    caller can persist what it just verified; None on any failure."""
    try:
        source = extract_code(client.complete(system, user))
    except Exception:
        return None
    instance = _verified_instance(source, class_name, methods)
    if instance is None:
        return None
    return instance, source


def synthesise_state_inference(
    client: LLMClient,
    config: SynthConfig,
    rules: str,
    trajectories: list[Trajectory],
    fallback_model: CodeWorldModel | None = None,
) -> StateInference:
    """Synthesise a ``resample_state`` sampler. The closed-deck primary path.

    Returns ``FallbackInference`` on failure so the planner keeps running;
    a failed sampler that crashed the planner would turn an inference
    quality problem into an availability problem, and nothing about the
    first justifies the second.
    """
    system, user = _inference_prompt(
        "closed-deck state inference",
        STATE_INFERENCE_CLASS,
        "resample_state",
        rules,
        trajectories,
    )
    loaded = _load_inference_class(
        client, config, system, user, STATE_INFERENCE_CLASS, _STATE_METHODS
    )
    if loaded is None or not isinstance(loaded[0], StateInference):
        return FallbackInference(model=fallback_model)
    return loaded[0]


def synthesise_state_inference_source(
    client: LLMClient,
    config: SynthConfig,
    rules: str,
    trajectories: list[Trajectory],
) -> str | None:
    """Synthesise a ``resample_state`` sampler and return its SOURCE.

    This is what ``fronts synth`` persists beside the world model, so a later
    ``fronts plan`` in a fresh process can sandbox-load the same sampler with
    ``load_state_inference``. None when synthesis fails -- the caller plans
    open-loop and says so, rather than writing a file that will not load.
    """
    system, user = _inference_prompt(
        "closed-deck state inference",
        STATE_INFERENCE_CLASS,
        "resample_state",
        rules,
        trajectories,
    )
    loaded = _load_inference_class(
        client, config, system, user, STATE_INFERENCE_CLASS, _STATE_METHODS
    )
    if loaded is None or not isinstance(loaded[0], StateInference):
        return None
    return loaded[1]


def load_state_inference(
    source: str, timeout_seconds: float | None = None
) -> StateInference | None:
    """Sandbox-load a persisted ``resample_state`` sampler.

    None when the source does not load or does not satisfy the protocol; the
    caller chooses its own degradation. ``fronts plan`` degrades to open-loop
    search against the true state -- NOT ``FallbackInference``, whose
    initial-state resample would silently discard the episode's progress.

    The returned sampler's ``resample_state`` is wrapped in the in-process
    timeout guard: the source came off disk and is untrusted, and a spinning
    sampler must cost the planner one skipped determinization (it catches the
    ``SandboxTimeout``), never the whole run.
    """
    budget = (
        SandboxConfig().timeout_seconds if timeout_seconds is None else timeout_seconds
    )
    instance = _verified_instance(source, STATE_INFERENCE_CLASS, _STATE_METHODS)
    if instance is None or not isinstance(instance, StateInference):
        return None
    return cast(
        StateInference, guard_methods(instance, ("resample_state",), budget)
    )


def synthesise_history_inference(
    client: LLMClient,
    config: SynthConfig,
    rules: str,
    trajectories: list[Trajectory],
    fallback_model: CodeWorldModel | None = None,
) -> HistoryInference:
    """Synthesise a ``resample_history`` sampler. The open-deck variant,
    provided for completeness; organic distribution is closed deck, so this
    path exists for evaluation against the paper, not for production."""
    system, user = _inference_prompt(
        "open-deck history inference",
        HISTORY_INFERENCE_CLASS,
        "resample_history",
        rules,
        trajectories,
    )
    loaded = _load_inference_class(
        client, config, system, user, HISTORY_INFERENCE_CLASS, _HISTORY_METHODS
    )
    if loaded is None or not isinstance(loaded[0], HistoryInference):
        return FallbackInference(model=fallback_model)
    return loaded[0]


def validate_history(
    model: CodeWorldModel,
    sampled_history: list[ActionKey],
    observations: list[Observation],
    chance: Sequence[ActionKey] | None = None,
) -> bool:
    """Replay ``sampled_history`` through ``model`` and confirm every
    recorded observation is reproduced.

    What this guarantees: the sample is in the SUPPORT of the posterior --
    no recorded observation is contradicted. What it does not: correct
    DENSITY. The sample need not be at the right place in the posterior,
    only at a place the data does not rule out (PAPER.md s6). Chance nodes
    are sampled with a fixed seed so the check is deterministic; the seed is
    part of the test, not part of the claim.

    Note where the loop stops. Exhausting the action list is not the end of the
    replay: the last operator move still has a chance ply after it, and that
    ply is what resolves the final publish into an observation. Breaking on the
    action list at the top of the loop skipped it, so a model asked to validate
    its OWN recorded history returned False at horizon 10 and True at 6 and 20 --
    a bug that looks exactly like a horizon-dependent modelling weakness and is
    an off-by-one. We continue until the model says terminal, or nothing is left
    to resolve.
    """
    seen: dict[str, Observation] = {}
    rng = random.Random(0)
    pending = list(chance or ())
    cursor = model.initial_state()
    index = 0
    plies = 0
    while plies < 10_000:
        plies += 1
        player = model.get_current_player(cursor)
        if player == TERMINAL_PLAYER:
            break
        if player != CHANCE_PLAYER and index >= len(sampled_history):
            break
        if player == CHANCE_PLAYER:
            outcomes = model.chance_outcomes(cursor)
            if not outcomes:
                break
            # Replay the recorded draw where we have one. Re-rolling here
            # compares the model against a branch that never happened, which is
            # the same defect that made transition tests unmeasurable. The
            # draw rule itself is shared with ``replay_traced`` so the two
            # replay paths cannot drift apart.
            action, _recorded = _draw_recorded_chance(outcomes, pending, rng)
        else:
            action = sampled_history[index]
            index += 1
        cursor = model.apply_action(cursor, action)
        obs = model.get_observations(cursor).get(OPERATOR)
        if obs is not None and obs.utm_content:
            seen[obs.utm_content] = obs
    for recorded in observations:
        if recorded.is_partial:
            continue
        predicted = seen.get(recorded.utm_content)
        if predicted is None:
            return False
        for metric in (
            "reach",
            "hook_rate",
            "hold_rate",
            "attributed_conversions",
            "attribution_coverage",
        ):
            predicted_value = float(getattr(predicted, metric))
            recorded_value = float(getattr(recorded, metric))
            if not _VALIDATION_TOLERANCE.allows(
                metric, predicted_value, recorded_value
            ):
                return False
    return True


def inference_accuracy(
    model: CodeWorldModel,
    inference: StateInference | HistoryInference,
    trajectories: list[Trajectory],
    mode: str = "state",
    tolerance: Tolerance | None = None,
) -> float:
    """Fraction of settled observations the inference+CWM autoencoder
    reproduces.

    ``mode`` is explicit and required in spirit, because sniffing for a
    ``resample_history`` attribute silently mis-dispatched every sampler that
    defines both methods -- ``FallbackInference`` among them -- down the history
    path, where its empty sample failed validation immediately and the function
    returned 0.0 for everything, forever.

    The comparison is on METRICS within ``tolerance``, not on identifier
    equality. Matching ``utm_content`` only asks whether the reconstructed state
    is pointing at the right post, which any state carrying an empty log fails
    and no state carrying the right log can fail informatively. The question
    worth asking is whether replaying the sampled state through the model
    reproduces the numbers that were actually seen.
    """
    from fronts.game.action_space import ActionCodec

    if mode not in {"state", "history"}:
        raise ValueError(f"mode must be 'state' or 'history', got {mode!r}")
    tol = tolerance if tolerance is not None else _VALIDATION_TOLERANCE
    codec = ActionCodec()
    total = 0
    matched = 0
    for trajectory in trajectories:
        history: list[tuple[Observation | None, ActionKey | None]] = []
        for step in trajectory.steps:
            action = codec.encode(step.move)
            recorded = step.observation
            if recorded is not None and not recorded.is_partial:
                total += 1
                try:
                    if mode == "history":
                        sampled = inference.resample_history(history, OPERATOR)  # type: ignore[union-attr]
                        ok = validate_history(
                            model, sampled, [recorded], trajectory.chance
                        )
                    else:
                        state = inference.resample_state(history, OPERATOR)  # type: ignore[union-attr]
                        ok = _reproduces(model, state, recorded, tol)
                except Exception:
                    # A sampler that cannot be asked has missed. The module
                    # contract is fall back, never propagate: a raising
                    # sampler must read as a bad score, not a stack trace.
                    ok = False
                matched += 1 if ok else 0
            history.append((recorded, action))
    return matched / total if total else 0.0


def _reproduces(
    model: CodeWorldModel,
    state: State,
    recorded: Observation,
    tolerance: Tolerance,
) -> bool:
    """Does the model, read at ``state``, reproduce ``recorded`` within
    tolerance? Any failure to read the state at all counts as a miss, because a
    sampler whose output the model cannot interpret is not a working sampler."""
    try:
        predicted = model.get_observations(state).get(OPERATOR)
    except Exception:
        return False
    if predicted is None or predicted.utm_content != recorded.utm_content:
        return False
    return all(
        tolerance.allows(
            metric, float(getattr(predicted, metric)), float(getattr(recorded, metric))
        )
        for metric in ("reach", "impressions", "attributed_conversions", "hook_rate")
    )
