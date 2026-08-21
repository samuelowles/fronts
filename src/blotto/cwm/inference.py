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
from dataclasses import dataclass

from blotto.cwm.llm import LLMClient, extract_code
from blotto.cwm.sandbox import (
    Sandbox,
    SandboxConfig,
    check_protocol_methods,
)
from blotto.cwm.synth import SynthConfig
from blotto.cwm.tests_from_traj import Tolerance
from blotto.game.types import (
    CHANCE_PLAYER,
    TERMINAL_PLAYER,
    ActionKey,
    Observation,
    State,
    Trajectory,
)
from blotto.protocols import CodeWorldModel, HistoryInference, StateInference

__all__ = [
    "STATE_INFERENCE_CLASS",
    "HISTORY_INFERENCE_CLASS",
    "FallbackInference",
    "synthesise_state_inference",
    "synthesise_history_inference",
    "validate_history",
    "inference_accuracy",
]


STATE_INFERENCE_CLASS = "StateInferenceSampler"
HISTORY_INFERENCE_CLASS = "HistoryInferenceSampler"

_STATE_METHODS = {"resample_state": 2}
_HISTORY_METHODS = {"resample_history": 2}

# Validation compares model-reproduced observations to recorded ones at the
# same standard the unit tests use; stricter than this rejects honest
# approximations, looser admits samples that contradict the data.
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
    from blotto.cwm.synth import _serialise_trajectory

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


def _load_inference_class(
    client: LLMClient,
    config: SynthConfig,
    system: str,
    user: str,
    class_name: str,
    methods: dict[str, int],
) -> object | None:
    """Complete, extract, sandbox, verify. None on any failure -- the caller
    falls back rather than propagating, per the module contract."""
    try:
        response = client.complete(system, user)
        namespace = Sandbox().load(extract_code(response), SandboxConfig())
        cls = getattr(namespace, class_name, None)
        if cls is None:
            return None
        instance = cls()
        if check_protocol_methods(instance, methods, class_name):
            return None
        return instance
    except Exception:
        return None


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
    instance = _load_inference_class(
        client, config, system, user, STATE_INFERENCE_CLASS, _STATE_METHODS
    )
    if instance is None:
        return FallbackInference(model=fallback_model)
    return instance  # type: ignore[return-value]


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
    instance = _load_inference_class(
        client, config, system, user, HISTORY_INFERENCE_CLASS, _HISTORY_METHODS
    )
    if instance is None:
        return FallbackInference(model=fallback_model)
    return instance  # type: ignore[return-value]


def validate_history(
    model: CodeWorldModel,
    sampled_history: list[ActionKey],
    observations: list[Observation],
) -> bool:
    """Replay ``sampled_history`` through ``model`` and confirm every
    recorded observation is reproduced.

    What this guarantees: the sample is in the SUPPORT of the posterior --
    no recorded observation is contradicted. What it does not: correct
    DENSITY. The sample need not be at the right place in the posterior,
    only at a place the data does not rule out (PAPER.md s6). Chance nodes
    are sampled with a fixed seed so the check is deterministic; the seed is
    part of the test, not part of the claim.
    """
    seen: dict[str, Observation] = {}
    rng = random.Random(0)
    cursor = model.initial_state()
    index = 0
    plies = 0
    while plies < 10_000:
        plies += 1
        player = model.get_current_player(cursor)
        if player == TERMINAL_PLAYER or index >= len(sampled_history):
            break
        if player == CHANCE_PLAYER:
            outcomes = model.chance_outcomes(cursor)
            if not outcomes:
                break
            action = rng.choices(
                [key for key, _ in outcomes],
                weights=[prob for _, prob in outcomes],
                k=1,
            )[0]
        else:
            action = sampled_history[index]
            index += 1
        cursor = model.apply_action(cursor, action)
        obs = model.get_observations(cursor).get(0)
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
) -> float:
    """Fraction of settled observations the inference+CWM autoencoder
    reproduces.

    For a ``StateInference``: resample a state from the history up to each
    step, read the model's observation there, compare to what was recorded.
    For a ``HistoryInference``: resample a full history and validate it.
    Either way the unit is the observation, because the observation is the
    only thing both sides can see."""
    from blotto.game.action_space import ActionCodec

    codec = ActionCodec()
    total = 0
    matched = 0
    for trajectory in trajectories:
        history: list[tuple[Observation | None, ActionKey | None]] = []
        for step in trajectory.steps:
            action = codec.encode(step.move)
            if step.observation is not None and not step.observation.is_partial:
                total += 1
                if isinstance(inference, HistoryInference) or hasattr(
                    inference, "resample_history"
                ):
                    sampled = inference.resample_history(history, 0)  # type: ignore[attr-defined]
                    ok = validate_history(model, sampled, [step.observation])
                else:
                    state = inference.resample_state(history, 0)  # type: ignore[attr-defined]
                    predicted = model.get_observations(state).get(0)
                    ok = (
                        predicted is not None
                        and predicted.utm_content == step.observation.utm_content
                    )
                matched += 1 if ok else 0
            history.append((step.observation, action))
    return matched / total if total else 0.0
