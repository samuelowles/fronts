"""Turn recorded history into executable unit tests.

This is the feedback signal that makes iterative synthesis converge: a
candidate model is judged by the tests generated here, its failures go back
into the prompt as tracebacks, and the loop repeats. The paper's whole
training signal is exactly this -- not gradients, not a reward model, just
"the code you wrote fails these cases you were shown."

Test kinds:

* TRANSITION -- given a prefix and an action, does the predicted observation
  match the recorded one within tolerance.
* LEGALITY -- a move the operator actually made must be legal in the model.
* OBSERVATION_RECONSTRUCTION -- the closed-deck autoencoder: rebuild the
  latent by replaying the action prefix, decode through the model, compare
  the predicted observation to what was seen.
* NO_CRASH -- random legal play for N steps must not raise.
* TERMINATION -- the model must reach a terminal state within the horizon.

Closed deck is the default and it is a real constraint, not a flag. In the
closed-deck setting the agent only ever sees its own observations and
actions, so with ``include_hidden=False`` no test that references hidden
state is emitted -- only observation-reconstruction and no-crash survive.
As ``docs/PAPER.md`` s3 puts it, the paper's closed-deck procedure "drops
every unit test requiring hidden state and keeps only observation -> latent
-> observation reconstruction plus no-crash tests," producing "a kind of
autoencoder, where the inference function acts as an encoder ... and the CWM
acts as a decoder."
"""

from __future__ import annotations

import random
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from blotto.game.action_space import ActionCodec
from blotto.game.types import (
    CHANCE_PLAYER,
    OPERATOR,
    TERMINAL_PLAYER,
    ActionKey,
    Observation,
    State,
    Trajectory,
)
from blotto.protocols import CodeWorldModel

__all__ = [
    "ModelTest",
    "TestResult",
    "Tolerance",
    "TestKind",
    "generate",
    "split",
]


TestKind = str
"""The five kinds above, as plain strings, so a test payload stays JSON-ish
and a failing refinement prompt can quote it without importing anything."""

TRANSITION: TestKind = "transition"
LEGALITY: TestKind = "legality"
OBSERVATION_RECONSTRUCTION: TestKind = "observation_reconstruction"
NO_CRASH: TestKind = "no_crash"
TERMINATION: TestKind = "termination"

_COUNT_METRICS: tuple[str, ...] = (
    "impressions",
    "reach",
    "saves",
    "shares",
    "comments",
    "profile_visits",
    "link_clicks",
    "attributed_conversions",
)
_RATE_METRICS: tuple[str, ...] = (
    "hook_rate",
    "hold_rate",
    "attribution_coverage",
    "incrementality",
    "dark_social_estimate",
)


@dataclass(frozen=True, slots=True)
class Tolerance:
    """Separate relative tolerances for count and rate metrics, because a
    reach estimate 10% off is fine and a hook rate 10% off is not.

    Counts are noisy in proportion (a viral tail dominates any sum of
    impressions); rates are bounded estimators whose whole information
    content is the ratio, so the same relative error means something much
    worse.

    The defaults were 0.25 and 0.10 while replay re-sampled chance, and had to
    be that wide to absorb a +/-50% response bucket the model had no way to
    predict. That was a tolerance concealing a broken measurement rather than
    accommodating real variance. With chance replayed from the recording
    (``Trajectory.chance``) the remaining error is genuinely the model's, so the
    thresholds tighten to where a wrong model actually fails.
    """

    counts: float = 0.05
    rates: float = 0.02

    def allows(self, metric: str, predicted: float, actual: float) -> bool:
        is_rate = metric in _RATE_METRICS
        tol = self.rates if is_rate else self.counts
        # Counts get a max(|actual|, 1) floor so a zero actual does not
        # demand an exactly-zero prediction: near-zero counts are noise, and
        # a suite that fails on noise teaches the model to memorise it.
        # Rates get no floor -- their whole information content is the
        # ratio, so their tolerance is strictly relative.
        scale = abs(actual) if is_rate else max(abs(actual), 1.0)
        return abs(predicted - actual) <= tol * scale


@dataclass(frozen=True, slots=True)
class TestResult:
    """Passed, a human-readable detail line, and the traceback when it
    failed. The traceback is not optional decoration -- it is the exact
    string that goes back into the next synthesis prompt."""

    passed: bool
    detail: str = ""
    traceback: str | None = None


def _metrics(observation: Observation) -> dict[str, float]:
    return {name: float(getattr(observation, name)) for name in _ALL_METRICS}


_ALL_METRICS: tuple[str, ...] = _COUNT_METRICS + _RATE_METRICS


def replay_traced(
    model: CodeWorldModel,
    actions: list[ActionKey],
    seed: int,
    chance: Sequence[ActionKey] | None = None,
    max_plies: int = 10_000,
) -> tuple[State, bool]:
    """Walk ``actions`` through ``model`` and return ``(state, degraded)``.

    Recorded chance outcomes are consumed in temporal order. This is what makes
    a transition test a measurement rather than a coin flip: transitions are
    deterministic given the chance action, so replaying the recorded one
    compares the model's prediction against a recording made under the same
    branch. Drawing a fresh one instead scores the difference between two
    branches as model error, which is how the reference model came to fail 70%
    of tests generated from its own trajectories.

    ``degraded`` is True when the recorded outcomes ran out and a draw had to be
    sampled. A degraded test is still worth running and is not worth trusting,
    so callers surface the flag rather than silently averaging it in.

    A model may legitimately reach chance nodes at different points than the
    recording did -- that is itself a modelling error, and one worth seeing. If
    a recorded outcome is not legal at the node reached, we fall back to
    sampling and mark the test degraded rather than forcing an invalid action.
    """
    rng = random.Random(seed)
    pending = list(chance or ())
    state = model.initial_state()
    index = 0
    plies = 0
    degraded = False
    while plies < max_plies:
        plies += 1
        player = model.get_current_player(state)
        if player == TERMINAL_PLAYER:
            break
        if player == CHANCE_PLAYER:
            outcomes = model.chance_outcomes(state)
            if not outcomes:
                break
            keys = [key for key, _ in outcomes]
            action: ActionKey | None = None
            while pending and action is None:
                candidate = pending.pop(0)
                if candidate in keys:
                    action = candidate
            if action is None:
                degraded = True
                action = rng.choices(
                    keys, weights=[prob for _, prob in outcomes], k=1
                )[0]
            state = model.apply_action(state, action)
            continue
        if index >= len(actions):
            break
        state = model.apply_action(state, actions[index])
        index += 1
    return state, degraded


def replay(
    model: CodeWorldModel,
    actions: list[ActionKey],
    seed: int,
    chance: Sequence[ActionKey] | None = None,
    max_plies: int = 10_000,
) -> State:
    """``replay_traced`` without the degradation flag, for callers that do not
    report it."""
    state, _ = replay_traced(model, actions, seed, chance, max_plies)
    return state


def _without_utm(action: ActionKey) -> str:
    """An action key with its tracking id stripped, for identity comparisons
    where the tracking id is not part of what is being compared."""
    return "|".join(
        part for part in str(action).split("|") if not part.startswith("utm=")
    )


def _operator_observation(model: CodeWorldModel, state: State) -> Observation:
    observations = model.get_observations(state)
    return observations[OPERATOR]


def settle(
    model: CodeWorldModel,
    state: State,
    max_plies: int = 32,
) -> tuple[State, Observation]:
    """Advance until the latest operator observation stops being provisional.

    A recorded trajectory stores what an operator eventually SAW, and what they
    eventually saw is the settled number -- metrics inside the 24-72h reporting
    window are provisional and are excluded from test generation for exactly
    that reason. Replaying only up to the moment of publication therefore reads
    a different quantity than the one recorded: in the reference model the
    provisional reach for a post is 2304 where the settled figure is 705, a
    factor of three that has nothing to do with model quality.

    So the comparison has to be made at the same point in the lifecycle. We
    advance chance plies, and spend operator plies on ``hold`` where one is
    legal, because holding passes time without publishing something new that
    would displace the observation under test.

    Bounded, and gives up quietly: a candidate model that never settles anything
    simply gets compared on what it does report, and fails on the metrics, which
    is the correct outcome for a model that cannot represent a reporting lag.
    """
    observation = _operator_observation(model, state)
    for _ in range(max_plies):
        if not observation.is_partial:
            break
        player = model.get_current_player(state)
        if player == TERMINAL_PLAYER:
            break
        if player == CHANCE_PLAYER:
            outcomes = model.chance_outcomes(state)
            if not outcomes:
                break
            state = model.apply_action(state, max(outcomes, key=lambda o: o[1])[0])
        else:
            legal = model.get_legal_actions(state)
            hold = next((a for a in legal if a.startswith("hold")), None)
            if hold is None:
                break
            state = model.apply_action(state, hold)
        observation = _operator_observation(model, state)
    return state, observation


@dataclass(frozen=True, slots=True)
class ModelTest:
    """One executable assertion about a candidate model.

    ``payload`` carries everything ``run`` needs -- the action prefix, the
    recorded metrics, the seed -- so a test is data, can be serialised into a
    prompt, and can be regenerated deterministically from the trajectory.
    """

    name: str
    kind: TestKind
    payload: dict[str, Any]
    tolerance: Tolerance = field(default_factory=Tolerance)

    def run(self, model: CodeWorldModel) -> TestResult:
        try:
            return self._run(model)
        except Exception:
            return TestResult(
                passed=False,
                detail=f"{self.name} raised",
                traceback=traceback.format_exc(),
            )

    def _run(self, model: CodeWorldModel) -> TestResult:
        if self.kind == NO_CRASH:
            return self._run_no_crash(model)
        if self.kind == OBSERVATION_RECONSTRUCTION:
            return self._run_reconstruction(model)
        if self.kind == TRANSITION:
            return self._run_transition(model)
        if self.kind == LEGALITY:
            return self._run_legality(model)
        if self.kind == TERMINATION:
            return self._run_termination(model)
        return TestResult(passed=False, detail=f"unknown test kind {self.kind!r}")

    def _run_reconstruction(self, model: CodeWorldModel) -> TestResult:
        actions = [ActionKey(a) for a in self.payload["actions"]]
        state, degraded = replay_traced(
            model, actions, self.payload["seed"], self.payload.get("chance")
        )
        state, predicted = settle(model, state)
        recorded = self.payload["expected"]
        if predicted.utm_content != recorded["utm_content"]:
            return TestResult(
                passed=False,
                detail=(
                    f"{self.name}: model's current observation is utm "
                    f"{predicted.utm_content!r}, expected {recorded['utm_content']!r}"
                ),
            )
        failures: list[str] = []
        for metric, value in recorded["metrics"].items():
            got = float(getattr(predicted, metric))
            if not self.tolerance.allows(metric, got, value):
                failures.append(f"{metric}: {got:.4g} vs {value:.4g}")
        if failures:
            return TestResult(
                passed=False, detail=f"{self.name}: {'; '.join(failures)}"
            )
        return TestResult(passed=True, detail=self.name)

    def _run_transition(self, model: CodeWorldModel) -> TestResult:
        prefix = [ActionKey(a) for a in self.payload["prefix"]]
        action = ActionKey(self.payload["action"])
        state, degraded = replay_traced(
            model, [*prefix, action], self.payload["seed"], self.payload.get("chance")
        )
        state, predicted = settle(model, state)
        recorded = self.payload["expected"]
        failures = [
            f"{metric}: {float(getattr(predicted, metric)):.4g} vs {value:.4g}"
            for metric, value in recorded["metrics"].items()
            if not self.tolerance.allows(
                metric, float(getattr(predicted, metric)), value
            )
        ]
        if failures:
            return TestResult(passed=False, detail=f"{self.name}: {'; '.join(failures)}")
        return TestResult(passed=True, detail=self.name)

    def _run_legality(self, model: CodeWorldModel) -> TestResult:
        prefix = [ActionKey(a) for a in self.payload["prefix"]]
        action = ActionKey(self.payload["action"])
        state, degraded = replay_traced(
            model, prefix, self.payload["seed"], self.payload.get("chance")
        )
        legal = model.get_legal_actions(state)
        # Compare modulo the tracking id. A utm is minted per post and is
        # required to be non-empty, but no specific value ever makes a move
        # legal or illegal -- so a model that offers the right move under a
        # different utm has got the rule right, and failing it here would be
        # scoring bookkeeping as understanding.
        if _without_utm(action) not in {_without_utm(a) for a in legal}:
            return TestResult(
                passed=False,
                detail=(
                    f"{self.name}: recorded action {action!r} is illegal in "
                    f"the model ({len(legal)} legal actions at that state)"
                ),
            )
        return TestResult(passed=True, detail=self.name)

    def _run_no_crash(self, model: CodeWorldModel) -> TestResult:
        rng = random.Random(self.payload["seed"])
        state = model.initial_state()
        plies = 0
        while plies < self.payload["steps"]:
            plies += 1
            player = model.get_current_player(state)
            if player == TERMINAL_PLAYER:
                break
            if player == CHANCE_PLAYER:
                outcomes = model.chance_outcomes(state)
                if not outcomes:
                    break
                action = rng.choices(
                    [key for key, _ in outcomes],
                    weights=[prob for _, prob in outcomes],
                    k=1,
                )[0]
            else:
                legal = model.get_legal_actions(state)
                if not legal:
                    break
                action = rng.choice(legal)
            state = model.apply_action(state, action)
        return TestResult(passed=True, detail=f"{self.name}: survived {plies} plies")

    def _run_termination(self, model: CodeWorldModel) -> TestResult:
        rng = random.Random(self.payload["seed"])
        state = model.initial_state()
        plies = 0
        while plies < self.payload["max_plies"]:
            player = model.get_current_player(state)
            if player == TERMINAL_PLAYER:
                return TestResult(passed=True, detail=f"{self.name}: terminal at ply {plies}")
            if player == CHANCE_PLAYER:
                outcomes = model.chance_outcomes(state)
                if not outcomes:
                    break
                action = rng.choices(
                    [key for key, _ in outcomes],
                    weights=[prob for _, prob in outcomes],
                    k=1,
                )[0]
            else:
                legal = model.get_legal_actions(state)
                if not legal:
                    break
                action = rng.choice(legal)
            state = model.apply_action(state, action)
            plies += 1
        return TestResult(
            passed=False,
            detail=f"{self.name}: no terminal state within {self.payload['max_plies']} plies",
        )


def generate(
    trajectory: Trajectory,
    tolerance: Tolerance | None = None,
    include_hidden: bool = False,
) -> list[ModelTest]:
    """Generate tests from one recorded trajectory.

    ``include_hidden=False`` is the closed-deck default: only
    observation-reconstruction and no-crash tests, per PAPER.md s3. The
    open-deck variants (transition, legality, termination) are emitted only
    when the caller can legitimately see hidden state -- offline logs that
    recorded it, which organic distribution never has.
    """
    tolerance = tolerance if tolerance is not None else Tolerance()
    codec = ActionCodec()
    actions = [codec.encode(step.move) for step in trajectory.steps]
    # The whole recorded chance sequence goes into every payload. Replay
    # consumes it in order and stops when the prefix is exhausted, so each test
    # sees exactly the draws that occurred before its own decision point.
    chance = list(trajectory.chance)
    tests: list[ModelTest] = []

    for index, step in enumerate(trajectory.steps):
        observation = step.observation
        # Partial observations must never become test cases. A metric read
        # inside the 24-72h reporting window is provisional; testing against
        # it teaches the model to predict noise, exactly the failure the
        # evidence gates exist to prevent on the acting side (PAPER.md s5).
        if observation is None or observation.is_partial:
            continue
        metrics = _metrics(observation)
        if include_hidden:
            tests.append(
                ModelTest(
                    name=f"transition[{index}]",
                    kind=TRANSITION,
                    payload={
                        "prefix": actions[:index],
                        "action": actions[index],
                        "expected": {
                            "utm_content": observation.utm_content,
                            "metrics": metrics,
                        },
                        "seed": index,
                        "chance": chance,
                    },
                    tolerance=tolerance,
                )
            )
            tests.append(
                ModelTest(
                    name=f"legality[{index}]",
                    kind=LEGALITY,
                    payload={
                        "prefix": actions[:index],
                        "action": actions[index],
                        "seed": index,
                        "chance": chance,
                    },
                    tolerance=tolerance,
                )
            )
        tests.append(
            ModelTest(
                name=f"reconstruction[{index}]",
                kind=OBSERVATION_RECONSTRUCTION,
                payload={
                    "actions": actions[: index + 1],
                    "expected": {
                        "utm_content": observation.utm_content,
                        "metrics": metrics,
                    },
                    "seed": index,
                        "chance": chance,
                },
                tolerance=tolerance,
            )
        )

    if trajectory.steps:
        tests.append(
            ModelTest(
                name="no_crash",
                kind=NO_CRASH,
                payload={"steps": 2 * len(trajectory.steps) + 10, "seed": 13, "chance": chance},
                tolerance=tolerance,
            )
        )
        if include_hidden:
            tests.append(
                ModelTest(
                    name="termination",
                    kind=TERMINATION,
                    payload={
                        "max_plies": 4 * len(trajectory.steps) + 20,
                        "seed": 17,
                        "chance": chance,
                    },
                    tolerance=tolerance,
                )
            )
    return tests


def split(
    tests: list[ModelTest], train_fraction: float, seed: int
) -> tuple[list[ModelTest], list[ModelTest]]:
    """Split tests into train and held-out test sets.

    Held-out evaluation is the whole diagnostic: the paper's Gin rummy
    failure is visible only because train (0.78) and test (0.75) were
    reported separately. Shuffled with a seed so a split is reproducible --
    an unseeded split lets a lucky shuffle flatter a bad model."""
    rng = random.Random(seed)
    shuffled = list(tests)
    rng.shuffle(shuffled)
    cut = int(round(train_fraction * len(shuffled)))
    return shuffled[:cut], shuffled[cut:]
