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
from dataclasses import dataclass, field
from typing import Any

from blotto.game.action_space import ActionCodec
from blotto.game.types import (
    CHANCE_PLAYER,
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
    worse. Defaults: counts 0.25, rates 0.10.
    """

    counts: float = 0.25
    rates: float = 0.10

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


def replay(
    model: CodeWorldModel,
    actions: list[ActionKey],
    seed: int,
    max_plies: int = 10_000,
) -> State:
    """Walk ``actions`` through ``model``, sampling chance nodes with a
    seeded rng, and return the state after the last one.

    Chance draws cannot be replayed from the trajectory -- which moves the
    chance player made is invisible to the operator, and that is the point of
    closed deck. The seed makes a test reproducible without making it
    correct: a model can still land on a different bucket than the recording
    did, which is why transition tests carry tolerance and the pass rate,
    not the pass, is the signal.
    """
    rng = random.Random(seed)
    state = model.initial_state()
    index = 0
    plies = 0
    while plies < max_plies:
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
            state = model.apply_action(state, action)
            continue
        if index >= len(actions):
            break
        state = model.apply_action(state, actions[index])
        index += 1
    return state


def _operator_observation(model: CodeWorldModel, state: State) -> Observation:
    observations = model.get_observations(state)
    return observations[0]


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
        state = replay(model, actions, self.payload["seed"])
        predicted = _operator_observation(model, state)
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
        state = replay(model, [*prefix, action], self.payload["seed"])
        predicted = _operator_observation(model, state)
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
        state = replay(model, prefix, self.payload["seed"])
        legal = model.get_legal_actions(state)
        if action not in legal:
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
                },
                tolerance=tolerance,
            )
        )

    if trajectory.steps:
        tests.append(
            ModelTest(
                name="no_crash",
                kind=NO_CRASH,
                payload={"steps": 2 * len(trajectory.steps) + 10, "seed": 13},
                tolerance=tolerance,
            )
        )
        if include_hidden:
            tests.append(
                ModelTest(
                    name="termination",
                    kind=TERMINATION,
                    payload={"max_plies": 4 * len(trajectory.steps) + 20, "seed": 17},
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
