"""The interfaces that synthesised code must satisfy.

Everything in this file is a boundary. On one side sits code we wrote; on the
other sits Python an LLM produced from a natural-language brief and a handful of
observed trajectories. These protocols are the entire contract between them, and
they are also, per the paper, the regulariser that stops the synthesiser from
inventing a degenerate representation:

    "Instead of a bottleneck, or a regularization term, the game rules and the
    required OpenSpiel API (used in the unit tests) introduced in the context of
    the LLM act as regularizers to prevent trivial latent spaces from being
    discovered."
        (Lehrach et al., Code World Models for General Game Playing, s4.4)

Keep them narrow. Every method added here is a method the synthesiser can get
wrong, and every method removed is a way for it to cheat.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from fronts.game.types import ActionKey, Observation, State

__all__ = [
    "CodeWorldModel",
    "HistoryInference",
    "StateInference",
    "ValueFunction",
    "Planner",
    "PlanResult",
    "ModelReport",
]


@runtime_checkable
class CodeWorldModel(Protocol):
    """A playable, approximate copy of the distribution environment.

    Deliberately shaped like OpenSpiel, for the reason the paper gives: the
    synthesiser has seen a great deal of OpenSpiel and very little of us. Naming
    the methods what it already expects them to be called removes an entire
    class of synthesis error for free.

    All methods are deterministic. Randomness enters only through the chance
    player. The paper is strict about this and so are we, because a model with
    hidden nondeterminism cannot be unit-tested against a recorded trajectory.
    """

    def initial_state(self) -> State:
        """Return the opening state. Must be deterministic."""
        ...

    def apply_action(self, state: State, action: ActionKey) -> State:
        """Return the successor state. Must not mutate ``state``.

        Non-mutation is not stylistic. The planner holds many states from one
        subtree simultaneously; an in-place update corrupts siblings and the
        resulting bug is close to undiagnosable from a win rate.
        """
        ...

    def get_current_player(self, state: State) -> int:
        """Return the player to act.

        ``CHANCE_PLAYER`` (-1) when the next event is a chance draw:
        audience arrival, the virality lottery, attribution noise.
        ``TERMINAL_PLAYER`` (-4) when the horizon is reached.
        """
        ...

    def get_legal_actions(self, state: State) -> list[ActionKey]:
        """Enumerate every legal move, and nothing else.

        This method is the paper's verifiability claim made concrete, and in
        this domain it carries real weight. An illegal move here is an
        unsubstantiated health claim, an undisclosed paid endorsement, or a
        budget commitment made on five conversions. The planner can only
        select from what this returns, so anything this method excludes is a
        mistake the system cannot make.
        """
        ...

    def get_observations(self, state: State) -> dict[int, Observation]:
        """Return each player's observation of ``state``.

        The operator's entry must reflect real degradation: reporting lag,
        attribution coverage below 1.0, and non-incremental conversions. A model
        that hands the operator clean numbers has quietly changed the game to an
        easier one, and will plan beautifully for a world that does not exist.
        """
        ...

    def get_rewards(self, state: State) -> dict[int, float]:
        """Return per-player reward for ``state``.

        The operator's reward is contribution margin from paying users. It is
        not reach, not engagement, and not attributed conversions taken at face
        value. See ``fronts.game.payoff``.
        """
        ...

    def chance_outcomes(self, state: State) -> list[tuple[ActionKey, float]]:
        """Return ``(outcome, probability)`` pairs summing to 1.0.

        Only called when ``get_current_player`` returns ``CHANCE_PLAYER``.
        """
        ...


@runtime_checkable
class HistoryInference(Protocol):
    """Sample a plausible hidden history from the operator's own observations.

    The open-deck variant of the paper's inference-as-code. Exact posterior
    inference is exponential in the worst case, so the synthesiser writes an
    approximate sampler instead, and correctness is established structurally:
    replaying the sampled history through the CWM must reproduce every
    observation actually seen.

    The guarantee places the sample inside the posterior's support without
    placing it at the right density, and the paper's argument for why support
    membership is enough applies to distribution:

        "Although this does not guarantee that s_t is correctly distributed, the
         correct support is already very informative, given the extremely sparse
         support of state posteriors in games."
    """

    def resample_history(
        self,
        obs_action_history: list[tuple[Observation | None, ActionKey | None]],
        player_id: int,
    ) -> list[ActionKey]:
        """Return a full action sequence consistent with what the player saw."""
        ...


@runtime_checkable
class StateInference(Protocol):
    """Sample the hidden state directly. The closed-deck variant.

    Cheaper and less safe than history inference: it ignores the dependency
    between consecutive states, so it can produce a state that is not reachable
    at all. We use it anyway, because closed deck is the accurate description
    of organic distribution. You see your analytics and nothing else; the
    ranking function and your competitors' test schedules stay hidden.
    """

    def resample_state(
        self,
        obs_action_history: list[tuple[Observation | None, ActionKey | None]],
        player_id: int,
    ) -> State:
        """Return one plausible hidden state given only the player's own view."""
        ...


@runtime_checkable
class ValueFunction(Protocol):
    """Heuristic state value, for use at planning leaves.

    Synthesised rather than fitted, because there is no ground truth to fit
    against. The paper's remedy is selection rather than training: generate
    several candidates and run a tournament between the agents that use them.
    We do the same, in ``fronts.cwm.value``.
    """

    def __call__(self, state: State, player: int) -> float: ...


@runtime_checkable
class Planner(Protocol):
    """Turns compute into decisions inside a world model.

    The load-bearing sentence of the entire paper is that this is where strength
    comes from: shift the LLM's job from producing a good policy to producing a
    good model, and let search convert compute into performance. A planner here
    never calls a language model. If it does, the architecture has collapsed
    back into the thing we replaced.
    """

    def plan(
        self,
        model: CodeWorldModel,
        state: State,
        player: int,
        budget: int,
    ) -> PlanResult:
        """Return the chosen move plus its supporting statistics."""
        ...


class PlanResult(Protocol):
    """What a planner hands back. Statistics are part of the contract.

    An operator asked to spend a day's output on a plan is owed the visit counts
    and value estimates behind it, not just the recommendation. A planner that
    reports only its choice is not auditable and should not be trusted with a
    budget.
    """

    @property
    def move(self) -> ActionKey: ...

    @property
    def value(self) -> float: ...

    @property
    def visits(self) -> dict[ActionKey, int]: ...

    @property
    def values(self) -> dict[ActionKey, float]: ...


class ModelReport(Protocol):
    """Quality metrics for a synthesised model.

    Reported separately for train, held-out test and online play, as the paper
    does, because the gap between them is the interesting quantity. The paper's
    own Gin rummy result (0.78 train, 0.75 test transition accuracy, 500 LLM
    calls, budget exhausted) is the failure case worth being able to show, and
    a system that cannot report a number that bad is hiding something.
    """

    @property
    def transition_accuracy(self) -> float: ...

    @property
    def inference_accuracy(self) -> float | None:
        """``None`` means not measured. It is a gap, never a zero."""
        ...

    @property
    def llm_calls(self) -> int: ...

    @property
    def passed_tests(self) -> int: ...

    @property
    def total_tests(self) -> int: ...
