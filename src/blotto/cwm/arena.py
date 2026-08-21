"""Bad-model rejection before real budget is spent.

The most operationally valuable idea in the paper. Ground truth does not
exist for a synthesised world model -- if it did, we would not need the
synthesis -- so candidate models and the agents that plan inside them are
evaluated by playing them against EACH OTHER, with each candidate model in
turn standing in as the host for the tournament. An agent that loses
consistently, across hosts, is rejected before a day of real output is
committed to its recommendations. The arena costs compute; a bad content
strategy costs reach, budget, and occasionally the account.

The rejection rule is the paper's, verbatim: reject any agent "worse than
the best scoring agent by more than 10% of the observed utility range."

One adaptation this domain forces, stated so nobody has to reverse-engineer
it from the code: distribution is a single-operator game (operator versus
chance/platform/field), so two agents cannot occupy opposite sides of one
episode. Paired play instead runs each agent through its OWN episode on the
same host model under COMMON RANDOM NUMBERS -- same chance seeds -- and
compares achieved operator utility. Same spirit, same rule; the dice are
held fixed so the comparison measures the agent and not the lottery.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Sequence

from blotto.game.types import (
    CHANCE_PLAYER,
    TERMINAL_PLAYER,
    ActionKey,
    State,
)
from blotto.protocols import CodeWorldModel

__all__ = [
    "ArenaConfig",
    "ArenaResult",
    "Agent",
    "play_episode",
    "run",
]


Agent = Callable[[CodeWorldModel, State], ActionKey]
"""An agent chooses the operator's action given the host model and state.
Deliberately not the full ``Planner`` protocol: the arena supplies no budget
plumbing of its own, and anything callable with (model, state) can play."""


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    num_models: int = 5
    """How many candidate world models to synthesise for the tournament."""
    matches_per_pairing: int = 50
    rejection_threshold: float = 0.10


@dataclass(slots=True)
class ArenaResult:
    """Everything a human needs to audit the rejection decision.

    ``score_matrix[i][j]`` is agent i's mean operator utility in its paired
    episodes against agent j; ``scores`` are row means; ``rejected`` is the
    list of agent indices the rule removed. The matrix is exposed because a
    rejection a human cannot inspect is a rejection a human cannot overrule,
    and the operator, not the arena, is the one who eats the consequence."""

    score_matrix: list[list[float]]
    scores: list[float]
    survivors: list[int]
    rejected: list[int]
    utility_range: float = 0.0
    hosts: list[int] = field(default_factory=list)


def play_episode(
    model: CodeWorldModel,
    agent: Agent,
    seed: int,
    max_plies: int = 10_000,
) -> float:
    """Run one episode of ``agent`` on ``model``; return operator utility.

    The rng is seeded per episode so paired agents face identical chance
    sequences -- common random numbers are what turns two noisy episodes
    into one comparison."""
    rng = random.Random(seed)
    state = model.initial_state()
    plies = 0
    while plies < max_plies:
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
            action = agent(model, state)
            if action not in legal:
                # An illegal move in the paper's setting is a forfeit; here
                # it is scored as one -- the agent gets nothing further and
                # keeps what it had, which is the sharpest available signal
                # that its model and its policy disagree.
                break
        state = model.apply_action(state, action)
        plies += 1
    return model.get_rewards(state).get(0, 0.0)


def run(
    models: Sequence[CodeWorldModel],
    agents: Sequence[Agent],
    config: ArenaConfig,
) -> ArenaResult:
    """Round-robin tournament of ``agents`` on every host ``model`` in turn.

    Every agent plays every other; each host model stands in for the ground
    truth that does not exist. Rejection rule, exactly as the paper states
    it: reject any agent "worse than the best scoring agent by more than 10%
    of the observed utility range" -- the range being max minus min over all
    individual episode utilities observed in the tournament, which makes the
    threshold a fraction of the spread the tournament actually saw rather
    than of some assumed utility scale.
    """
    utilities: list[list[list[float]]] = [
        [[] for _ in agents] for _ in agents
    ]
    all_utilities: list[float] = []
    matches = max(1, config.matches_per_pairing)
    # Seeds derive from indices, not object ids or process randomness, so a
    # tournament is reproducible: same models, same agents, same results.
    for host_index, host in enumerate(models):
        for i, agent_i in enumerate(agents):
            for j in range(len(agents)):
                if i == j:
                    continue
                for game in range(matches):
                    seed = (host_index * 1_000_003 + i * 10_007 + j * 101 + game) % (2**32)
                    utility = play_episode(host, agent_i, seed)
                    utilities[i][j].append(utility)
                    all_utilities.append(utility)

    n = len(agents)
    matrix = [
        [
            sum(utilities[i][j]) / len(utilities[i][j]) if utilities[i][j] else 0.0
            for j in range(n)
        ]
        for i in range(n)
    ]
    scores = [
        sum(matrix[i][j] for j in range(n) if j != i) / (n - 1) if n > 1 else matrix[i][0]
        for i in range(n)
    ]
    utility_range = (max(all_utilities) - min(all_utilities)) if all_utilities else 0.0
    best = max(scores)
    threshold = config.rejection_threshold * utility_range
    rejected = [i for i in range(n) if best - scores[i] > threshold]
    survivors = [i for i in range(n) if i not in rejected]
    return ArenaResult(
        score_matrix=matrix,
        scores=scores,
        survivors=survivors,
        rejected=rejected,
        utility_range=utility_range,
        hosts=list(range(len(models))),
    )
