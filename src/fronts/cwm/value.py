"""Synthesised value functions, selected by tournament rather than refined.

A value function scores a state for planning at leaf depth. There is no
ground truth to refine one against -- no oracle says "this state was worth
0.63" -- so the paper's remedy is selection, not training: generate several
candidates one-shot and run a tournament between the agents that use them,
keeping whichever agent wins. That is why nothing in this module feeds a
value function's failures back into a prompt: there is nothing to feed back.
The candidate either survived the arena or it did not.

Returning no value function at all is a first-class outcome. The paper found
value functions helped in only two of its ten games; in the rest the plain
planner did as well or better. A heuristic that has not EARNED its place in
the tournament is not a tiebreaker, it is noise with a decimal point.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from fronts.cwm.arena import Agent, play_episode
from fronts.cwm.llm import LLMClient, extract_code
from fronts.cwm.sandbox import Sandbox, SandboxConfig
from fronts.cwm.synth import SynthConfig
from fronts.protocols import CodeWorldModel, ValueFunction

__all__ = [
    "VALUE_FUNCTION_CLASS",
    "synthesise_value_functions",
    "tournament",
    "select_best",
]


VALUE_FUNCTION_CLASS = "HeuristicValue"

PlannerFactory = Callable[[ValueFunction], Agent]
"""Builds the agent that plans with a given value function. The arena then
measures the value function by what its agent achieves, which is the only
measurement available in the absence of ground truth."""


def synthesise_value_functions(
    client: LLMClient,
    config: SynthConfig,
    rules: str,
    n: int,
) -> list[ValueFunction]:
    """Generate ``n`` candidate value functions one-shot.

    Failures are skipped, not retried: without a feedback signal a retry is
    the same bet placed twice, and the tournament needs candidates, not
    persistence."""
    system = (
        "You are a program synthesiser. You write heuristic state-value "
        "functions for a content-distribution planning problem, as Python "
        "classes. You write code, not explanations."
    )
    survivors: list[ValueFunction] = []
    for attempt in range(n):
        user = (
            f"Write a Python module defining a class named "
            f"{VALUE_FUNCTION_CLASS} (no constructor arguments) whose "
            "instances are callable as __call__(self, state: dict, player: "
            "int) -> float, scoring how good ``state`` is for ``player``. "
            "Higher is better for the operator (player 0).\n\n"
            "== GAME RULES ==\n"
            + rules.strip()
            + "\n\nVary your approach: this is candidate "
            f"{attempt + 1} of {n}, and the tournament keeps only the best. "
            "Respond with a single fenced Python code block."
        )
        try:
            response = client.complete(system, user)
            namespace = Sandbox().load(extract_code(response), SandboxConfig())
            cls = getattr(namespace, VALUE_FUNCTION_CLASS, None)
            if cls is None:
                continue
            instance = cls()
            if not callable(instance):
                continue
            survivors.append(instance)
        except Exception:
            continue
    return survivors


def tournament(
    models: Sequence[CodeWorldModel],
    value_functions: Sequence[ValueFunction],
    planner_factory: PlannerFactory,
    matches: int,
    rng: random.Random,
) -> list[tuple[ValueFunction, float]]:
    """Round-robin between agents built on each value function, on every
    host model in turn. Returns (value function, mean utility), best first.

    Value functions are NOT refined, because there is no ground truth to
    refine against; they are SELECTED by playing agents that use them
    against each other (Lehrach et al. 2025, s5.3 -- the value-function
    tournament). The host models supply the environment the agents plan
    inside; a heuristic that only wins on one host was overfit to that
    host's quirks, which is precisely what rotating hosts catches."""
    agents = [planner_factory(vf) for vf in value_functions]
    scores = [0.0] * len(agents)
    games = [0] * len(agents)
    base_seed = rng.randrange(2**32)
    for host_index, host in enumerate(models):
        for i in range(len(agents)):
            for j in range(len(agents)):
                if i == j:
                    continue
                for game in range(matches):
                    seed = (
                        base_seed + host_index * 1_000_003 + i * 10_007 + j * 101 + game
                    ) % (2**32)
                    scores[i] += play_episode(host, agents[i], seed)
                    games[i] += 1
    averaged = sorted(
        (
            (vf, scores[i] / games[i] if games[i] else 0.0)
            for i, vf in enumerate(value_functions)
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return averaged


def select_best(
    scored: list[tuple[ValueFunction, float]],
    baseline: float | None = None,
    min_edge: float = 0.0,
) -> ValueFunction | None:
    """Return the best value function, or None when none earned its place.

    ``baseline`` is the planner's mean utility with NO value function; a
    candidate must beat it by ``min_edge`` to be adopted. Returning None is
    legitimate and often right: the paper's value functions helped in only
    two of ten games, and a heuristic that does not clear the baseline is a
    regression wearing a tiebreaker's clothes."""
    if not scored:
        return None
    best_function, best_score = scored[0]
    # Without a no-value-function baseline there is nothing to be worse than,
    # so the best candidate wins by default; with one, it must clear it.
    floor = baseline if baseline is not None else float("-inf")
    if best_score > floor + min_edge:
        return best_function
    return None
