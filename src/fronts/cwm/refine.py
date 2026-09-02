"""Thompson-sampled tree search over candidate models -- the paper's REx.

Synthesis is not one shot. The paper holds several candidate models
simultaneously, and after each round chooses which to refine next by
Thompson sampling: every node draws from a Beta posterior over its pass
rate, and the highest draw is refined. The constants below are the paper's
and are not ours to tune.

The sampling rule, quoted from the paper's procedure: each node draws with

    alpha = 1 + C * h,   beta = 1 + (1 - h) * C,   C = 5.0

where h is the average unit-test pass rate -- "favoring those that either
have high transition accuracy or have been refined few times" (a fresh node
with h = 0.5 has a wider posterior than a heavily-refined one at the same
mean, so exploration is built into the prior rather than bolted on as an
epsilon).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from fronts.cwm.llm import LLMClient
from fronts.cwm.sandbox import Sandbox, SandboxConfig, instantiate
from fronts.cwm.synth import WORLD_MODEL_CLASS, SynthConfig, synthesise
from fronts.cwm.tests_from_traj import ModelTest, TestResult
from fronts.game.types import Trajectory
from fronts.protocols import CodeWorldModel

__all__ = [
    "RefineConfig",
    "RefinementNode",
    "RefinementTree",
    "refine",
    "best",
]


@dataclass(frozen=True, slots=True)
class RefineConfig:
    """REx constants, from the paper. Do not change them."""

    heuristic_weight: float = 5.0
    """C in the Beta prior."""
    num_retries: int = 500
    """Refinement attempts before giving up -- the synthesis budget."""
    num_tests_on_init: int = 5
    num_tests_on_error: int = 1
    min_heuristic_value_on_init: float = 0.01
    """A candidate below this pass rate is never selected for refinement:
    too wrong to be worth a call."""
    min_heuristic_value_gain: float = 0.01
    """A refinement must improve on its parent by at least this much to stay
    in the tree."""
    seed: int = 11


@dataclass(slots=True)
class RefinementNode:
    """One candidate model in the refinement tree."""

    source: str
    h: float
    """Average unit-test pass rate, in [0, 1]. Named h for the paper's
    heuristic value."""
    refinements: int = 0
    """How many times this node has been selected for refinement."""
    parent: RefinementNode | None = None
    children: list[RefinementNode] = field(default_factory=list)
    model: CodeWorldModel | None = None
    """Cached instance, so a node is sandbox-loaded once, not once per
    selection round."""


@dataclass(slots=True)
class RefinementTree:
    """The set of candidates, plus what the prompts need to rebuild them."""

    nodes: list[RefinementNode] = field(default_factory=list)
    rules_text: str = ""
    trajectories: list[Trajectory] = field(default_factory=list)
    heuristic_weight: float = 5.0
    min_heuristic_value: float = 0.01

    def select(self, rng: random.Random) -> RefinementNode:
        """Thompson sampling: draw Beta(1 + C*h, 1 + (1-h)*C) per node, pick
        the argmax.

        Nodes below ``min_heuristic_value`` are excluded unless nothing
        qualifies, so a hopeless first candidate does not eat the budget --
        but an all-hopeless tree still selects, because refusing to choose is
        not a strategy the paper provides for.
        """
        eligible = [n for n in self.nodes if n.h >= self.min_heuristic_value]
        pool = eligible or self.nodes
        c = self.heuristic_weight

        def draw(node: RefinementNode) -> float:
            alpha = 1.0 + c * node.h
            beta = 1.0 + (1.0 - node.h) * c
            return rng.betavariate(alpha, beta)

        return max(pool, key=draw)


def _evaluate(
    node: RefinementNode, tests: list[ModelTest]
) -> tuple[float, list[tuple[ModelTest, TestResult]]]:
    """(Re)run ``tests`` against a node's model. Returns the pass rate and
    the failing tests with their results, whose tracebacks become the
    refinement prompt."""
    if node.model is None:
        namespace = Sandbox().load(node.source, SandboxConfig())
        node.model = instantiate(namespace, WORLD_MODEL_CLASS)
    failures = [
        (test, result)
        for test in tests
        if not (result := test.run(node.model)).passed
    ]
    passed = len(tests) - len(failures)
    return (passed / len(tests) if tests else 0.0), failures


def refine(
    client: LLMClient,
    tree: RefinementTree,
    tests: list[ModelTest],
    config: RefineConfig,
) -> RefinementTree:
    """Refine candidates until one passes everything or the budget is gone.

    Each round: select a node, run its tests, feed the first failure's
    traceback back as the refinement prompt, and add the child only if it
    clears ``min_heuristic_value_gain`` over its parent -- the paper keeps
    refinements that improve and discards the rest, spending the call either
    way, which is what makes the retry budget a real budget.

    The tree's sampling knobs (``heuristic_weight``,
    ``min_heuristic_value``) are taken from ``config`` rather than left at
    the tree's own defaults: the config is the object callers tune, and a
    tuned value that selection silently ignored would be a knob that lies.
    """
    rng = random.Random(config.seed)
    tree.heuristic_weight = config.heuristic_weight
    tree.min_heuristic_value = config.min_heuristic_value_on_init
    synth_config = SynthConfig(
        num_tests_on_init=config.num_tests_on_init,
        num_tests_on_error=config.num_tests_on_error,
    )
    for _ in range(config.num_retries):
        if any(node.h >= 1.0 for node in tree.nodes):
            break
        selected = tree.select(rng)
        h, failures = _evaluate(selected, tests)
        selected.h = h
        if not failures:
            break
        first_failure, first_result = failures[0]
        error_text = first_result.traceback or first_result.detail or first_failure.name
        result = synthesise(
            client,
            synth_config,
            tree.rules_text,
            tree.trajectories,
            tests,
            previous_error=error_text,
        )
        selected.refinements += 1
        if result.model is None:
            continue
        if result.pass_rate >= selected.h + config.min_heuristic_value_gain:
            child = RefinementNode(
                source=result.source,
                h=result.pass_rate,
                parent=selected,
                model=result.model,
            )
            selected.children.append(child)
            tree.nodes.append(child)
    return tree


def best(tree: RefinementTree) -> RefinementNode:
    """The node to deploy: highest pass rate, fewest refinements on ties --
    a simpler candidate that scores the same is the better artefact."""
    return min(tree.nodes, key=lambda node: (-node.h, node.refinements))
