"""Information Set Monte Carlo Tree Search.

Origin: Cowling, Powley and Whitehouse, "Information Set Monte Carlo Tree
Search", IEEE Transactions on Computational Intelligence and AI in Games 4(3),
2012. Their problem is ours in miniature: a player who sees only part of the
state must still commit to one move, and the honest way to do that is to
search over determinizations -- complete states sampled to be consistent with
everything the player has actually observed.

Two design decisions are load-bearing enough to state up front.

First, tree nodes correspond to *information sets*, not states. The tree
branches only on the searching player's own actions; opponent decisions and
chance events are resolved inside the current determinization during descent
and never create nodes. This is what makes the tree a representation of the
player's knowledge rather than of a particular world.

Second, the exploration term tracks *availability*, which is the part naive
implementations get wrong. Cowling et al. select

    argmax_a [ Q(a) + c * sqrt( ln n'(a) / n(a) ) ]

with ``n'(a)`` the number of iterations in which ``a`` was legal at this node
and ``n(a)`` the number in which it was played. Availability goes inside the
logarithm; visits is the denominator.

Where legality depends on hidden state, an action appears in only some
determinizations and so is playable in only some iterations. Charging it for
iterations in which it was never an option -- by putting the node's total visit
count in the logarithm -- inflates its bonus and drives the search toward
rarely-legal moves. In this domain those are precisely the moves the legality
engine is refusing, so the bug would push the planner at content it is not
allowed to ship.

Determinization failure is handled the way the reference paper handles it:
retry the sample, and if the sampler cannot produce a usable state at all,
give up on search for this move and return a uniformly random legal action
rather than crashing. A planner that throws inside a production loop because
the inference model had a bad episode is worse than a planner that admits it
is guessing.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from fronts.game.types import (
    CHANCE_PLAYER,
    TERMINAL_PLAYER,
    ActionKey,
    Observation,
    State,
)
from fronts.protocols import CodeWorldModel, StateInference, ValueFunction

__all__ = ["ISMCTSConfig", "ISMCTS", "MCTSResult"]


@dataclass
class ISMCTSConfig:
    """Numeric knobs for ISMCTS, all with documented defaults.

    Every constant the algorithm needs lives here so that callers tuning a
    campaign never edit algorithm bodies, and so the defaults are visible in
    one place.
    """

    simulations: int = 1000
    """Search budget per move. 1000 matches the per-move budget of Cowling et
    al. (2012), whose experiments found it sufficient for strong play on
    medium-sized games."""

    exploration_c: float = math.sqrt(2.0)
    """UCT exploration coefficient. sqrt(2) is the classic Kocsis and Szepesvari
    (2006) default for zero-sum-style reward scales."""

    rollout_depth: int = 10
    """Random plies per rollout before truncation when no value function is
    supplied."""

    rollouts_per_leaf: int = 10
    """Averaging over more rollouts costs simulations linearly and reduces leaf
    variance only as 1/sqrt(n); 10 is the paper's practical middle ground."""

    max_resample_retries: int = 10
    """Determinization attempts per iteration before the iteration is skipped.
    Mirrors the retry-then-fallback behaviour of the reference
    implementation."""

    seed: int | None = None
    """Seed for every random draw the search itself makes. None means the
    system entropy source is used and the search is not reproducible."""

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError(f"simulations must be >= 1, got {self.simulations}")
        if self.exploration_c <= 0.0:
            raise ValueError(f"exploration_c must be > 0, got {self.exploration_c}")
        if self.rollout_depth < 0:
            raise ValueError(f"rollout_depth must be >= 0, got {self.rollout_depth}")
        if self.rollouts_per_leaf < 1:
            raise ValueError(
                f"rollouts_per_leaf must be >= 1, got {self.rollouts_per_leaf}"
            )
        if self.max_resample_retries < 0:
            raise ValueError(
                f"max_resample_retries must be >= 0, got {self.max_resample_retries}"
            )


class _Edge:
    """Statistics for one action at one node.

    ``availability`` is the count that matters. See the module docstring: it is
    incremented every time the action is legal in a determinization passing
    through the node, whether or not it was selected, and it is the term that
    goes inside the UCT logarithm -- with ``visits`` as the denominator.
    """

    __slots__ = ("availability", "total_value", "visits")

    def __init__(self) -> None:
        self.visits: int = 0
        self.availability: int = 0
        self.total_value: float = 0.0


class _Node:
    """One information set of the searching player.

    Created only at the searching player's decision points, so the node's
    identity is the sequence of that player's own actions from the root --
    which is precisely an information set in the tree-form sense.
    """

    __slots__ = ("children", "edges", "visits")

    def __init__(self) -> None:
        self.visits: int = 0
        self.edges: dict[ActionKey, _Edge] = {}
        self.children: dict[ActionKey, _Node] = {}


@dataclass(frozen=True)
class MCTSResult:
    """Search output, satisfying ``PlanResult``.

    ``visits`` and ``values`` are per root action. ``availability`` is exposed
    beyond the protocol because it is the audit trail for the UCT correction:
    any action's availability must be >= its visits, and an action legal in a
    fraction of determinizations should show availability at roughly that
    fraction of the simulation count.
    """

    move: ActionKey
    value: float
    visits: dict[ActionKey, int]
    values: dict[ActionKey, float]
    availability: dict[ActionKey, int] = field(default_factory=dict)


class ISMCTS:
    """Determinized UCT search over information sets. Satisfies ``Planner``.

    Opponents are modelled as uniform-random over their legal actions during
    descent and rollouts. That is the weakest defensible opponent model and
    therefore the safest default: a planner that assumes a strong opponent is
    right twice and catastrophically wrong once, while a planner that assumes
    a random one is merely pessimistic. Callers with a real opponent model can
    wrap it in the world model rather than here.
    """

    def __init__(
        self,
        config: ISMCTSConfig | None = None,
        inference: StateInference | None = None,
        value_function: ValueFunction | None = None,
    ) -> None:
        self._config = config if config is not None else ISMCTSConfig()
        self._inference = inference
        self._value_function = value_function

    def plan(
        self,
        model: CodeWorldModel,
        state: State,
        player: int,
        budget: int,
    ) -> MCTSResult:
        """Search from ``state`` for ``player`` and return the chosen move.

        ``budget`` is the simulation count for this call and overrides
        ``config.simulations``; a non-positive budget falls back to the config
        value so that protocol callers passing 0 still get a real search.

        Determinism: every draw the search makes comes from a fresh
        ``random.Random(config.seed)`` created per call, so two calls with the
        same arguments produce identical visit counts even within one process.
        Total determinism additionally requires the injected ``StateInference``
        to be reproducible across call sequences, which is the inference
        object's responsibility, not the search's.
        """
        simulations = budget if budget > 0 else self._config.simulations
        rng = random.Random(self._config.seed)

        root_actions = self._root_actions(model, state)
        if not root_actions:
            raise ValueError(
                "cannot plan: state has no legal actions for the searching player"
            )

        root = _Node()
        history = self._history_of(state)
        searched = 0

        for _ in range(simulations):
            determinization = self._determinize(model, history, player, state, rng)
            if determinization is None:
                continue
            searched += 1
            self._run_iteration(model, determinization, player, root, rng)

        if searched == 0:
            # Every determinization attempt failed even after retries. Cowling
            # et al. degrade to a random legal move here rather than abort the
            # game; so do we, and for the same reason -- a bad inference
            # episode must not take the planning loop down with it.
            return MCTSResult(
                move=rng.choice(root_actions),
                value=0.0,
                visits={},
                values={},
                availability={},
            )

        best_action = max(
            (a for a in root.children),
            key=lambda a: (root.children[a].visits, root.edges[a].total_value),
        )
        best_edge = root.edges[best_action]
        return MCTSResult(
            move=best_action,
            value=best_edge.total_value / best_edge.visits if best_edge.visits else 0.0,
            visits={a: root.edges[a].visits for a in root.edges},
            values={
                a: root.edges[a].total_value / root.edges[a].visits
                for a in root.edges
                if root.edges[a].visits
            },
            availability={a: root.edges[a].availability for a in root.edges},
        )

    # -- iteration ---------------------------------------------------------

    def _run_iteration(
        self,
        model: CodeWorldModel,
        determinization: State,
        player: int,
        root: _Node,
        rng: random.Random,
    ) -> None:
        """One determinize-descend-evaluate-backpropagate pass."""
        node: _Node | None = root
        root.visits += 1
        state: State = determinization
        path: list[tuple[_Node, ActionKey]] = []
        value: float

        while True:
            who = model.get_current_player(state)
            if who == TERMINAL_PLAYER:
                value = model.get_rewards(state).get(player, 0.0)
                break
            if who == CHANCE_PLAYER:
                state = model.apply_action(
                    state, self._sample_chance(model, state, rng)
                )
                continue

            legal = model.get_legal_actions(state)
            if not legal:
                # A decision state with no moves is the model's way of ending
                # the game without marking it terminal; honour its rewards.
                value = model.get_rewards(state).get(player, 0.0)
                break

            if who == player:
                assert node is not None
                for action in legal:
                    if action not in node.edges:
                        node.edges[action] = _Edge()
                    # Availability is counted for every legal action in this
                    # determinization, selected or not. This is the denominator
                    # correction described in the module docstring.
                    node.edges[action].availability += 1

                untried = [a for a in legal if a not in node.children]
                if untried:
                    action = rng.choice(untried)
                    child = _Node()
                    node.children[action] = child
                    state = model.apply_action(state, action)
                    path.append((node, action))
                    value = self._evaluate(model, state, player, rng)
                    break

                action = self._select_uct(node, legal)
                state = model.apply_action(state, action)
                path.append((node, action))
                node = node.children[action]
            else:
                # Opponent or third player: resolved inside the
                # determinization, no tree node, uniform over legal actions.
                state = model.apply_action(state, rng.choice(legal))

        for parent, action in path:
            edge = parent.edges[action]
            edge.visits += 1
            edge.total_value += value
            parent.children[action].visits += 1

    def _select_uct(
        self,
        node: _Node,
        legal: list[ActionKey],
    ) -> ActionKey:
        """Pick an action by UCT among children legal in this determinization.

        Selection is restricted to ``legal`` -- the actions legal in the
        determinization being descended, not everything the node has ever
        expanded. Without that restriction an action materialised under one
        determinization gets applied under another where it does not exist,
        and the search quietly plans in worlds that cannot occur.

        The exploration term is the one thing ISMCTS changes about UCB1, and it
        is worth being exact. Cowling, Powley and Whitehouse (2012) select

            argmax_a [ Q(a) + c * sqrt( ln n'(a) / n(a) ) ]

        where ``n'(a)`` is the AVAILABILITY count -- iterations in which ``a``
        was legal at this node -- and ``n(a)`` is the visit count, iterations in
        which it was actually played. Availability sits inside the logarithm;
        visits is the denominator.

        Getting these the wrong way round is the classic ISMCTS bug, and it is
        silent: the search still runs and still returns a move. The reason the
        substitution matters is that an action legal in only a fraction of
        determinizations should not be charged for iterations in which it was
        never an option. Using the node's total visit count in the logarithm
        does exactly that, and inflates the bonus on rarely-legal moves until
        the search over-commits to them. Here that would mean systematically
        favouring content the legality engine mostly forbids, which is the
        opposite of the behaviour the gates exist to produce.
        """
        best_score = -math.inf
        best_action: ActionKey | None = None
        for action in legal:
            if action not in node.children:
                continue
            edge = node.edges[action]
            if edge.visits == 0:
                # Expanded this very iteration; cannot be selected yet.
                continue
            mean_value = edge.total_value / edge.visits
            log_availability = (
                math.log(edge.availability) if edge.availability > 1 else 0.0
            )
            exploration = self._config.exploration_c * math.sqrt(
                log_availability / edge.visits
            )
            score = mean_value + exploration
            if score > best_score:
                best_score = score
                best_action = action
        if best_action is None:
            raise ValueError("UCT selection found no legal expanded children")
        return best_action

    def _evaluate(
        self,
        model: CodeWorldModel,
        state: State,
        player: int,
        rng: random.Random,
    ) -> float:
        """Value a fresh leaf: heuristic if supplied, else averaged rollouts."""
        if model.get_current_player(state) == TERMINAL_PLAYER:
            return model.get_rewards(state).get(player, 0.0)
        if self._value_function is not None:
            return self._value_function(state, player)

        total = 0.0
        for _ in range(self._config.rollouts_per_leaf):
            rollout_state = state
            for _ in range(self._config.rollout_depth):
                who = model.get_current_player(rollout_state)
                if who == TERMINAL_PLAYER:
                    break
                if who == CHANCE_PLAYER:
                    action = self._sample_chance(model, rollout_state, rng)
                else:
                    legal = model.get_legal_actions(rollout_state)
                    if not legal:
                        break
                    action = rng.choice(legal)
                rollout_state = model.apply_action(rollout_state, action)
            # One reading at the end, exactly like the terminal branch above:
            # get_rewards is cumulative-to-date, so summing it per ply would
            # count a reward that settles early up to rollout_depth times.
            total += model.get_rewards(rollout_state).get(player, 0.0)
        return total / self._config.rollouts_per_leaf

    def _sample_chance(
        self,
        model: CodeWorldModel,
        state: State,
        rng: random.Random,
    ) -> ActionKey:
        """Sample one outcome from a chance node by inverse-CDF draw."""
        outcomes = model.chance_outcomes(state)
        if not outcomes:
            raise ValueError("chance node returned no outcomes")
        draw = rng.random()
        cumulative = 0.0
        for outcome, probability in outcomes:
            if probability < 0.0:
                raise ValueError(f"negative chance probability {probability}")
            cumulative += probability
            if draw < cumulative:
                return outcome
        # Floating point shortfall: return the last outcome rather than fail.
        return outcomes[-1][0]

    # -- determinization ---------------------------------------------------

    def _determinize(
        self,
        model: CodeWorldModel,
        history: list[tuple[Observation | None, ActionKey | None]],
        player: int,
        fallback: State,
        rng: random.Random,
    ) -> State | None:
        """Sample one hidden state consistent with the player's observations.

        Returns None when no usable sample can be produced after
        ``config.max_resample_retries`` attempts. Without an injected
        ``StateInference`` there is nothing to infer and the caller's state is
        used directly, collapsing the search to plain (perfect-information)
        UCT -- the honest degradation, since claiming to model hidden state we
        cannot sample would be worse.
        """
        if self._inference is None:
            return fallback

        attempts = self._config.max_resample_retries + 1
        for _ in range(attempts):
            try:
                candidate = self._inference.resample_state(history, player)
            except Exception:
                continue
            if self._usable(model, candidate):
                return candidate
        return None

    def _usable(self, model: CodeWorldModel, candidate: Any) -> bool:
        """A determinization is usable if the model can advance it.

        Deliberately minimal: it must be a state-shaped mapping and the model
        must be able to say whose turn it is. Anything stricter would encode
        assumptions about the synthesised state schema, which is the model's
        business, not the search's.
        """
        if not isinstance(candidate, dict):
            return False
        try:
            who = model.get_current_player(State(candidate))
        except Exception:
            return False
        return isinstance(who, int)

    def _root_actions(self, model: CodeWorldModel, state: State) -> list[ActionKey]:
        """Legal actions at the root, tolerating a chance or terminal root."""
        who = model.get_current_player(state)
        if who in (CHANCE_PLAYER, TERMINAL_PLAYER):
            return []
        return model.get_legal_actions(state)

    def _history_of(
        self, state: State
    ) -> list[tuple[Observation | None, ActionKey | None]]:
        """Read the player's observation-action history from the state.

        Convention, not contract: if the state mapping carries a ``history``
        key holding an observation/action sequence, it is passed to
        ``resample_state`` so the sampler can condition on what the player
        actually saw. Otherwise an empty history is passed. The state schema
        belongs to the world model; the search only reads this one optional
        key and writes nothing.
        """
        candidate = state.get("history")
        if isinstance(candidate, list):
            return candidate
        return []
