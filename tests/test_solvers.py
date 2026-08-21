"""Unit tests for the solvers layer.

The ISMCTS tests run against a hand-written ``ToyWorld``: a one-decision
imperfect-information game with a known optimum. The operator picks one of
three moves at an information set; a hidden quality ``q`` (good with p=0.3)
gates whether ``rare_move`` is legal; a chance node then adds a bonus of 0.1
with probability 0.25. Expected values: good_move 1.025, rare_move 0.925,
bad_move 0.225 -- so the known-optimal move is ``good_move`` regardless of
``q``, and ``rare_move`` is available in only some determinizations, which is
what makes the fixture exercise the availability-denominator machinery.
"""

from __future__ import annotations

import math
import random

import pytest

from blotto.game.types import State
from blotto.protocols import Planner
from blotto.solvers.blotto import BlottoAllocator, BlottoConfig, Front
from blotto.solvers.congestion import (
    CongestionConfig,
    CongestionGame,
    congested_payoff,
    crowding_adjusted_ranking,
)
from blotto.solvers.exp3 import EXP3, EXP3P, EXP3Config, EXP3PConfig, regret_bound
from blotto.solvers.ismcts import ISMCTS, ISMCTSConfig, MCTSResult
from blotto.solvers.signalling import (
    ClaimCredibility,
    SignalCost,
    separates,
    separating_power,
)
from blotto.solvers.stackelberg import (
    LeaderEstimate,
    best_response,
    robust_best_response,
    value_of_information,
)

# -- toy world model ---------------------------------------------------------


class ToyWorld:
    """One decision, hidden quality, one chance node, terminal reward.

    Satisfies ``CodeWorldModel`` structurally. Every method is deterministic;
    the only randomness in the game is the chance draw, as the protocol
    requires. ``rare_value`` lets tests control how attractive the
    sometimes-legal action is.
    """

    def __init__(self, rare_value: float = 0.9) -> None:
        self._rare_value = rare_value

    def initial_state(self) -> State:
        return State({"q": None, "phase": "decide", "move": None, "bonus": 0.0})

    def apply_action(self, state: State, action: str) -> State:
        successor = dict(state)
        if successor["phase"] == "decide":
            successor["phase"] = "chance"
            successor["move"] = action
        else:
            successor["phase"] = "done"
            successor["bonus"] = 0.1 if action == "boost" else 0.0
        return State(successor)

    def get_current_player(self, state: State) -> int:
        if state["phase"] == "decide":
            return 0
        if state["phase"] == "chance":
            return -1
        return -4

    def get_legal_actions(self, state: State) -> list[str]:
        if state["phase"] != "decide":
            return []
        if state["q"] == "bad":
            return ["good_move", "bad_move"]
        return ["good_move", "bad_move", "rare_move"]

    def get_observations(self, state: State) -> dict[int, None]:
        return {0: None}

    def get_rewards(self, state: State) -> dict[int, float]:
        if state["phase"] != "done":
            return {0: 0.0}
        base = {"good_move": 1.0, "bad_move": 0.2, "rare_move": self._rare_value}
        return {0: base[state["move"]] + state["bonus"]}

    def chance_outcomes(self, state: State) -> list[tuple[str, float]]:
        return [("boost", 0.25), ("normal", 0.75)]


class ToyInference:
    """Determinization sampler: q good with p_good, reproducible per call index.

    Seeding from the call count (rather than holding one stream) makes the
    sequence of determinizations identical for two identical plan runs -- the
    property the ISMCTS determinism contract needs from any injected
    inference.
    """

    def __init__(self, p_good: float = 0.3, base_seed: int = 1234) -> None:
        self._p_good = p_good
        self._base_seed = base_seed
        self._calls = 0

    def resample_state(self, history: object, player_id: int) -> State:
        rng = random.Random(self._base_seed + self._calls)
        self._calls += 1
        q = "good" if rng.random() < self._p_good else "bad"
        return State({"q": q, "phase": "decide", "move": None, "bonus": 0.0})


class FailingInference:
    """Always raises: exercises the retry-then-random fallback path."""

    def resample_state(self, history: object, player_id: int) -> State:
        raise RuntimeError("inference unavailable")


# -- ISMCTS -------------------------------------------------------------------


@pytest.mark.unit
def test_ismcts_satisfies_planner_protocol() -> None:
    assert isinstance(ISMCTS(ISMCTSConfig(seed=1)), Planner)


@pytest.mark.unit
def test_ismcts_deterministic_under_fixed_seed() -> None:
    world = ToyWorld()
    first = ISMCTS(ISMCTSConfig(seed=42), inference=ToyInference()).plan(
        world, world.initial_state(), 0, 1000
    )
    second = ISMCTS(ISMCTSConfig(seed=42), inference=ToyInference()).plan(
        world, world.initial_state(), 0, 1000
    )
    assert first.move == second.move
    assert first.visits == second.visits
    assert first.values == second.values
    assert first.availability == second.availability

    # Without inference the search owns all of its own randomness, so the
    # same instance called twice in one process must also repeat exactly.
    planner = ISMCTS(ISMCTSConfig(seed=7))
    state = State({"q": "good", "phase": "decide", "move": None, "bonus": 0.0})
    again_a = planner.plan(world, state, 0, 500)
    again_b = planner.plan(world, state, 0, 500)
    assert again_a.visits == again_b.visits
    assert again_a.move == again_b.move


@pytest.mark.unit
def test_ismcts_finds_known_optimal_move() -> None:
    world = ToyWorld()
    result = ISMCTS(ISMCTSConfig(seed=42), inference=ToyInference()).plan(
        world, world.initial_state(), 0, 1000
    )
    assert result.move == "good_move"
    # Terminal payoffs: 1.0 + 0.1 with probability 0.25 = 1.025.
    assert result.value == pytest.approx(1.025, abs=0.05)
    assert isinstance(result, MCTSResult)


@pytest.mark.unit
def test_ismcts_availability_denominator_uses_legal_count() -> None:
    """The UCT denominator is availability, not visits through the node.

    ``rare_move`` is legal only when the determinized quality is good
    (p = 0.3), so over 1000 simulations it should be *available* about 300
    times while being *selected* far less often -- it is strictly worse than
    ``good_move``. If selection were not restricted to the current
    determinization's legal actions, visits would exceed availability (that
    bug produced 995 selections out of 318 availabilities when first written).
    """
    world = ToyWorld(rare_value=0.4)  # clearly worse than good_move's 1.0
    result = ISMCTS(ISMCTSConfig(seed=42), inference=ToyInference()).plan(
        world, world.initial_state(), 0, 1000
    )

    assert result.availability["good_move"] == 1000
    assert result.availability["bad_move"] == 1000
    # Binomial(1000, 0.3): mean 300, sd ~14.5; allow a 4-sigma band.
    assert 240 <= result.availability["rare_move"] <= 360

    for action, available in result.availability.items():
        assert result.visits.get(action, 0) <= available

    rare_pull_rate = result.visits["rare_move"] / result.availability["rare_move"]
    assert rare_pull_rate < 0.1
    assert result.visits["good_move"] > result.visits["rare_move"]
    assert result.move == "good_move"


@pytest.mark.unit
def test_ismcts_falls_back_when_inference_always_raises() -> None:
    world = ToyWorld()
    result = ISMCTS(
        ISMCTSConfig(seed=1, max_resample_retries=3), inference=FailingInference()
    ).plan(world, world.initial_state(), 0, 100)
    assert result.move in ("good_move", "bad_move", "rare_move")
    assert result.visits == {}


@pytest.mark.unit
def test_ismcts_value_function_replaces_rollouts() -> None:
    world = ToyWorld()

    class ConstantValue:
        def __call__(self, state: State, player: int) -> float:
            return 0.5

    class OneShotInference:
        """Determinizes to a fixed good world, so the tree is fully seen."""

        def resample_state(self, history: object, player_id: int) -> State:
            return State({"q": "good", "phase": "decide", "move": None, "bonus": 0.0})

    result = ISMCTS(
        ISMCTSConfig(seed=3), inference=OneShotInference(), value_function=ConstantValue()
    ).plan(world, world.initial_state(), 0, 50)
    assert result.move == "good_move"


@pytest.mark.unit
def test_ismcts_rejects_state_without_actions() -> None:
    world = ToyWorld()
    terminal = State({"q": None, "phase": "done", "move": "good_move", "bonus": 0.0})
    with pytest.raises(ValueError):
        ISMCTS(ISMCTSConfig(seed=1)).plan(world, terminal, 0, 10)


@pytest.mark.unit
def test_ismcts_config_validates_itself() -> None:
    with pytest.raises(ValueError):
        ISMCTSConfig(simulations=0)
    with pytest.raises(ValueError):
        ISMCTSConfig(exploration_c=0.0)


# -- Blotto -------------------------------------------------------------------


@pytest.mark.unit
def test_blotto_allocations_sum_exactly_and_stay_nonnegative() -> None:
    for units in (0, 1, 3, 10):
        fronts = [
            Front("ig", "f1", "v1", 10.0),
            Front("ig", "f2", "v2", 6.0),
            Front("tt", "f3", "v1", 0.0),  # zero-value front exercises share guard
        ]
        allocator = BlottoAllocator(BlottoConfig(units=units, seed=units))
        mixture = allocator.equilibrium_mixture(fronts, 200)
        assert mixture
        for allocation, probability in mixture:
            assert sum(allocation.values()) == units
            assert all(count >= 0 for count in allocation.values())
            assert probability > 0.0
        assert sum(p for _, p in mixture) == pytest.approx(1.0)

        response = allocator.pure_best_response(fronts, {f: 1 for f in fronts})
        assert sum(response.values()) == units
        assert all(count >= 0 for count in response.values())


@pytest.mark.unit
def test_blotto_mixture_less_exploitable_than_pure_strategies() -> None:
    """Mixed beats committed pure allocations, against every candidate pure.

    The opponent commits to one response without seeing the draw, which is
    the regime where mixing pays. The pure candidates are the natural ones an
    operator would write by hand: stack the best front, spread evenly, spread
    by value share, and the greedy best response to an even spread.
    """
    fronts = [
        Front("ig", f"a{i}", "founder", value)
        for i, value in enumerate([10.0, 9.0, 8.0, 7.0, 6.0])
    ]
    allocator = BlottoAllocator(BlottoConfig(units=12, seed=3))
    mixture = allocator.equilibrium_mixture(fronts, 300)
    mixture_score = allocator.mixture_exploitability(mixture, fronts, 40)

    stacked = {fronts[0]: 12, **{f: 0 for f in fronts[1:]}}
    even = {f: 2 for f in fronts[:4]} | {fronts[4]: 4}
    by_value = {fronts[0]: 3, fronts[1]: 3, fronts[2]: 2, fronts[3]: 2, fronts[4]: 2}
    greedy = allocator.pure_best_response(fronts, {f: 2 for f in fronts})

    for name, pure in (
        ("stacked", stacked),
        ("even", even),
        ("by_value", by_value),
        ("greedy", greedy),
    ):
        pure_score = allocator.exploitability(pure, fronts, 40)
        assert mixture_score < pure_score, f"{name}: {mixture_score} !< {pure_score}"


@pytest.mark.unit
def test_blotto_sample_draws_from_the_mixture() -> None:
    fronts = [Front("ig", "f1", "v1", 5.0), Front("tt", "f2", "v2", 5.0)]
    allocator = BlottoAllocator(BlottoConfig(units=4, seed=11))
    mixture = allocator.equilibrium_mixture(fronts, 50)
    rng = random.Random(5)
    support = {id(allocation) for allocation, _ in mixture}
    for _ in range(100):
        allocation = allocator.sample(mixture, rng)
        assert sum(allocation.values()) == 4
        assert id(allocation) in support


@pytest.mark.unit
def test_blotto_config_rejects_bad_opponent_model() -> None:
    with pytest.raises(ValueError):
        BlottoConfig(opponent_model="psychic")


# -- EXP3 -----------------------------------------------------------------------


@pytest.mark.unit
def test_exp3_probabilities_sum_to_one_and_converge() -> None:
    exp3 = EXP3(["alpha", "beta", "gamma"], EXP3Config(gamma=0.2))
    rng = random.Random(9)
    stationary = {"alpha": 0.2, "beta": 0.5, "gamma": 1.0}
    for round_index in range(3000):
        probabilities = exp3.probabilities()
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-9)
        arm = exp3.select(rng)
        exp3.update(arm, stationary[arm])
        if round_index % 500 == 0:
            assert all(0.0 <= p <= 1.0 for p in probabilities.values())
    assert exp3.probabilities()["gamma"] > 0.5


@pytest.mark.unit
def test_exp3_survives_100k_rounds_without_overflow() -> None:
    """All-ones rewards drive the winning weight toward overflow.

    Without renormalisation the importance-weighted update compounds the
    winner by exp(gamma * (1/p) / K) per round; over 100k rounds that passes
    float range and every probability collapses. With the ceiling guard the
    weights stay finite and the mixture stays a distribution throughout.
    """
    for bandit in (
        EXP3(["a", "b", "c"], EXP3Config(gamma=0.1, weight_ceiling=1e100)),
        EXP3P(["a", "b", "c"], EXP3PConfig(gamma=0.1, eta=0.1, beta=0.01)),
    ):
        rng = random.Random(4)
        for round_index in range(100_000):
            arm = bandit.select(rng)
            bandit.update(arm, 1.0)
            if round_index % 1000 == 0:
                probabilities = bandit.probabilities()
                assert all(math.isfinite(w) for w in bandit.weights.values())
                assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-9)
        probabilities = bandit.probabilities()
        assert all(math.isfinite(w) for w in bandit.weights.values())
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_exp3_rejects_unnormalised_rewards() -> None:
    exp3 = EXP3(["a", "b"])
    with pytest.raises(ValueError):
        exp3.update("a", 1.5)
    with pytest.raises(ValueError):
        exp3.update("a", -0.1)


@pytest.mark.unit
def test_exp3_regret_bound_is_finite_and_tight_at_optimum() -> None:
    num_arms, rounds = 5, 10_000
    optimal = math.sqrt(num_arms * math.log(num_arms) / ((math.e - 1.0) * rounds))
    at_optimum = regret_bound(num_arms, rounds, optimal)
    assert at_optimum > 0.0
    # Moving away from gamma* in either direction must not improve the bound.
    assert regret_bound(num_arms, rounds, optimal / 2) > at_optimum
    assert regret_bound(num_arms, rounds, optimal * 2) > at_optimum
    with pytest.raises(ValueError):
        regret_bound(num_arms, rounds, 0.0)


# -- congestion -----------------------------------------------------------------


@pytest.mark.unit
def test_congestion_best_response_converges_to_nash() -> None:
    game = CongestionGame(
        players=["p1", "p2", "p3", "p4"],
        angle_payoffs={"benchmark": 1.0, "niche_a": 0.8, "niche_b": 0.7},
    )
    initial = {p: "benchmark" for p in game.players}
    converged = game.best_response_dynamics(initial)
    assert game.is_nash(converged)
    # Nobody stays on a crowded benchmark for free: congestion moved someone.
    assert len({converged[p] for p in game.players}) > 1


@pytest.mark.unit
def test_congestion_potential_strictly_increases_along_improvements() -> None:
    game = CongestionGame(
        players=["p1", "p2", "p3"],
        angle_payoffs={"benchmark": 1.0, "niche": 0.75},
        config=CongestionConfig(decay="exponential", decay_rate=1.5, floor=0.05),
    )
    profile = {p: "benchmark" for p in game.players}
    potentials = [game.potential(profile)]
    for _ in range(50):
        moved = False
        for player in game.players:
            trial = dict(profile)
            trial[player] = game.best_response(profile, player)
            if game.payoff(trial, player) > game.payoff(profile, player):
                profile = trial
                potentials.append(game.potential(profile))
                moved = True
        if not moved:
            break
    assert len(potentials) > 1
    for earlier, later in zip(potentials, potentials[1:], strict=False):
        assert later > earlier
    assert game.is_nash(profile)


@pytest.mark.unit
def test_congested_payoff_decay_families_and_floor() -> None:
    linear = CongestionConfig(decay="linear", decay_rate=0.5, floor=0.1)
    assert congested_payoff(1.0, 0.0, linear) == pytest.approx(1.0)
    assert congested_payoff(1.0, 1.0, linear) == pytest.approx(0.5)
    # The floor binds when the curve falls below it.
    harsh = CongestionConfig(decay="linear", decay_rate=2.0, floor=0.3)
    assert congested_payoff(1.0, 1.0, harsh) == pytest.approx(0.3)
    exponential = CongestionConfig(decay="exponential", decay_rate=2.0)
    assert congested_payoff(1.0, 0.5, exponential) == pytest.approx(math.exp(-1.0))
    power = CongestionConfig(decay="power", decay_rate=1.0)
    assert congested_payoff(1.0, 1.0, power) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        congested_payoff(1.0, 1.5, linear)


@pytest.mark.unit
def test_crowding_adjusted_ranking_flips_the_naive_order() -> None:
    angles = [("benchmarked_angle", 1.0), ("unfashionable_angle", 0.6)]
    occupancy = {"benchmarked_angle": 0.9, "unfashionable_angle": 0.0}
    config = CongestionConfig(decay="linear", decay_rate=0.9, floor=0.0)
    ranked = crowding_adjusted_ranking(angles, occupancy, config)
    assert [angle for angle, _ in ranked] == [
        "unfashionable_angle",
        "benchmarked_angle",
    ]


# -- signalling -----------------------------------------------------------------


@pytest.mark.unit
def test_separating_condition_true_and_false_cases() -> None:
    # Expensive to fake, cheap to tell the truth: separates.
    assert separates(
        SignalCost(
            cost_high_type=1.0,
            cost_low_type=5.0,
            gain_from_deception=3.0,
            benefit_high_type=4.0,
        )
    )
    # Cheap to fake relative to the gain: pooling, whatever it costs the
    # genuine operator.
    assert not separates(
        SignalCost(
            cost_high_type=1.0,
            cost_low_type=2.0,
            gain_from_deception=3.0,
            benefit_high_type=4.0,
        )
    )
    # Too expensive for the high type to bother: an equilibrium nobody plays.
    assert not separates(
        SignalCost(
            cost_high_type=5.0,
            cost_low_type=5.0,
            gain_from_deception=3.0,
            benefit_high_type=4.0,
        )
    )
    # Boundary: the high type exactly indifferent still sends.
    assert separates(
        SignalCost(
            cost_high_type=4.0,
            cost_low_type=5.0,
            gain_from_deception=3.0,
            benefit_high_type=4.0,
        )
    )


@pytest.mark.unit
def test_separating_power_orders_claims_and_stays_in_range() -> None:
    weak = SignalCost(1.0, 4.0, 3.0, 4.0)  # low-type margin only 25%
    strong = SignalCost(0.5, 20.0, 2.0, 10.0)  # wide margins both sides
    violated = SignalCost(1.0, 1.0, 5.0, 4.0)
    assert 0.0 <= separating_power(weak) <= 1.0
    assert separating_power(strong) > separating_power(weak)
    assert separating_power(violated) == 0.0
    assert not separates(violated)

    scorer = ClaimCredibility()
    ranked = scorer.rank(
        [
            ("verifiable_metric", weak),
            ("first_party_proof", strong),
            ("outcome_promise", violated),
        ]
    )
    assert [claim for claim, _ in ranked] == [
        "first_party_proof",
        "verifiable_metric",
        "outcome_promise",
    ]
    assert ranked[0][1] > 0.0
    assert ranked[-1][1] == 0.0


# -- stackelberg ----------------------------------------------------------------


@pytest.mark.unit
def test_point_and_robust_best_response_disagree_under_drift() -> None:
    moves = ["aggressive", "safe"]

    def score(move: str, weights: dict[str, float]) -> float:
        if move == "aggressive":
            return 10.0 * weights["boost"]
        return 4.0 * weights["boost"] + 3.0 * weights["shield"]

    confident_boost = LeaderEstimate(
        weights={"boost": 1.0, "shield": 0.0}, confidence=0.9, drift_rate=0.0
    )
    shield_heavy = LeaderEstimate(
        weights={"boost": 0.0, "shield": 1.0}, confidence=0.5, drift_rate=0.4
    )

    point = best_response(moves, score, confident_boost)
    assert point[0][0] == "aggressive"

    worst_case = robust_best_response(moves, score, [confident_boost, shield_heavy])
    assert worst_case[0][0] == "safe"
    # Aggressive scores 0 in the worst case; safe scores 3.
    assert worst_case[0][1] == pytest.approx(3.0)

    expected = robust_best_response(
        moves, score, [confident_boost, shield_heavy], aggregator="expected"
    )
    # Confidence-weighted means: aggressive 10*0.9/1.4 ~= 6.43 vs safe
    # (4*0.9+3*0.5)/1.4 ~= 3.64 -- the expectation regime flips back to
    # aggressive, which is precisely the brittleness trade.
    assert expected[0][0] == "aggressive"
    assert expected[0][1] == pytest.approx(10.0 * 0.9 / (0.9 + 0.5))

    with pytest.raises(ValueError):
        robust_best_response(moves, score, [], "worst_case")
    with pytest.raises(ValueError):
        robust_best_response(moves, score, [confident_boost], "optimistic")


@pytest.mark.unit
def test_value_of_information_quantifies_posterior_spread() -> None:
    moves = ["aggressive", "safe"]

    def score(move: str, weights: dict[str, float]) -> float:
        if move == "aggressive":
            return 10.0 * weights["boost"]
        return 4.0 * weights["boost"] + 3.0 * weights["shield"]

    disagreeing = [
        LeaderEstimate(weights={"boost": 1.0, "shield": 0.0}, confidence=0.5),
        LeaderEstimate(weights={"boost": 0.0, "shield": 1.0}, confidence=0.5),
    ]
    # Clairvoyance picks aggressive in world 1 (10) and safe in world 2 (3),
    # for 6.5 expected; committing now gets 5.0 from aggressive. Gap 1.5.
    assert value_of_information(disagreeing, moves, score) == pytest.approx(1.5)

    collapsed = [LeaderEstimate(weights={"boost": 1.0, "shield": 0.0}, confidence=1.0)]
    assert value_of_information(collapsed, moves, score) == pytest.approx(0.0)


@pytest.mark.unit
def test_leader_estimate_validates_confidence() -> None:
    with pytest.raises(ValueError):
        LeaderEstimate(weights={"a": 1.0}, confidence=1.5)
    with pytest.raises(ValueError):
        LeaderEstimate(weights={})


def test_uct_formula_places_availability_inside_the_logarithm() -> None:
    """Pin the UCT term placement, which the accounting test cannot see.

    Cowling et al. (2012) select on ``Q(a) + c * sqrt(ln n'(a) / n(a))``, with
    availability inside the logarithm and visits underneath. The classic bug
    swaps them, putting the node's total visit count in the logarithm and
    availability underneath. Both variants still run, still terminate, and
    still respect ``visits <= availability`` -- so the availability-accounting
    test above passes either way. This fixture is built so the two disagree.

    Two actions with identical mean value:

      wide   availability 1000, visits 10  -- legal almost always, rarely tried
      narrow availability   20, visits 10  -- legal rarely, tried just as often

    Correct:  sqrt(ln 1000 / 10) = 0.831  >  sqrt(ln 20 / 10) = 0.547  -> wide
    Inverted: sqrt(ln 1000 / 1000) = 0.083 < sqrt(ln 1000 / 20) = 0.588 -> narrow

    An action legal almost every iteration and still barely explored is exactly
    what the exploration term should be reaching for. If this test ever fails
    with ``narrow``, the logarithm and the denominator have been swapped back.
    """
    from blotto.solvers.ismcts import _Edge, _Node

    node = _Node()
    node.visits = 1000
    for action, availability in (("wide", 1000), ("narrow", 20)):
        edge = _Edge()
        edge.visits = 10
        edge.availability = availability
        edge.total_value = 5.0
        node.edges[action] = edge
        node.children[action] = _Node()

    search = ISMCTS(ISMCTSConfig(seed=7))
    assert search._select_uct(node, ["wide", "narrow"]) == "wide"
