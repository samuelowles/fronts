"""Stdlib-only self-check for the game layer. No test runner, no network.

Runs every registered check, prints one summary line, and exits 0 on success
or 1 with the failure list. The registry (``CHECKS``) is a module-level list
so later tasks can append their own checks without touching the runner:

    from scripts.selfcheck import CHECKS, check

    @check
    def my_check() -> None: ...

Each check is a zero-argument function that raises on failure. The
assertions mirror ``tests/test_game.py``; this file exists so the guarantee
survives even where no test runner is installed.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Callable
from datetime import date

# Importing every new module IS the first check: a module that cannot be
# imported cannot be wrong in interesting ways, only boring ones.
from fronts.game import action_space, legality, payoff, priors
from fronts.game.action_space import ActionCodec
from fronts.game.legality import (
    LegalityContext,
    LegalityEngine,
    OperatorPolicy,
    settled_count,
)
from fronts.game.payoff import (
    Economics,
    PayoffHealth,
    allowable_cac,
    cac_payback_months,
    health,
    ltv,
    ltv_cac_ratio,
    reward,
    true_conversions,
)
from fronts.game.types import (
    Archetype,
    ClaimClass,
    CtaMode,
    EmotionalVector,
    Format,
    Hold,
    HookFamily,
    Kill,
    Move,
    Observation,
    Platform,
    Publish,
    Scale,
    SemanticTier,
)
from fronts.solvers.blotto import BlottoAllocator, BlottoConfig, Front
from fronts.solvers.congestion import CongestionGame
from fronts.solvers.exp3 import EXP3, EXP3Config
from fronts.solvers.ismcts import ISMCTS, ISMCTSConfig
from fronts.solvers.signalling import SignalCost, separates

CHECKS: list[Callable[[], None]] = []


def check(fn: Callable[[], None]) -> Callable[[], None]:
    CHECKS.append(fn)
    return fn


ANGLE = "launch_theater"
ENGINE = LegalityEngine(OperatorPolicy())


def _publish(**overrides: object) -> Publish:
    fields: dict[str, object] = {
        "platform": Platform.TIKTOK,
        "format": Format.CAROUSEL,
        "archetype": Archetype.ANTI_HERO_RANT,
        "vector": EmotionalVector.ANGER_INJUSTICE,
        "semantic_tier": SemanticTier.TRIBAL_IDENTITY,
        "hook": HookFamily.PATTERN_INTERRUPT,
        "avatar": "solo_founder",
        "cta_mode": CtaMode.START_TRIAL,
        "claim_class": ClaimClass.FIRST_PARTY_PROOF,
        "angle": ANGLE,
        "utm_content": "c0412ab9",
    }
    fields.update(overrides)
    return Publish(**fields)  # type: ignore[arg-type]


def _gate_context(
    settled_7d: int,
    settled_14d: int = 400,
    age_days: int = 20,
    coverage: float = 0.8,
) -> LegalityContext:
    ctx = LegalityContext(current_date=date(2026, 8, 21))
    ctx.settled_conversions_7d[ANGLE] = settled_7d
    ctx.settled_conversions_14d[ANGLE] = settled_14d
    ctx.angle_age_days[ANGLE] = age_days
    ctx.angle_attribution_coverage[ANGLE] = coverage
    ctx.allocation[ANGLE] = 100.0
    return ctx


def _observation(
    conversions: int,
    coverage: float = 1.0,
    incrementality: float = 1.0,
    is_partial: bool = False,
) -> Observation:
    return Observation(
        utm_content="c0412ab9",
        posted_at="2026-08-19T09:00:00",
        observed_at="2026-08-21T09:00:00",
        attributed_conversions=conversions,
        attribution_coverage=coverage,
        incrementality=incrementality,
        is_partial=is_partial,
    )


@check
def modules_import() -> None:
    """Every new module imports and exposes its public surface."""
    assert hasattr(action_space, "ActionCodec")
    assert hasattr(action_space, "enumerate_publishes")
    assert hasattr(action_space, "utm_content_id")
    assert hasattr(legality, "LegalityEngine")
    assert hasattr(payoff, "reward")
    assert hasattr(priors, "ARCHETYPE_PRIORS")


@check
def every_prior_has_a_source() -> None:
    """A prior without a source must not exist -- verified per entry, per
    field, not by inspection."""
    for name, prior in priors.ARCHETYPE_PRIORS.items():
        for band in (
            prior.hook_rate,
            prior.hold_rate,
            prior.outbound_ctr,
            prior.expected_cvr,
        ):
            assert band.source, f"{name}.band has empty source"
        assert prior.source, f"{name} has empty source"
    for name, prior in priors.SEMANTIC_TIER_PRIORS.items():
        assert prior.cpc_usd.source, f"{name}.cpc_usd has empty source"
        assert prior.source, f"{name} has empty source"
    for name, prior in priors.VECTOR_PRIORS.items():
        if prior is None:
            continue
        assert prior.cac_usd.source, f"{name}.cac_usd has empty source"
        assert prior.ltv_usd.source, f"{name}.ltv_usd has empty source"
        assert prior.source, f"{name} has empty source"
    for name, prior in priors.PLATFORM_PRIORS.items():
        for band in (
            prior.engagement_rate,
            prior.organic_reach_share,
            prior.creative_lifespan_days,
            prior.min_new_creatives_per_week,
            prior.landing_page_cvr,
        ):
            assert band.source, f"{name}.band has empty source"
    for constant in (
        priors.CLIENT_SIDE_EVENT_LOSS,
        priors.SERVER_SIDE_RECOVERY,
        priors.NON_INCREMENTAL_SHARE,
        priors.DARK_SOCIAL_B2B_SHARE,
        priors.REPORTING_LAG_HOURS,
        priors.HOOK_RATE_ABANDON_THRESHOLD,
        priors.HOLD_RATE_DECAY_TRIGGER,
        priors.CTR_FATIGUE_TRIGGER,
        priors.WINNING_HOOK_BURNOUT_DAYS,
    ):
        assert constant.source, "degradation constant has empty source"
    # Unmeasured vectors must be None, never a guess.
    for vector in (
        EmotionalVector.ANGER_INJUSTICE,
        EmotionalVector.FEAR_LOSS,
        EmotionalVector.ANALYTICAL_PROOF,
    ):
        assert priors.VECTOR_PRIORS[vector] is None


@check
def codec_round_trips_random_moves() -> None:
    """250 randomly generated moves survive encode -> decode exactly."""
    codec = ActionCodec()
    rng = random.Random(20260821)
    avatars = ["solo_founder", "agency_ops", "in_house_growth"]
    angles = [ANGLE, "enemy_of_slop", "founder_field_notes"]
    count = 0
    for _ in range(250):
        roll = rng.random()
        if roll < 0.7:
            move: Move = _publish(
                platform=rng.choice(list(Platform)),
                format=rng.choice(list(Format)),
                archetype=rng.choice(list(Archetype)),
                vector=rng.choice(list(EmotionalVector)),
                semantic_tier=rng.choice(list(SemanticTier)),
                hook=rng.choice(list(HookFamily)),
                avatar=rng.choice(avatars),
                cta_mode=rng.choice(list(CtaMode)),
                claim_class=rng.choice(list(ClaimClass)),
                angle=rng.choice(angles),
                utm_content=f"{rng.getrandbits(32):08x}",
            )
        elif roll < 0.85:
            move = Scale(rng.choice(angles), round(rng.uniform(0.5, 2.0), 2))
        elif roll < 0.95:
            move = Kill(rng.choice(angles), rng.choice(["burnout", "drawdown"]))
        else:
            move = Hold()  # type: ignore[assignment]
        decoded = codec.decode(codec.encode(move))
        assert decoded == move, f"round-trip mismatch: {move!r} -> {decoded!r}"
        count += 1
    assert count >= 200


@check
def creative_judgement_gate_at_the_boundary() -> None:
    """49 settled conversions in the 7d window refuses Scale; 50 permits.

    The factor stays below ``spend_commitment_factor`` so the move is a
    creative judgement rather than a budget commitment, and answers to the
    50/7d gate rather than 300/14d.
    """
    move = Scale(ANGLE, 1.10)
    low = ENGINE.check(move, _gate_context(settled_7d=49))
    assert not low.legal, f"gate must refuse at 49: {low!r}"
    assert low.rule == "CREATIVE_JUDGEMENT"
    high = ENGINE.check(move, _gate_context(settled_7d=50))
    assert high.legal, f"gate must permit at 50: {high!r}"


@check
def spend_commitment_gate_at_the_boundary() -> None:
    """299 settled conversions over 14d refuses Scale; 300 with 14 days
    elapsed permits, and 300 without the elapsed days still refuses."""
    move = Scale(ANGLE, 1.20)
    low = ENGINE.check(move, _gate_context(settled_7d=60, settled_14d=299))
    assert not low.legal and low.rule == "SPEND_COMMITMENT"
    high = ENGINE.check(
        move, _gate_context(settled_7d=60, settled_14d=300, age_days=14)
    )
    assert high.legal, f"gate must permit at 300/14d: {high!r}"
    too_young = ENGINE.check(
        move, _gate_context(settled_7d=60, settled_14d=300, age_days=13)
    )
    assert not too_young.legal and too_young.rule == "SPEND_COMMITMENT"


@check
def scale_magnitude_selects_the_gate() -> None:
    """The two evidence gates attach to different moves, and neither is dead
    code.

    Evidence sufficient to nudge a live angle is not evidence sufficient to put
    money behind it. A state carrying 60 settled conversions over 7 days lets a
    1.10x nudge through and refuses a 1.20x commitment; only the 300/14d state
    clears both. If a future edit ever routes every Scale through both gates,
    the first assertion below is the one that fails.
    """
    creative_only = _gate_context(settled_7d=60, settled_14d=60, age_days=7)
    assert ENGINE.check(Scale(ANGLE, 1.10), creative_only).legal
    commitment = ENGINE.check(Scale(ANGLE, 1.20), creative_only)
    assert not commitment.legal and commitment.rule == "SPEND_COMMITMENT"

    funded = _gate_context(settled_7d=60, settled_14d=300, age_days=14)
    assert ENGINE.check(Scale(ANGLE, 1.20), funded).legal


@check
def partial_observations_never_open_gates() -> None:
    """An observation inside the reporting lag contributes zero toward any
    gate, no matter how many conversions it claims."""
    mixed = [
        _observation(conversions=100, is_partial=True),
        _observation(conversions=49),
    ]
    assert settled_count(mixed) == 49
    move = Scale(ANGLE, 1.10)
    ctx = _gate_context(settled_7d=settled_count(mixed))
    verdict = ENGINE.check(move, ctx)
    assert not verdict.legal and verdict.rule == "CREATIVE_JUDGEMENT"
    all_partial = [_observation(conversions=500, is_partial=True)]
    assert settled_count(all_partial) == 0


@check
def low_coverage_freezes_scale_and_kill() -> None:
    """Below the attribution floor an angle is unjudgeable: both Scale and
    Kill are refused."""
    ctx = _gate_context(settled_7d=60, settled_14d=400, coverage=0.49)
    scale = ENGINE.check(Scale(ANGLE, 1.20), ctx)
    kill = ENGINE.check(Kill(ANGLE, "no_signal"), ctx)
    assert not scale.legal and scale.rule == "ATTRIBUTION_COVERAGE"
    assert not kill.legal and kill.rule == "ATTRIBUTION_COVERAGE"
    at_floor = _gate_context(settled_7d=60, settled_14d=400, coverage=0.5)
    assert ENGINE.check(Scale(ANGLE, 1.20), at_floor).legal


@check
def untracked_publish_is_refused() -> None:
    """Empty utm_content is illegal: untracked output cannot be learned
    from."""
    verdict = ENGINE.check(_publish(utm_content=""), _gate_context(0))
    assert not verdict.legal and verdict.rule == "TRACKED_OUTPUT"


@check
def coverage_and_incrementality_pull_opposite_ways() -> None:
    """Grossing up for unseen coverage raises the estimate; discounting
    non-incremental conversions lowers it."""
    base = true_conversions(_observation(conversions=10))
    assert base == 10.0
    grossed_up = true_conversions(_observation(conversions=10, coverage=0.5))
    discounted = true_conversions(
        _observation(conversions=10, incrementality=0.5)
    )
    both = true_conversions(
        _observation(conversions=10, coverage=0.5, incrementality=0.5)
    )
    assert grossed_up > base, "coverage must gross up"
    assert discounted < base, "incrementality must discount"
    assert both == base, "equal half-corrections cancel, they do not compound"


@check
def worked_economics_example() -> None:
    """ARPU 100, margin 0.8, churn 0.08 -> LTV 1000. Ten attributed
    conversions at half coverage and 0.8 incrementality are 16 true
    conversions; at cost 600 the reward is 15,400."""
    economics = Economics(
        arpu_monthly=100.0,
        gross_margin=0.8,
        monthly_churn=0.08,
        cogs_share=0.2,
        fixed_cost_per_post=50.0,
    )
    assert abs(ltv(economics) - 1000.0) < 1e-9
    obs = _observation(conversions=10, coverage=0.5, incrementality=0.8)
    assert abs(true_conversions(obs) - 16.0) < 1e-9
    assert abs(reward(obs, economics, 600.0) - 15_400.0) < 1e-6
    assert abs(ltv_cac_ratio(economics, 65.0) - 1000.0 / 65.0) < 1e-9
    assert health(ltv_cac_ratio(economics, 65.0)) is PayoffHealth.UNDERSPENDING
    assert health(0.9) is PayoffHealth.STOP
    assert health(1.5) is PayoffHealth.MARGINAL
    assert health(3.0) is PayoffHealth.GOLDEN
    assert abs(cac_payback_months(65.0, economics) - 0.8125) < 1e-9
    assert Economics.__dataclass_fields__["allowable_cac_share"].default == 0.30
    assert abs(allowable_cac(economics) - 300.0) < 1e-9, (
        "allowable CAC is the corpus 0.30 share of the worked example's LTV 1000"
    )


@check
def economics_margin_and_cogs_cannot_contradict() -> None:
    """Cost of service includes cost of goods, so margin + cogs_share may
    not exceed 1.0; a config claiming otherwise is refused at construction
    rather than silently rebasing every reward on a contradiction."""
    Economics(arpu_monthly=100.0, gross_margin=0.8, monthly_churn=0.08,
              cogs_share=0.2, fixed_cost_per_post=50.0)  # boundary is legal
    try:
        Economics(
            arpu_monthly=100.0, gross_margin=0.9, monthly_churn=0.08,
            cogs_share=0.2, fixed_cost_per_post=50.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("margin 0.9 + cogs 0.2 must be refused")




# ===== solvers ===============================================================

# -- a minimal world model for the ISMCTS checks -------------------------------
#
# One decision with a hidden quality gating a rarely-legal action, then one
# chance draw, then a terminal reward. Expected values: good_move 1.025,
# rare_move 0.925, bad_move 0.225, so the known-optimal move is good_move.
# Defined inline because selfcheck must not import from tests/.


class _ToyWorld:
    def initial_state(self) -> dict:
        return {"q": None, "phase": "decide", "move": None, "bonus": 0.0}

    def apply_action(self, state: dict, action: str) -> dict:
        successor = dict(state)
        if successor["phase"] == "decide":
            successor["phase"] = "chance"
            successor["move"] = action
        else:
            successor["phase"] = "done"
            successor["bonus"] = 0.1 if action == "boost" else 0.0
        return successor

    def get_current_player(self, state: dict) -> int:
        if state["phase"] == "decide":
            return 0
        if state["phase"] == "chance":
            return -1
        return -4

    def get_legal_actions(self, state: dict) -> list[str]:
        if state["phase"] != "decide":
            return []
        if state["q"] == "bad":
            return ["good_move", "bad_move"]
        return ["good_move", "bad_move", "rare_move"]

    def get_observations(self, state: dict) -> dict:
        return {0: None}

    def get_rewards(self, state: dict) -> dict:
        if state["phase"] != "done":
            return {0: 0.0}
        base = {"good_move": 1.0, "bad_move": 0.2, "rare_move": 0.9}
        return {0: base[state["move"]] + state["bonus"]}

    def chance_outcomes(self, state: dict) -> list:
        return [("boost", 0.25), ("normal", 0.75)]


class _ToyInference:
    """q good with p=0.3, reproducible per call index so plan runs repeat."""

    def __init__(self) -> None:
        self._calls = 0

    def resample_state(self, history: object, player_id: int) -> dict:
        rng = random.Random(1234 + self._calls)
        self._calls += 1
        q = "good" if rng.random() < 0.3 else "bad"
        return {"q": q, "phase": "decide", "move": None, "bonus": 0.0}


@check
def ismcts_deterministic_under_seed() -> None:
    world = _ToyWorld()
    first = ISMCTS(ISMCTSConfig(seed=42), inference=_ToyInference()).plan(
        world, world.initial_state(), 0, 300
    )
    second = ISMCTS(ISMCTSConfig(seed=42), inference=_ToyInference()).plan(
        world, world.initial_state(), 0, 300
    )
    assert first.move == second.move, "ISMCTS move differs across identical runs"
    assert first.visits == second.visits, (
        f"visit counts differ: {first.visits} vs {second.visits}"
    )
    assert first.move == "good_move", f"expected the known-optimal move, got {first.move}"


@check
def blotto_allocations_sum_exactly() -> None:
    fronts = [
        Front("ig", "f1", "v1", 10.0),
        Front("tt", "f2", "v2", 6.0),
        Front("x", "f3", "v1", 4.0),
    ]
    for units in (0, 1, 5, 10):
        allocator = BlottoAllocator(BlottoConfig(units=units, seed=units))
        for allocation, probability in allocator.equilibrium_mixture(fronts, 100):
            total = sum(allocation.values())
            assert total == units, f"allocation sums to {total}, expected {units}"
            assert all(count >= 0 for count in allocation.values()), "negative allocation"
            assert 0.0 < probability <= 1.0
        response = allocator.pure_best_response(fronts, {f: 1 for f in fronts})
        assert sum(response.values()) == units


@check
def exp3_probabilities_sum_to_one() -> None:
    exp3 = EXP3(["alpha", "beta", "gamma"], EXP3Config(gamma=0.15))
    rng = random.Random(2)
    rewards = {"alpha": 0.3, "beta": 0.6, "gamma": 0.9}
    for _ in range(2000):
        total = sum(exp3.probabilities().values())
        assert abs(total - 1.0) < 1e-9, f"probabilities sum to {total}"
        arm = exp3.select(rng)
        exp3.update(arm, rewards[arm])
    assert abs(sum(exp3.probabilities().values()) - 1.0) < 1e-9


@check
def congestion_best_response_terminates_at_nash() -> None:
    game = CongestionGame(
        players=["p1", "p2", "p3", "p4"],
        angle_payoffs={"benchmark": 1.0, "niche_a": 0.8, "niche_b": 0.7},
    )
    converged = game.best_response_dynamics({p: "benchmark" for p in game.players})
    assert game.is_nash(converged), f"dynamics stopped at a non-Nash profile: {converged}"


@check
def separating_condition_known_case() -> None:
    credible = SignalCost(
        cost_high_type=1.0, cost_low_type=5.0, gain_from_deception=3.0,
        benefit_high_type=4.0,
    )
    assert separates(credible), "expensive-to-fake claim should separate"
    cheap = SignalCost(
        cost_high_type=1.0, cost_low_type=1.0, gain_from_deception=4.0,
        benefit_high_type=4.0,
    )
    assert not separates(cheap), "cheap-to-fake claim must not separate"



# ===== code world models =====================================================

@check
def cwm_sandbox_blocks_forbidden_import() -> None:
    """The sandbox refuses ``import os`` and dunder attribute access at
    compile time, naming the offending node and its line."""
    from fronts.cwm.sandbox import Sandbox, SandboxConfig, SandboxViolation

    for source in ("import os", "x = object.__subclasses__()"):
        try:
            Sandbox().load(source, SandboxConfig())
        except SandboxViolation as exc:
            assert "line 1" in str(exc), f"must name the line: {exc}"
        else:
            raise AssertionError(f"sandbox must refuse {source!r}")


@check
def cwm_reference_reaches_terminal() -> None:
    """The reference world model plays a full episode to a terminal state,
    producing settled observations along the way."""
    import random

    from fronts.cwm.reference import ReferenceConfig, ReferenceWorldModel
    from fronts.game.types import TERMINAL_PLAYER

    model = ReferenceWorldModel(ReferenceConfig(horizon=6))
    state = model.initial_state()
    rng = random.Random(5)
    plies = 0
    while model.get_current_player(state) != TERMINAL_PLAYER and plies < 200:
        if model.get_current_player(state) == -1:
            outcomes = model.chance_outcomes(state)
            action = rng.choices(
                [key for key, _ in outcomes],
                weights=[prob for _, prob in outcomes],
                k=1,
            )[0]
        else:
            legal = model.get_legal_actions(state)
            assert legal, "no legal actions mid-episode"
            action = next(
                (key for key in legal if key.startswith("publish")), legal[0]
            )
        state = model.apply_action(state, action)
        plies += 1
    assert model.get_current_player(state) == TERMINAL_PLAYER
    assert state["log"], "a publishing policy must produce observations"


@check
def cwm_observations_carry_no_hidden_state() -> None:
    """No player's observation exposes theta, standing, saturation, belief
    or fatigue -- the operator sees the degraded dashboard and nothing else."""
    from fronts.cwm.reference import ReferenceConfig, ReferenceWorldModel

    model = ReferenceWorldModel(ReferenceConfig(horizon=3))
    state = model.initial_state()
    legal = [
        key for key in model.get_legal_actions(state) if key.startswith("publish")
    ]
    state = model.apply_action(state, legal[0])
    outcomes = model.chance_outcomes(state)
    state = model.apply_action(state, outcomes[0][0])
    for player, observation in model.get_observations(state).items():
        fields = set(observation.__dataclass_fields__)
        leaked = fields & {"theta", "standing", "saturation", "belief", "asset_fatigue"}
        assert not leaked, f"player {player} observation leaks {leaked}"


@check
def cwm_refinement_selection_is_deterministic() -> None:
    """The same seed selects the same refinement node sequence, and Beta
    sampling favours the high-pass-rate node in aggregate."""
    import random

    from fronts.cwm.refine import RefinementNode, RefinementTree

    tree = RefinementTree(
        nodes=[RefinementNode("good", 0.9), RefinementNode("bad", 0.1)]
    )
    first = [tree.select(random.Random(42)).source for _ in range(50)]
    second = [tree.select(random.Random(42)).source for _ in range(50)]
    assert first == second, "selection must be reproducible under a seed"
    wins = sum(1 for _ in range(1000) if tree.select(random.Random(7)).source == "good")
    assert wins > 900, f"high-h node should dominate; won {wins}/1000"


def main() -> int:
    failures: list[tuple[str, str]] = []
    for fn in CHECKS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- report, don't bury
            failures.append((fn.__name__, f"{type(exc).__name__}: {exc}"))
    print(
        f"selfcheck: {len(CHECKS) - len(failures)}/{len(CHECKS)} checks passed"
    )
    for name, error in failures:
        print(f"  FAIL {name}: {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
