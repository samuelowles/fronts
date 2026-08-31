"""Unit tests for the game layer: priors, action space, legality, payoff.

Mirrors ``scripts/selfcheck.py`` and adds the edge cases a boundary condition
is hiding in: empty candidate lists, zero churn, coverage exactly at the
floor, velocity exactly at the cap, lane split exactly at tolerance.
"""

from __future__ import annotations

import inspect
import math
import random
from datetime import date, datetime, timedelta

import pytest

import blotto.game.payoff as payoff_module
from blotto.game import priors
from blotto.game.action_space import (
    ActionCodec,
    ActionDecodeError,
    enumerate_publishes,
    utm_content_id,
)
from blotto.game.legality import (
    AllocationChange,
    LegalityContext,
    LegalityEngine,
    OperatorPolicy,
    settled_count,
)
from blotto.game.payoff import (
    Economics,
    PayoffHealth,
    cac_payback_months,
    health,
    ltv,
    ltv_cac_ratio,
    reward,
    true_conversions,
)
from blotto.game.types import (
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

pytestmark = pytest.mark.unit

ANGLE = "launch_theater"
ENGINE = LegalityEngine(OperatorPolicy())


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def publish(**overrides: object) -> Publish:
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


def gate_context(
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


def observation(
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


# ---------------------------------------------------------------------------
# Priors.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("table", "band_fields"),
    [
        (priors.ARCHETYPE_PRIORS, ("hook_rate", "hold_rate", "outbound_ctr", "expected_cvr")),
        (priors.SEMANTIC_TIER_PRIORS, ("cpc_usd",)),
        (priors.PLATFORM_PRIORS, (
            "engagement_rate",
            "organic_reach_share",
            "creative_lifespan_days",
            "min_new_creatives_per_week",
            "landing_page_cvr",
        )),
    ],
)
def test_every_prior_entry_carries_a_source(table: dict, band_fields: tuple) -> None:
    for name, prior in table.items():
        for field in band_fields:
            assert getattr(prior, field).source, f"{name}.{field} has no source"
        assert prior.source, f"{name} has no entry-level source"


def test_vector_priors_only_where_measured() -> None:
    measured = {
        EmotionalVector.ASPIRATION_STATUS,
        EmotionalVector.EXHAUSTION_RELIEF,
    }
    for vector, prior in priors.VECTOR_PRIORS.items():
        if vector in measured:
            assert prior is not None
            assert prior.cac_usd.source and prior.ltv_usd.source
        else:
            # Unmeasured in the corpus: None, never a guess.
            assert prior is None


def test_archetype_benchmark_values_are_exact() -> None:
    assert priors.ARCHETYPE_PRIORS[Archetype.UGC_REVIEW].hook_rate.low == 0.22
    assert priors.ARCHETYPE_PRIORS[Archetype.UGC_REVIEW].hook_rate.high == 0.28
    assert priors.ARCHETYPE_PRIORS[Archetype.FOUNDER_TRAUMA].hold_rate.high == 0.42
    assert priors.ARCHETYPE_PRIORS[Archetype.ANTI_HERO_RANT].outbound_ctr.low == 0.025
    assert priors.ARCHETYPE_PRIORS[Archetype.AUTONOMOUS_VOXEL].expected_cvr.high == 0.035


def test_semantic_tier_economics_spread() -> None:
    assert priors.SEMANTIC_TIER_PRIORS[SemanticTier.GENERIC_SLOP].cpc_usd.low == 3.50
    assert priors.SEMANTIC_TIER_PRIORS[SemanticTier.TRIBAL_IDENTITY].cvr == 0.068


def test_band_mid_is_the_mean() -> None:
    assert priors.WINNING_HOOK_BURNOUT_DAYS.mid == 4.0


def test_degradation_constants_carry_sources() -> None:
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
        assert constant.source


# ---------------------------------------------------------------------------
# Action codec.
# ---------------------------------------------------------------------------


def test_codec_round_trips_random_moves() -> None:
    codec = ActionCodec()
    rng = random.Random(20260821)
    avatars = ["solo_founder", "agency_ops", "in_house_growth"]
    angles = [ANGLE, "enemy_of_slop", "founder_field_notes"]
    for _ in range(250):
        roll = rng.random()
        if roll < 0.7:
            move: Move = publish(
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
            move = Hold()
        assert codec.decode(codec.encode(move)) == move


def test_codec_canonical_scale_factor() -> None:
    codec = ActionCodec()
    assert codec.encode(Scale(ANGLE, 1.2)) == f"scale|angle={ANGLE}|factor=1.20"
    assert codec.encode(Scale(ANGLE, 1.234)) == f"scale|angle={ANGLE}|factor=1.23"
    assert codec.decode(f"scale|angle={ANGLE}|factor=1.20") == Scale(ANGLE, 1.2)


def test_codec_spec_example_key() -> None:
    codec = ActionCodec()
    move = publish(
        platform=Platform.TIKTOK,
        format=Format.CAROUSEL,
        archetype=Archetype.ANTI_HERO_RANT,
        vector=EmotionalVector.ANGER_INJUSTICE,
        semantic_tier=SemanticTier.TRIBAL_IDENTITY,
        hook=HookFamily.PATTERN_INTERRUPT,
        avatar="solo_founder",
        cta_mode=CtaMode.START_TRIAL,
        claim_class=ClaimClass.FIRST_PARTY_PROOF,
        angle="launch_theater",
        utm_content="c0412",
    )
    assert codec.encode(move) == (
        "publish|tiktok|carousel|anti_hero_rant|anger_injustice|tribal_identity|"
        "pattern_interrupt|avatar=solo_founder|cta=start_trial|"
        "claim=first_party_proof|angle=launch_theater|utm=c0412"
    )


def test_codec_rejects_pipes_in_free_text() -> None:
    with pytest.raises(ValueError):
        ActionCodec().encode(Scale("a|b", 1.0))


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "publish",
        "publish|tiktok",
        "publish|notaplatform|carousel|anti_hero_rant|anger_injustice|"
        "tribal_identity|pattern_interrupt|avatar=a|cta=start_trial|"
        "claim=first_party_proof|angle=x|utm=c0412",
        f"scale|angle={ANGLE}",
        f"scale|angle={ANGLE}|factor=abc",
        "kill|angle=x",
        "hold|extra",
        "teleport|angle=x",
    ],
)
def test_codec_rejects_malformed_keys(bad: str) -> None:
    with pytest.raises(ActionDecodeError):
        ActionCodec().decode(bad)


def test_utm_content_id_is_deterministic_and_short() -> None:
    move = publish()
    assert utm_content_id(move, "salt") == utm_content_id(move, "salt")
    assert len(utm_content_id(move, "salt")) == 8
    other = publish(hook=HookFamily.CURIOSITY_GAP)
    assert utm_content_id(move, "salt") != utm_content_id(other, "salt")


def test_enumerate_publishes_is_lazy_and_reproducible() -> None:
    kwargs = dict(
        platforms=list(Platform),
        formats=list(Format),
        archetypes=list(Archetype),
        vectors=list(EmotionalVector),
        tiers=list(SemanticTier),
        hooks=list(HookFamily),
        avatars=["solo_founder", "agency_ops"],
        cta_modes=list(CtaMode),
        claim_classes=[ClaimClass.MECHANISM],
        angles=[ANGLE, "enemy_of_slop"],
        limit=25,
        seed=7,
    )
    first = list(enumerate_publishes(**kwargs))
    second = list(enumerate_publishes(**kwargs))
    assert first == second
    assert len(first) == 25
    assert all(isinstance(move, Publish) and move.utm_content for move in first)


def test_enumerate_publishes_small_space_enumerates_all() -> None:
    moves = list(
        enumerate_publishes(
            platforms=[Platform.X],
            formats=[Format.TEXT_THREAD],
            archetypes=[Archetype.UGC_REVIEW],
            vectors=[EmotionalVector.FEAR_LOSS],
            tiers=[SemanticTier.GENERIC_SLOP],
            hooks=[HookFamily.CURIOSITY_GAP, HookFamily.PATTERN_INTERRUPT],
            avatars=["solo_founder"],
            cta_modes=[CtaMode.REPLY_KEYWORD],
            claim_classes=[ClaimClass.NO_CLAIM],
            angles=[ANGLE],
            limit=100,
            seed=1,
        )
    )
    assert len(moves) == 2


# ---------------------------------------------------------------------------
# Evidence gates.
# ---------------------------------------------------------------------------


def test_creative_judgement_refuses_at_49_permits_at_50() -> None:
    # Below spend_commitment_factor, so this is a creative judgement and the
    # 50/7d gate is the one that applies.
    move = Scale(ANGLE, 1.10)
    low = ENGINE.check(move, gate_context(settled_7d=49))
    assert low.legal is False
    assert low.rule == "CREATIVE_JUDGEMENT"
    assert low.source
    high = ENGINE.check(move, gate_context(settled_7d=50))
    assert high.legal is True


def test_scale_magnitude_selects_which_gate_applies() -> None:
    """Neither evidence gate is dead code.

    Evidence enough to nudge a live angle is not evidence enough to fund it.
    Routing every Scale through both gates would retire the creative gate,
    since anything clearing 300/14d clears 50/7d; this test fails first if
    that ever happens.
    """
    creative_only = gate_context(settled_7d=60, settled_14d=60, age_days=7)
    assert ENGINE.check(Scale(ANGLE, 1.10), creative_only).legal is True

    commitment = ENGINE.check(Scale(ANGLE, 1.20), creative_only)
    assert commitment.legal is False
    assert commitment.rule == "SPEND_COMMITMENT"

    funded = gate_context(settled_7d=60, settled_14d=300, age_days=14)
    assert ENGINE.check(Scale(ANGLE, 1.20), funded).legal is True


def test_spend_commitment_refuses_at_299_permits_at_300() -> None:
    move = Scale(ANGLE, 1.20)
    low = ENGINE.check(move, gate_context(settled_7d=60, settled_14d=299))
    assert low.legal is False
    assert low.rule == "SPEND_COMMITMENT"
    assert low.source
    high = ENGINE.check(
        move, gate_context(settled_7d=60, settled_14d=300, age_days=14)
    )
    assert high.legal is True


def test_spend_commitment_requires_elapsed_days() -> None:
    verdict = ENGINE.check(
        Scale(ANGLE, 1.20),
        gate_context(settled_7d=60, settled_14d=300, age_days=13),
    )
    assert verdict.legal is False
    assert verdict.rule == "SPEND_COMMITMENT"


def test_partial_observations_contribute_zero_to_gates() -> None:
    assert settled_count([observation(100, is_partial=True)]) == 0
    assert settled_count([observation(100, is_partial=True), observation(49)]) == 49
    verdict = ENGINE.check(
        Scale(ANGLE, 1.10),
        gate_context(settled_7d=settled_count(
            [observation(100, is_partial=True), observation(49)]
        )),
    )
    assert verdict.legal is False
    assert verdict.rule == "CREATIVE_JUDGEMENT"


def test_below_coverage_floor_refuses_both_scale_and_kill() -> None:
    ctx = gate_context(settled_7d=60, settled_14d=400, coverage=0.49)
    scale = ENGINE.check(Scale(ANGLE, 1.20), ctx)
    kill = ENGINE.check(Kill(ANGLE, "no_signal"), ctx)
    assert scale.legal is False and scale.rule == "ATTRIBUTION_COVERAGE"
    assert kill.legal is False and kill.rule == "ATTRIBUTION_COVERAGE"


def test_coverage_exactly_at_floor_is_judgeable() -> None:
    ctx = gate_context(settled_7d=60, settled_14d=400, coverage=0.5)
    assert ENGINE.check(Scale(ANGLE, 1.20), ctx).legal is True
    assert ENGINE.check(Kill(ANGLE, "done"), ctx).legal is True


# ---------------------------------------------------------------------------
# Velocity, cadence, drawdown.
# ---------------------------------------------------------------------------


def test_velocity_exactly_at_cap_is_legal() -> None:
    assert ENGINE.check(Scale(ANGLE, 1.20), gate_context(60)).legal is True


def test_velocity_above_cap_is_illegal() -> None:
    verdict = ENGINE.check(Scale(ANGLE, 1.21), gate_context(60))
    assert verdict.legal is False
    assert verdict.rule == "VELOCITY_48H"
    assert verdict.source


def test_velocity_accumulates_within_the_48h_window() -> None:
    ctx = gate_context(60)
    ctx.allocation[ANGLE] = 110.0
    now = datetime(2026, 8, 21, 12, 0, 0)
    ctx.allocation_history = [
        AllocationChange(
            at=now - timedelta(hours=96), angle=ANGLE, allocation=100.0
        ),
        AllocationChange(
            at=now - timedelta(hours=24), angle=ANGLE, allocation=110.0
        ),
    ]
    # 110 * 1.10 = 121 vs baseline 100 -> +21%, past the cap.
    assert ENGINE.check(Scale(ANGLE, 1.10), ctx).legal is False
    # 110 * 1.09 = 119.9 -> +19.9%, inside.
    assert ENGINE.check(Scale(ANGLE, 1.09), ctx).legal is True


def test_carousel_cap_refuses_the_eleventh() -> None:
    ctx = gate_context(60)
    ctx.published_today_by_format[Format.CAROUSEL] = 10
    verdict = ENGINE.check(publish(), ctx)
    assert verdict.legal is False
    assert verdict.rule == "CADENCE_FORMAT_CAP"
    ctx.published_today_by_format[Format.CAROUSEL] = 9
    assert ENGINE.check(publish(), ctx).legal is True


def test_lane_split_at_tolerance_boundary_is_legal() -> None:
    ctx = gate_context(60)
    ctx.acquisition_posts_today = 3
    ctx.service_posts_today = 1
    # 4 / 5 = 0.80 exactly: the ceiling itself is allowed.
    assert ENGINE.check(publish(), ctx).legal is True


def test_lane_split_past_ceiling_is_illegal() -> None:
    ctx = gate_context(60)
    ctx.acquisition_posts_today = 9
    ctx.service_posts_today = 1
    verdict = ENGINE.check(publish(cta_mode=CtaMode.START_TRIAL), ctx)
    assert verdict.legal is False
    assert verdict.rule == "CADENCE_LANE_SPLIT"


def test_lane_split_service_post_past_floor_is_illegal() -> None:
    ctx = gate_context(60)
    ctx.acquisition_posts_today = 1
    ctx.service_posts_today = 10
    verdict = ENGINE.check(publish(cta_mode=CtaMode.REPLY_KEYWORD), ctx)
    assert verdict.legal is False
    assert verdict.rule == "CADENCE_LANE_SPLIT"


def test_lane_split_undefined_on_empty_day() -> None:
    ctx = gate_context(60)
    assert ENGINE.check(publish(), ctx).legal is True


def test_drawdown_freezes_scale_and_forces_kill() -> None:
    ctx = gate_context(60)
    ctx.angle_spend[ANGLE] = 1200.0
    ctx.angle_total_conversions[ANGLE] = 0
    scale = ENGINE.check(Scale(ANGLE, 1.20), ctx)
    assert scale.legal is False
    assert scale.rule == "DRAWDOWN"
    # Kill becomes legal even though the evidence gates have not opened.
    kill = ENGINE.check(Kill(ANGLE, "drawdown"), ctx)
    assert kill.legal is True


def test_drawdown_overridden_by_any_conversion() -> None:
    ctx = gate_context(60)
    ctx.angle_spend[ANGLE] = 1200.0
    ctx.angle_total_conversions[ANGLE] = 1
    assert ENGINE.check(Scale(ANGLE, 1.20), ctx).rule != "DRAWDOWN"


# ---------------------------------------------------------------------------
# Compliance.
# ---------------------------------------------------------------------------


def test_untracked_publish_is_illegal() -> None:
    verdict = ENGINE.check(publish(utm_content=""), gate_context(60))
    assert verdict.legal is False
    assert verdict.rule == "TRACKED_OUTPUT"


def test_outcome_promise_requires_substantiation() -> None:
    ctx = gate_context(60)
    verdict = ENGINE.check(publish(claim_class=ClaimClass.OUTCOME_PROMISE), ctx)
    assert verdict.legal is False
    assert verdict.rule == "OUTCOME_PROMISE"
    assert verdict.source
    ctx.substantiated_claims.add(ANGLE)
    assert ENGINE.check(publish(claim_class=ClaimClass.OUTCOME_PROMISE), ctx).legal is True


def test_health_claims_must_be_mechanism() -> None:
    ctx = gate_context(60)
    ctx.health_related = True
    # Substantiate the claim so the generic outcome rule passes and the
    # health-specific one is what refuses it.
    ctx.substantiated_claims.add(ANGLE)
    verdict = ENGINE.check(publish(claim_class=ClaimClass.OUTCOME_PROMISE), ctx)
    assert verdict.legal is False
    assert verdict.rule == "HEALTH_MECHANISM"
    assert verdict.source
    assert ENGINE.check(publish(claim_class=ClaimClass.MECHANISM), ctx).legal is True


def test_ai_testimonial_is_illegal() -> None:
    ctx = gate_context(60)
    ctx.ai_generated_faces = True
    verdict = ENGINE.check(
        publish(claim_class=ClaimClass.THIRD_PARTY_TESTIMONIAL), ctx
    )
    assert verdict.legal is False
    assert verdict.rule == "TESTIMONIAL_AUTHENTICITY"
    assert verdict.source


def test_tiktok_ai_disclosure_required() -> None:
    ctx = gate_context(60)
    ctx.ai_generated_faces = True
    ctx.ai_disclosure_present = False
    verdict = ENGINE.check(publish(platform=Platform.TIKTOK), ctx)
    assert verdict.legal is False
    assert verdict.rule == "AI_DISCLOSURE_TIKTOK"
    assert verdict.source
    ctx.ai_disclosure_present = True
    assert ENGINE.check(publish(platform=Platform.TIKTOK), ctx).legal is True
    # Off TikTok the flag does not matter.
    ctx.ai_disclosure_present = False
    assert ENGINE.check(publish(platform=Platform.X), ctx).legal is True


def test_instagram_before_after_is_illegal() -> None:
    ctx = gate_context(60)
    ctx.before_after_imagery = True
    verdict = ENGINE.check(publish(platform=Platform.INSTAGRAM), ctx)
    assert verdict.legal is False
    assert verdict.rule == "BEFORE_AFTER_INSTAGRAM"
    assert verdict.source
    assert ENGINE.check(publish(platform=Platform.X), ctx).legal is True


def test_comparative_claims_require_policy_opt_in() -> None:
    ctx = gate_context(60)
    verdict = ENGINE.check(publish(claim_class=ClaimClass.COMPARATIVE), ctx)
    assert verdict.legal is False
    assert verdict.rule == "COMPARATIVE_CLAIM"
    assert verdict.source
    permissive = LegalityEngine(OperatorPolicy(comparative_allowed=True))
    assert permissive.check(publish(claim_class=ClaimClass.COMPARATIVE), ctx).legal is True


def test_low_standing_restricts_to_low_risk_claims() -> None:
    ctx = gate_context(60)
    ctx.account_standing = 0.3
    ctx.substantiated_claims.add(ANGLE)
    verdict = ENGINE.check(publish(claim_class=ClaimClass.OUTCOME_PROMISE), ctx)
    assert verdict.legal is False
    assert verdict.rule == "STANDING_FLOOR"
    assert verdict.source
    assert ENGINE.check(publish(claim_class=ClaimClass.MECHANISM), ctx).legal is True


def test_hold_is_always_legal() -> None:
    assert ENGINE.check(Hold(), gate_context(0)).legal is True


def test_legal_moves_filters_and_preserves_order() -> None:
    ctx = gate_context(60)
    candidates: list[Move] = [
        publish(utm_content=""),
        Hold(),
        publish(),
        Scale(ANGLE, 2.00),
    ]
    assert ENGINE.legal_moves(candidates, ctx) == [Hold(), publish()]


def test_legal_moves_on_empty_candidates() -> None:
    assert ENGINE.legal_moves([], gate_context(60)) == []


# ---------------------------------------------------------------------------
# Payoff.
# ---------------------------------------------------------------------------


ECONOMICS = Economics(
    arpu_monthly=100.0,
    gross_margin=0.8,
    monthly_churn=0.08,
    cogs_share=0.2,
    fixed_cost_per_post=50.0,
)


def test_ltv_worked_example() -> None:
    assert ltv(ECONOMICS) == pytest.approx(1000.0)


def test_ltv_zero_churn_is_infinite() -> None:
    zero_churn = Economics(
        arpu_monthly=100.0,
        gross_margin=0.8,
        monthly_churn=0.0,
        cogs_share=0.2,
        fixed_cost_per_post=50.0,
    )
    assert math.isinf(ltv(zero_churn))
    assert ltv_cac_ratio(zero_churn, 65.0) == pytest.approx(math.inf)


def test_true_conversions_corrections_pull_opposite_ways() -> None:
    base = true_conversions(observation(10))
    assert base == pytest.approx(10.0)
    assert true_conversions(observation(10, coverage=0.5)) == pytest.approx(20.0)
    assert true_conversions(observation(10, incrementality=0.5)) == pytest.approx(5.0)
    both = true_conversions(observation(10, coverage=0.5, incrementality=0.5))
    assert both == pytest.approx(10.0)


def test_reward_worked_example() -> None:
    obs = observation(10, coverage=0.5, incrementality=0.8)
    assert true_conversions(obs) == pytest.approx(16.0)
    assert reward(obs, ECONOMICS, 600.0) == pytest.approx(15_400.0)


def test_cac_ratios_and_payback() -> None:
    assert ltv_cac_ratio(ECONOMICS, 65.0) == pytest.approx(1000.0 / 65.0)
    assert cac_payback_months(65.0, ECONOMICS) == pytest.approx(0.8125)
    assert Economics.__dataclass_fields__["allowable_cac_share"].default == 0.30


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.5, PayoffHealth.STOP),
        (0.99, PayoffHealth.STOP),
        (1.0, PayoffHealth.MARGINAL),
        (1.9, PayoffHealth.MARGINAL),
        (2.0, PayoffHealth.HEALTHY),
        (2.9, PayoffHealth.HEALTHY),
        (3.0, PayoffHealth.GOLDEN),
        (5.0, PayoffHealth.GOLDEN),
        (5.1, PayoffHealth.UNDERSPENDING),
        (math.inf, PayoffHealth.UNDERSPENDING),
    ],
)
def test_health_bands(ratio: float, expected: PayoffHealth) -> None:
    assert health(ratio) is expected


def test_reward_ignores_reach_by_design() -> None:
    # Reach is not in the reward and there is no vanity_penalty helper; the
    # omission is the point (see payoff module docstring).
    assert not hasattr(payoff_module, "vanity_penalty")
    assert "reach" not in inspect.signature(payoff_module.reward).parameters
