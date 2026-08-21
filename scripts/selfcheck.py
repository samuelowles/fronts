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
from blotto.game import action_space, legality, payoff, priors
from blotto.game.action_space import ActionCodec, AngleRegistry
from blotto.game.legality import (
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
    assert hasattr(action_space, "AngleRegistry")
    assert hasattr(action_space, "enumerate_publishes")
    assert hasattr(action_space, "utm_content_id")
    assert hasattr(legality, "LegalityEngine")
    assert hasattr(payoff, "reward")
    assert hasattr(priors, "ARCHETYPE_PRIORS")
    assert AngleRegistry() is not None


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
