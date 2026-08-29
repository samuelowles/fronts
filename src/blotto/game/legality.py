"""The legality engine: rules that forbid a move, not penalties that
discourage it.

Why this module is the heart of the game rather than a compliance afterthought:
``CodeWorldModel.get_legal_actions`` is the entire interface between a planner
and the consequences it is allowed to consider. Anything excluded here is a
mistake the system *cannot make* -- an unsubstantiated health claim, a budget
commitment resting on five conversions, a sixth text thread past the day's
cadence. Anything merely scored or penalised is a mistake the system can still
choose under uncertainty, which is exactly the failure mode both corpora
identify as the way distribution budgets die: premature scaling on thin
evidence.

So every rule below returns a ``Verdict``, and a gate returns
``legal=False`` or it has failed at its only job. No warnings, no scores.

The counts the engine consumes are *settled* counts -- observations still
inside the 24-72h reporting lag never open a gate. Build contexts through
``settled_count`` so that rule is enforced in one place rather than trusted
as a convention at every call site.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from blotto.game.types import (
    ClaimClass,
    CtaMode,
    Format,
    Hold,
    Kill,
    Move,
    Observation,
    Platform,
    Publish,
    Scale,
)

__all__ = [
    "Verdict",
    "OperatorPolicy",
    "AllocationChange",
    "LegalityContext",
    "LegalityEngine",
    "settled_count",
    "LOW_RISK_CLAIM_CLASSES",
    "ACQUISITION_CTAS",
    "SERVICE_CTAS",
]


# Claim classes an account in poor standing is still allowed to make. The
# dividing line is verifiability at zero marginal risk: mechanism and own-data
# statements cannot become an FTC problem, while outcome promises, testimonials
# and comparisons each carry a distinct compliance failure mode.
LOW_RISK_CLAIM_CLASSES: frozenset[ClaimClass] = frozenset(
    {
        ClaimClass.NO_CLAIM,
        ClaimClass.MECHANISM,
        ClaimClass.VERIFIABLE_METRIC,
        ClaimClass.FIRST_PARTY_PROOF,
    }
)

# Which side of the acquisition/service lane split each CTA serves. Hard CTAs
# ask for a commitment; soft CTAs build the audience that makes later hard
# CTAs land. The day's mix between them is a cadence constraint, not a taste.
ACQUISITION_CTAS: frozenset[CtaMode] = frozenset(
    {
        CtaMode.START_TRIAL,
        CtaMode.BOOK_DEMO,
        CtaMode.INSTALL_APP,
        CtaMode.JOIN_WAITLIST,
    }
)
SERVICE_CTAS: frozenset[CtaMode] = frozenset(
    {CtaMode.REPLY_KEYWORD, CtaMode.FOLLOW_FOR_SERIES}
)

# Exact-boundary tolerance for ratio comparisons, so that "capped at +20%"
# means exactly +20.0% is legal and 20.000001% is not, without floating-point
# noise deciding which side of the line a move lands on.
_RATIO_EPS = 1e-9

_CREATIVE_JUDGEMENT_SOURCE = (
    "Storytelling Engineer/"
    "17_A_B_Testing_Story_Arcs_Statistical_Significance_in_Emotion.md"
)
_SPEND_COMMITMENT_SOURCE = (
    "GTM Engineer/Encyclopedia/08_Landing_Pages_and_CRO.md; "
    "GTM Engineer/Encyclopedia/04_The_100x_Engineer_Mindset.md"
)
_VELOCITY_SOURCE = (
    "DTC Engineer/21_Media_Buying_Risk_Management_and_Drawdowns.md; "
    "GTM Engineer/Encyclopedia/03_Channel_Playbooks_2026.md"
)
_COMPLIANCE_SOURCE = (
    "GTM Engineer/Encyclopedia/24_GTM_Legal_and_Compliance.md"
)
_CLAIMS_SOURCE = (
    "Storytelling Engineer/"
    "20_Compliance_and_Trust_Navigating_Claims_with_Authentic_Proof.md"
)
_COMPARATIVE_SOURCE = (
    "Storytelling Engineer/"
    "09_Creating_the_Enemy_Polarization_as_a_CTR_Mechanic.md"
)
_DRAWDOWN_SOURCE = (
    "DTC Engineer/21_Media_Buying_Risk_Management_and_Drawdowns.md"
)
_TRACKED_SOURCE = "blotto/game/types.py (Observation join contract)"
_CADENCE_SOURCE = (
    "GTM Engineer/Encyclopedia/02_Metrics_and_Baselines_2026.md"
)


@dataclass(frozen=True, slots=True)
class Verdict:
    """The result of one rule. Carries its own citation so that any refusal
    can be answered with the corpus passage that motivates it -- an operator
    asked to eat a day of silence is owed the reason, in source form."""

    legal: bool
    rule: str
    reason: str
    source: str


def _legal(rule: str, source: str) -> Verdict:
    return Verdict(legal=True, rule=rule, reason="", source=source)


def _illegal(rule: str, reason: str, source: str) -> Verdict:
    return Verdict(legal=False, rule=rule, reason=reason, source=source)


@dataclass(slots=True)
class OperatorPolicy:
    """Tunable thresholds. Defaults are the corpus figures; the knobs exist
    because an operator with different risk tolerance should change policy,
    not code. ``drawdown_spend_limit`` is the exception -- it is an operator
    risk choice, not a corpus number, and carries no citation for that reason.

    The band lower edges (3 text threads/day, the 0.60 acquisition floor) are
    targets a day's *plan* must satisfy, not per-move gates: no single move
    can be illegal for being insufficient, only for being excessive.
    """

    # Evidence gates.
    creative_judgement_min_conversions: int = 50
    creative_judgement_window_days: int = 7
    spend_commitment_min_conversions: int = 300
    spend_commitment_min_days: int = 14
    attribution_coverage_floor: float = 0.5

    spend_commitment_factor: float = 1.20
    """The Scale factor at or above which a move stops being a creative
    judgement and becomes a budget commitment.

    This threshold is what keeps the two evidence gates from collapsing into
    one. They license different decisions and therefore attach to different
    moves: 50 conversions in 7 days is enough to say "this arc works" and nudge
    a live angle; committing real money behind it needs 300 conversions and 14
    days. Gating every Scale on both would make the creative gate dead code,
    since anything clearing 300/14d clears 50/7d.

    The default coincides with ``max_allocation_increase_48h``, which is not an
    accident: a move inside the velocity limit is by construction a nudge, and
    a move that exceeds it is by construction a commitment. An operator who
    widens one should think about the other."""

    # Velocity.
    max_allocation_increase_48h: float = 0.20

    # Cadence.
    max_per_day_by_format: dict[Format, int] = field(
        default_factory=lambda: {
            Format.CAROUSEL: 10,
            Format.TEXT_THREAD: 5,
        }
    )
    acquisition_ratio_target: float = 0.70
    acquisition_ratio_tolerance: float = 0.10

    # Compliance.
    comparative_allowed: bool = False

    # Risk. Operator risk tolerance, deliberately uncited.
    drawdown_spend_limit: float = 1000.0

    # Standing.
    standing_floor: float = 0.5


@dataclass(frozen=True, slots=True)
class AllocationChange:
    """One entry in the allocation history. The first entry for an angle
    should be its opening allocation, so velocity is measurable from the
    moment the angle exists."""

    at: datetime
    angle: str
    allocation: float


@dataclass(slots=True)
class LegalityContext:
    """Everything the rules need to know about the operator's situation.

    Counts and coverage arrive per angle and pre-windowed; the engine does not
    see raw observations (except through ``settled_count`` at build time), so
    a partial observation cannot leak into a gate by accident of ordering.
    Unknown angles default to the conservative reading: no coverage data
    means unjudgeable, not clean.
    """

    current_date: date
    settled_conversions_7d: dict[str, int] = field(default_factory=dict)
    settled_conversions_14d: dict[str, int] = field(default_factory=dict)
    angle_reach: dict[str, int] = field(default_factory=dict)
    angle_attribution_coverage: dict[str, float] = field(default_factory=dict)
    angle_spend: dict[str, float] = field(default_factory=dict)
    angle_total_conversions: dict[str, int] = field(default_factory=dict)
    allocation: dict[str, float] = field(default_factory=dict)
    allocation_history: list[AllocationChange] = field(default_factory=list)
    angle_age_days: dict[str, int] = field(default_factory=dict)
    substantiated_claims: set[str] = field(default_factory=set)
    published_today_by_format: dict[Format, int] = field(default_factory=dict)
    acquisition_posts_today: int = 0
    service_posts_today: int = 0
    account_standing: float = 1.0
    platform_strikes: dict[Platform, int] = field(default_factory=dict)
    ai_generated_faces: bool = False
    before_after_imagery: bool = False
    ai_disclosure_present: bool = False
    """Whether the current creative pipeline's AI-generated faces carry the
    platform disclosure. Required on TikTok; the flag is context, not move,
    state because it describes the production process behind the post."""
    health_related: bool = False
    """Whether the campaign's subject matter is health or medical, which
    raises the claim bar from substantiated to mechanism-only."""


def settled_count(observations: Iterable[Observation]) -> int:
    """Sum conversions over observations the reporting lag has finished with.

    The only sanctioned way to derive the counts ``LegalityContext`` consumes.
    ``is_partial`` observations are dropped entirely -- not prorated, not
    estimated -- because peeking inside the 24-72h lag is the specific failure
    the gates exist to prevent (DTC 21_Media_Buying_Risk_Management...md).
    """
    return sum(
        obs.attributed_conversions for obs in observations if not obs.is_partial
    )


class LegalityEngine:
    """Applies the rules, in an order chosen so the most consequential
    refusal is the one reported: drawdown and unjudgeability before evidence,
    evidence before velocity, compliance before cadence."""

    def __init__(self, policy: OperatorPolicy | None = None) -> None:
        self.policy = policy if policy is not None else OperatorPolicy()

    # -- Public API -----------------------------------------------------------

    def check(self, move: Move, ctx: LegalityContext) -> Verdict:
        """Return the first failing rule's Verdict, or a legal Verdict."""
        if isinstance(move, Publish):
            verdict = self._check_tracked(move, ctx)
            if not verdict.legal:
                return verdict
            for rule in (
                self._check_outcome_promise,
                self._check_health_mechanism,
                self._check_testimonial_authenticity,
                self._check_tiktok_ai_disclosure,
                self._check_instagram_before_after,
                self._check_comparative_claim,
                self._check_standing,
                self._check_format_cap,
                self._check_lane_split,
            ):
                verdict = rule(move, ctx)
                if not verdict.legal:
                    return verdict
            return verdict
        if isinstance(move, Scale):
            # Which evidence gate applies depends on what the move actually
            # commits. A nudge inside the velocity limit is a creative
            # judgement and answers to 50/7d; anything larger puts money behind
            # the belief and answers to 300/14d. Applying both to every Scale
            # would retire the creative gate, since 300/14d strictly implies it.
            commits_budget = move.factor >= self.policy.spend_commitment_factor
            evidence_gate = (
                self._check_spend_commitment
                if commits_budget
                else self._check_creative_judgement
            )
            # Named separately from the Publish loop's ``rule`` because the
            # two dispatch over different Move types; one shared name would
            # force both into a common callable type.
            for scale_rule in (
                self._check_drawdown_scale,
                self._check_coverage,
                evidence_gate,
                self._check_velocity,
            ):
                verdict = scale_rule(move, ctx)
                if not verdict.legal:
                    return verdict
            return verdict
        if isinstance(move, Kill):
            drawdown = self._drawdown_state(move.angle, ctx)
            if drawdown:
                return _legal(
                    "DRAWDOWN",
                    f"{_DRAWDOWN_SOURCE} -- kill is the prescribed action",
                )
            return self._check_coverage(move, ctx)
        if isinstance(move, Hold):
            return _legal("HOLD", "blotto/game/types.py (Hold)")
        raise TypeError(f"not a Move: {move!r}")

    def legal_moves(
        self,
        candidates: Iterable[Move],
        ctx: LegalityContext,
    ) -> list[Move]:
        """Filter to the moves a planner may select from. Order-preserving."""
        return [move for move in candidates if self.check(move, ctx).legal]

    # -- Evidence gates ---------------------------------------------------

    def _check_creative_judgement(
        self, move: Scale, ctx: LegalityContext
    ) -> Verdict:
        """50 settled conversions in a trailing 7-day window before an angle
        may be declared a winner.

        Calling an arc at 5 conversions is the failure the corpus measures
        directly: CPA "frequently explode[s] to $150" on scale-up. This gate
        licenses the creative-level judgement that precedes any spend
        commitment, so it is checked before -- and independently of -- the
        heavier spend gate below.
        """
        settled = ctx.settled_conversions_7d.get(move.angle, 0)
        required = self.policy.creative_judgement_min_conversions
        if settled < required:
            return _illegal(
                "CREATIVE_JUDGEMENT",
                f"angle {move.angle!r} has {settled} settled conversions in "
                f"the trailing {self.policy.creative_judgement_window_days}d; "
                f"declaring a winner requires {required}",
                _CREATIVE_JUDGEMENT_SOURCE,
            )
        return _legal("CREATIVE_JUDGEMENT", _CREATIVE_JUDGEMENT_SOURCE)

    def _check_spend_commitment(
        self, move: Scale, ctx: LegalityContext
    ) -> Verdict:
        """300 settled conversions per variant and 14 days elapsed before a
        Scale committing budget is legal.

        The two thresholds differ because the decisions differ: 50/7d licenses
        saying "this arc works", 300/14d licenses putting money behind it. The
        elapsed-time requirement kills the peeking failure that sample size
        alone cannot -- a burst of conversions on day 2 is indistinguishable
        from a trend until the trend has had time to die.
        """
        settled = ctx.settled_conversions_14d.get(move.angle, 0)
        required = self.policy.spend_commitment_min_conversions
        if settled < required:
            return _illegal(
                "SPEND_COMMITMENT",
                f"angle {move.angle!r} has {settled} settled conversions per "
                f"variant over 14d; committing budget requires {required}",
                _SPEND_COMMITMENT_SOURCE,
            )
        age = ctx.angle_age_days.get(move.angle, 0)
        if age < self.policy.spend_commitment_min_days:
            return _illegal(
                "SPEND_COMMITMENT",
                f"angle {move.angle!r} is {age} days old; committing budget "
                f"requires {self.policy.spend_commitment_min_days} days "
                "elapsed",
                _SPEND_COMMITMENT_SOURCE,
            )
        return _legal("SPEND_COMMITMENT", _SPEND_COMMITMENT_SOURCE)

    def _check_coverage(self, move: Scale | Kill, ctx: LegalityContext) -> Verdict:
        """Below the attribution floor an angle is UNJUDGEABLE: both Scale
        and Kill are illegal.

        This is deliberately non-obvious and cuts against instinct. Coverage
        at 0.4 does not make an angle look 40% as good -- it makes the
        *direction* of the estimate unknowable, because what you cannot see is
        not missing at random (ad blockers skew toward the technical, ITP
        toward the affluent). You cannot conclude anything from data you
        mostly cannot see, and Kill is a conclusion just as much as Scale is.
        Killing an angle on unseeable data buries working creative as
        reliably as scaling buries failing creative. The escape is the
        drawdown rule, which fires on spend -- a number the operator *does*
        see in full -- rather than on conversions.
        """
        coverage = ctx.angle_attribution_coverage.get(move.angle, 0.0)
        floor = self.policy.attribution_coverage_floor
        if coverage < floor - _RATIO_EPS:
            return _illegal(
                "ATTRIBUTION_COVERAGE",
                f"angle {move.angle!r} attribution coverage {coverage:.2f} is "
                f"below the floor {floor:.2f}; the angle is unjudgeable, so "
                "neither Scale nor Kill may conclude anything about it",
                "GTM Engineer/Encyclopedia/09_Attribution_and_Analytics.md",
            )
        return _legal(
            "ATTRIBUTION_COVERAGE",
            "GTM Engineer/Encyclopedia/09_Attribution_and_Analytics.md",
        )

    def _drawdown_state(self, angle: str, ctx: LegalityContext) -> bool:
        """True when spend exceeds the limit with zero conversions to show."""
        spend = ctx.angle_spend.get(angle, 0.0)
        conversions = ctx.angle_total_conversions.get(angle, 0)
        return spend > self.policy.drawdown_spend_limit and conversions == 0

    def _check_drawdown_scale(
        self, move: Scale, ctx: LegalityContext
    ) -> Verdict:
        """Spend past the drawdown limit with zero conversions freezes
        scaling and forces the kill decision onto the table.

        Drawdown discipline is the one risk rule that must override evidence
        gates rather than sit beside them: waiting for 300 conversions against
        an angle that converts nothing is how a drawdown becomes a habit. Note
        the asymmetry with the coverage rule -- spend is fully observable, so
        this conclusion does not depend on the attribution floor.
        """
        if self._drawdown_state(move.angle, ctx):
            return _illegal(
                "DRAWDOWN",
                f"angle {move.angle!r} has "
                f"${ctx.angle_spend.get(move.angle, 0.0):.2f} spend and zero "
                f"conversions; further Scale is frozen and Kill is the "
                "prescribed move",
                _DRAWDOWN_SOURCE,
            )
        return _legal("DRAWDOWN", _DRAWDOWN_SOURCE)

    # -- Velocity -----------------------------------------------------------

    def _check_velocity(self, move: Scale, ctx: LegalityContext) -> Verdict:
        """Allocation increases are capped at +20% per rolling 48 hours.

        Larger jumps reset the platform's learning phase: the classifier that
        had begun to find the audience is asked to start over, and every
        creative-lifespan figure in the priors assumes it was allowed to
        finish. The baseline is the last recorded allocation older than 48h;
        with none, the current allocation stands as baseline, so a sequence of
        small scales still accumulates against the cap.
        """
        window_start = datetime.combine(ctx.current_date, time()) - timedelta(
            hours=48
        )
        latest_before_window: AllocationChange | None = None
        for change in ctx.allocation_history:
            if change.angle != move.angle or change.at >= window_start:
                continue
            if (
                latest_before_window is None
                or change.at > latest_before_window.at
            ):
                latest_before_window = change
        baseline = (
            latest_before_window.allocation
            if latest_before_window is not None
            else ctx.allocation.get(move.angle, 0.0)
        )

        current = ctx.allocation.get(move.angle, 0.0)
        projected = current * move.factor
        if baseline > 0.0:
            increase = projected / baseline - 1.0
            if increase > self.policy.max_allocation_increase_48h + _RATIO_EPS:
                return _illegal(
                    "VELOCITY_48H",
                    f"projected allocation {projected:.2f} is a "
                    f"{increase:.0%} increase over the 48h baseline "
                    f"{baseline:.2f}; the cap is "
                    f"+{self.policy.max_allocation_increase_48h:.0%}",
                    _VELOCITY_SOURCE,
                )
        elif projected > 0.0:
            # Baseline of zero with money now behind it: an unbounded jump.
            return _illegal(
                "VELOCITY_48H",
                f"angle {move.angle!r} had zero allocation 48h ago and now "
                f"projects {projected:.2f}; any increase from zero exceeds "
                "the cap",
                _VELOCITY_SOURCE,
            )
        return _legal("VELOCITY_48H", _VELOCITY_SOURCE)

    # -- Cadence ------------------------------------------------------------

    def _check_format_cap(self, move: Publish, ctx: LegalityContext) -> Verdict:
        """Per-day caps per format.

        The caps exist because creative volume is a production constraint
        (GTM 11) and because carpet-bombing one format trains the audience to
        scroll past it. The band's lower edges -- 3 text threads/day -- are
        planning targets and are deliberately not enforced here: no single
        move can be illegal for being insufficient.
        """
        cap = self.policy.max_per_day_by_format.get(move.format)
        if cap is None:
            return _legal("CADENCE_FORMAT_CAP", _CADENCE_SOURCE)
        published = ctx.published_today_by_format.get(move.format, 0)
        if published >= cap:
            return _illegal(
                "CADENCE_FORMAT_CAP",
                f"{published} {move.format.value} posts already published "
                f"today; the cap is {cap}",
                _CADENCE_SOURCE,
            )
        return _legal("CADENCE_FORMAT_CAP", _CADENCE_SOURCE)

    def _check_lane_split(self, move: Publish, ctx: LegalityContext) -> Verdict:
        """Keep the day's acquisition/service mix inside the 0.70 +/- 0.10
        band, checked in the direction the post pushes.

        A hard CTA past the ceiling is a day spent harvesting an audience that
        was never built; soft posts past the floor are a day spent building an
        audience that is never asked. The rule is directional because the
        ratio is undefined until the first post of the day exists.
        """
        is_acquisition = move.cta_mode in ACQUISITION_CTAS
        total = ctx.acquisition_posts_today + ctx.service_posts_today
        if total == 0:
            return _legal("CADENCE_LANE_SPLIT", _CADENCE_SOURCE)

        projected_acq = ctx.acquisition_posts_today + (1 if is_acquisition else 0)
        projected = projected_acq / (total + 1)
        ceiling = (
            self.policy.acquisition_ratio_target
            + self.policy.acquisition_ratio_tolerance
        )
        floor = (
            self.policy.acquisition_ratio_target
            - self.policy.acquisition_ratio_tolerance
        )
        if is_acquisition and projected > ceiling + _RATIO_EPS:
            return _illegal(
                "CADENCE_LANE_SPLIT",
                f"this acquisition post would put the day's split at "
                f"{projected:.2f}, above the ceiling {ceiling:.2f}",
                _CADENCE_SOURCE,
            )
        if not is_acquisition and projected < floor - _RATIO_EPS:
            return _illegal(
                "CADENCE_LANE_SPLIT",
                f"this service post would put the day's split at "
                f"{projected:.2f}, below the floor {floor:.2f}",
                _CADENCE_SOURCE,
            )
        return _legal("CADENCE_LANE_SPLIT", _CADENCE_SOURCE)

    # -- Compliance -----------------------------------------------------------

    def _check_tracked(self, move: Publish, ctx: LegalityContext) -> Verdict:
        """A Publish with empty utm_content is illegal.

        Untracked output cannot produce an ``Observation``, and a move whose
        outcome can never be observed is a move the planner cannot learn
        from -- it poisons every downstream estimate by being invisible where
        its consequences land. This is a system invariant rather than a corpus
        finding, so it cites the contract that makes it true.
        """
        if not move.utm_content:
            return _illegal(
                "TRACKED_OUTPUT",
                "publish has empty utm_content; untracked output cannot be "
                "learned from",
                _TRACKED_SOURCE,
            )
        return _legal("TRACKED_OUTPUT", _TRACKED_SOURCE)

    def _check_outcome_promise(
        self, move: Publish, ctx: LegalityContext
    ) -> Verdict:
        """OUTCOME_PROMISE claims are illegal unless the angle is in the
        substantiated set.

        The FTC requires every claim be provable at the moment it is made,
        not provable once results come in. "Substantiated" is a property of
        the angle, not the post, because a promise is a promise regardless of
        which creative carries it.
        """
        if (
            move.claim_class is ClaimClass.OUTCOME_PROMISE
            and move.angle not in ctx.substantiated_claims
        ):
            return _illegal(
                "OUTCOME_PROMISE",
                f"angle {move.angle!r} makes an outcome claim without "
                "substantiation on file",
                _COMPLIANCE_SOURCE,
            )
        return _legal("OUTCOME_PROMISE", _COMPLIANCE_SOURCE)

    def _check_health_mechanism(
        self, move: Publish, ctx: LegalityContext
    ) -> Verdict:
        """Health outcome claims must be expressed as MECHANISM, not outcome.

        In health contexts an outcome promise is a medical claim, and medical
        claims carry a burden no marketing substantiation can meet. Naming
        how something works makes no promise about what it cures, which is
        the compliance-safe shape and -- not coincidentally -- the shape with
        separating power.
        """
        if ctx.health_related and move.claim_class is ClaimClass.OUTCOME_PROMISE:
            return _illegal(
                "HEALTH_MECHANISM",
                "health-related content may claim mechanism, never outcomes",
                _CLAIMS_SOURCE,
            )
        return _legal("HEALTH_MECHANISM", _CLAIMS_SOURCE)

    def _check_testimonial_authenticity(
        self, move: Publish, ctx: LegalityContext
    ) -> Verdict:
        """A THIRD_PARTY_TESTIMONIAL behind an AI-generated face is illegal.

        A testimonial's entire evidentiary value is that a real human chose to
        stake their reputation on it. A generated face is not a hesitant
        witness; it is copy wearing a face, and presenting it as a testimonial
        is the definition of a deceptive endorsement.
        """
        if (
            move.claim_class is ClaimClass.THIRD_PARTY_TESTIMONIAL
            and ctx.ai_generated_faces
        ):
            return _illegal(
                "TESTIMONIAL_AUTHENTICITY",
                "third-party testimonial produced with AI-generated faces; a "
                "testimonial must be a verified human",
                _CLAIMS_SOURCE,
            )
        return _legal("TESTIMONIAL_AUTHENTICITY", _CLAIMS_SOURCE)

    def _check_tiktok_ai_disclosure(
        self, move: Publish, ctx: LegalityContext
    ) -> Verdict:
        """AI-generated faces on TikTok require the AI disclosure."""
        if (
            move.platform is Platform.TIKTOK
            and ctx.ai_generated_faces
            and not ctx.ai_disclosure_present
        ):
            return _illegal(
                "AI_DISCLOSURE_TIKTOK",
                "AI-generated faces on TikTok require ai_disclosure_present",
                _COMPLIANCE_SOURCE,
            )
        return _legal("AI_DISCLOSURE_TIKTOK", _COMPLIANCE_SOURCE)

    def _check_instagram_before_after(
        self, move: Publish, ctx: LegalityContext
    ) -> Verdict:
        """Before/after imagery is illegal outright on Instagram."""
        if move.platform is Platform.INSTAGRAM and ctx.before_after_imagery:
            return _illegal(
                "BEFORE_AFTER_INSTAGRAM",
                "before/after imagery is not permitted on Instagram",
                _CLAIMS_SOURCE,
            )
        return _legal("BEFORE_AFTER_INSTAGRAM", _CLAIMS_SOURCE)

    def _check_comparative_claim(
        self, move: Publish, ctx: LegalityContext
    ) -> Verdict:
        """COMPARATIVE claims naming a competitor require an explicit policy
        opt-in.

        Named-enemy framing is the highest-CTR mechanic in the corpus and also
        the one platform review rejects: Meta refuses ads that overly
        disparage a named brand. High reward, irreversible rejection risk --
        exactly the trade a policy flag exists for, not a default.
        """
        if (
            move.claim_class is ClaimClass.COMPARATIVE
            and not self.policy.comparative_allowed
        ):
            return _illegal(
                "COMPARATIVE_CLAIM",
                "comparative claims naming a competitor require "
                "policy.comparative_allowed=True",
                _COMPARATIVE_SOURCE,
            )
        return _legal("COMPARATIVE_CLAIM", _COMPARATIVE_SOURCE)

    def _check_standing(self, move: Publish, ctx: LegalityContext) -> Verdict:
        """Below the standing floor, only low-risk claim classes are legal.

        Standing is the account's accumulated trust with the platforms, and
        it is the one resource that compounds across every campaign: a strike
        from this week's creative taxes next month's reach. An account in bad
        standing therefore loses the right to make the claims that carry
        strike risk, not the right to post.
        """
        if (
            ctx.account_standing < self.policy.standing_floor - _RATIO_EPS
            and move.claim_class not in LOW_RISK_CLAIM_CLASSES
        ):
            return _illegal(
                "STANDING_FLOOR",
                f"account standing {ctx.account_standing:.2f} is below "
                f"the floor {self.policy.standing_floor:.2f}; only "
                "low-risk claim classes are legal "
                f"({ctx.platform_strikes.get(move.platform, 0)} strikes "
                f"on {move.platform.value})",
                _COMPLIANCE_SOURCE,
            )
        return _legal("STANDING_FLOOR", _COMPLIANCE_SOURCE)
