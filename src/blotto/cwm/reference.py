"""A hand-written reference world model implementing ``CodeWorldModel``.

This is the only place in the repository where the TRUE dynamics are written
down. It serves three purposes, and the third is why it must not be stubbed:

1. Ground truth for tests -- a synthesised model's transition accuracy is
   measured against trajectories this module generates.
2. Host for arena tournaments -- when no ground truth exists for a candidate
   model, some model has to stand in as the arena host, and the best available
   stand-in is this one.
3. Generator of synthetic trajectories -- the LLM's only training signal.

Because the synthesiser sees this model's OUTPUT and never its code, every
dynamic below is filtered through the observation model: the operator's view
is late, partly blind to conversions, and partly fictional about attribution
(``blotto.game.priors`` carries the measured figures). The hidden state --
``theta``, standing, saturation, belief, fatigue -- never crosses that line.
"""

from __future__ import annotations

import copy
import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from typing import Any

from blotto.game.action_space import ActionCodec, enumerate_publishes, utm_content_id
from blotto.game.legality import (
    ACQUISITION_CTAS,
    AllocationChange,
    LegalityContext,
    LegalityEngine,
)
from blotto.game.payoff import Economics, reward
from blotto.game.priors import (
    ARCHETYPE_PRIORS,
    CLIENT_SIDE_EVENT_LOSS,
    DARK_SOCIAL_B2B_SHARE,
    NON_INCREMENTAL_SHARE,
    PLATFORM_PRIORS,
    REPORTING_LAG_HOURS,
    SERVER_SIDE_RECOVERY,
    WINNING_HOOK_BURNOUT_DAYS,
)
from blotto.game.types import (
    CHANCE_PLAYER,
    FIELD,
    OPERATOR,
    PLATFORM,
    TERMINAL_PLAYER,
    ActionKey,
    Archetype,
    AudienceBelief,
    ClaimClass,
    CtaMode,
    EmotionalVector,
    Format,
    HiddenState,
    Hold,
    HookFamily,
    Kill,
    Move,
    Observation,
    Platform,
    Publish,
    RankingWeights,
    Scale,
    SemanticTier,
    State,
    Step,
    Trajectory,
)
from blotto.protocols import CodeWorldModel

__all__ = ["ReferenceConfig", "ReferenceWorldModel"]


_EPOCH = date(2026, 1, 1)
"""Fixed day zero, so posted_at/observed_at timestamps are stable across runs
and a recorded trajectory can be replayed bit-for-bit."""

_RESPONSE_BUCKETS: tuple[tuple[str, float], ...] = (
    ("low", 0.25),
    ("mid", 0.50),
    ("high", 0.25),
)
"""The audience-response lottery: the chance player draws one bucket per day
and the bucket multiplies realised reach. Spread wide enough to matter,
symmetric enough not to bias any archetype."""

_RESPONSE_MULTIPLIER = {"low": 0.70, "mid": 1.00, "high": 1.50}
_SATURATION_ADJUST = {"low": -0.05, "mid": 0.00, "high": 0.05}
_DRIFT_DIRECTIONS: tuple[tuple[str, float], ...] = (
    ("down", -1.0),
    ("flat", 0.0),
    ("up", 1.0),
)
_DRIFT_PROBABILITIES = {"down": 0.3, "flat": 0.4, "up": 0.3}

# CTAs that ask the audience to leave the feed. The ranking function's
# session-time objective punishes exactly these (GAME.md s3: the divergence
# point), which is why the penalty is structural rather than a knob.
_OFF_PLATFORM_CTAS = frozenset(
    {CtaMode.START_TRIAL, CtaMode.BOOK_DEMO, CtaMode.INSTALL_APP, CtaMode.JOIN_WAITLIST}
)

# Claim classes ranked by separating power: how expensive they are for a
# low-quality sender to imitate (types.py, ClaimClass docstring). Credence
# rises on the expensive ones and falls on cheap promises.
_CREDENCE_DELTA = {
    ClaimClass.NO_CLAIM: 0.00,
    ClaimClass.MECHANISM: 0.04,
    ClaimClass.VERIFIABLE_METRIC: 0.04,
    ClaimClass.FIRST_PARTY_PROOF: 0.06,
    ClaimClass.THIRD_PARTY_TESTIMONIAL: -0.02,
    ClaimClass.COMPARATIVE: -0.02,
    ClaimClass.OUTCOME_PROMISE: -0.05,
}
_STRIKE_RISK_CLAIMS = frozenset(
    {ClaimClass.COMPARATIVE, ClaimClass.THIRD_PARTY_TESTIMONIAL}
)

# The engagement priors are measured in aggregate, so the saves/shares/
# comments split of the engagement band is a modelling choice rather than a
# corpus figure. Stated once here, as a split, rather than dissolved into the
# reach formula where nobody could find it to question it.
_ENGAGEMENT_SPLIT = (0.5, 0.3, 0.2)

_STANDING_RECOVERY_PER_DAY = 0.01
_STRIKE_PENALTY = 0.20
_SPEND_PER_ALLOCATION_UNIT = 10.0

_THETA_WEIGHT_KEYS = (
    "hook_rate",
    "hold_rate",
    "saves",
    "shares",
    "comments",
    "dwell",
    "follow_through",
)


def _stable_uniform(seed_text: str, tag: str) -> float:
    """A deterministic U(0,1) draw keyed by (seed_text, tag).

    Observation degradation is a chance process, but ``get_observations``
    must be reproducible or no trajectory could be unit-tested against a
    model. A hash stands in for the draw: the same post always reports the
    same coverage, which is also true of the real thing -- a post's audience
    does not re-randomise its ad-blocker rate every time the dashboard
    refreshes.
    """
    digest = hashlib.sha256(f"{seed_text}|{tag}".encode()).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def _iso(day: int) -> str:
    return (_EPOCH + timedelta(days=day)).isoformat()


@dataclass(slots=True)
class ReferenceConfig:
    """Configuration for the reference world model."""

    horizon: int = 30
    """Days in the episode. One day is one operator ply plus one chance ply."""
    seed: int = 7
    initial_standing: float = 0.7
    field_size: int = 40
    """Competing creators. Sets the base audience the ranking score divides."""
    drift_rate: float = 0.05
    """Per-step magnitude of the theta random walk."""
    economics: Economics = field(
        default_factory=lambda: Economics(
            arpu_monthly=100.0,
            gross_margin=0.8,
            monthly_churn=0.08,
            cogs_share=0.2,
            fixed_cost_per_post=50.0,
        )
    )
    """The worked-example unit economics from ``blotto.game.payoff``'s tests,
    so a default-configured episode reproduces those numbers."""
    angles: tuple[str, ...] = (
        "launch_theater",
        "enemy_of_slop",
        "founder_field_notes",
    )
    avatars: tuple[str, ...] = ("solo_founder", "agency_ops")


class ReferenceWorldModel:
    """Ground-truth dynamics, per ``docs/GAME.md``, behind the CWM protocol.

    Determinism contract: every transition is a pure function of (state,
    action). Randomness enters ONLY through the chance player's outcome,
    which the caller samples from ``chance_outcomes`` -- the paper is strict
    about this and so is the protocol docstring, because hidden
    nondeterminism makes trajectory unit tests impossible.
    """

    def __init__(self, config: ReferenceConfig | None = None) -> None:
        self.config = config if config is not None else ReferenceConfig()
        self.engine = LegalityEngine()
        self.codec = ActionCodec()

    # -- State helpers --------------------------------------------------------

    def _hidden(self, state: State) -> dict[str, Any]:
        hidden: dict[str, Any] = state["hidden"]
        return hidden

    def _day(self, state: State) -> int:
        day: int = state["hidden"]["step"]
        return day

    def _serialise_hidden(self, hidden: HiddenState) -> dict[str, Any]:
        theta = hidden.theta
        return {
            "theta": {
                "hook_rate": theta.hook_rate,
                "hold_rate": theta.hold_rate,
                "saves": theta.saves,
                "shares": theta.shares,
                "comments": theta.comments,
                "dwell": theta.dwell,
                "follow_through": theta.follow_through,
                "external_link_penalty": theta.external_link_penalty,
            },
            "standing": hidden.standing,
            "saturation": dict(hidden.saturation),
            "belief": {
                avatar: {
                    "exposures": belief.exposures,
                    "credence": belief.credence,
                    "fatigue": belief.fatigue,
                }
                for avatar, belief in hidden.belief.items()
            },
            "asset_fatigue": dict(hidden.asset_fatigue),
            "step": hidden.step,
        }

    # -- CodeWorldModel --------------------------------------------------------

    def initial_state(self) -> State:
        hidden = HiddenState(
            theta=RankingWeights(drift_rate=self.config.drift_rate),
            standing=self.config.initial_standing,
            saturation={angle: 0.2 for angle in self.config.angles},
            belief={avatar: AudienceBelief() for avatar in self.config.avatars},
        )
        return State(
            {
                "hidden": self._serialise_hidden(hidden),
                "phase": "operator",
                "pending": [],
                "log": [],
                "true_log": [],
                "utm_angle": {},
                "angle_first_day": {},
                "angle_spend": {},
                "allocation": {},
                "allocation_history": [],
                "strikes": {},
                "published_today": {},
                "acq_today": 0,
                "svc_today": 0,
                "spend_committed": 0.0,
            }
        )

    def apply_action(self, state: State, action: ActionKey) -> State:
        """Return the successor state. Never mutates ``state``."""
        nxt = State(copy.deepcopy(dict(state)))
        if nxt["phase"] == "terminal":
            return nxt
        if nxt["phase"] == "chance":
            self._apply_chance(nxt, action)
        else:
            self._apply_operator(nxt, action)
        return nxt

    def get_current_player(self, state: State) -> int:
        phase = state["phase"]
        if phase == "terminal":
            return TERMINAL_PLAYER
        if phase == "chance":
            return CHANCE_PLAYER
        return OPERATOR

    def get_legal_actions(self, state: State) -> list[ActionKey]:
        """Delegate to ``LegalityEngine``; legality is not reimplemented here.

        The platform and the field never take a ply of their own: P1's
        commitment IS theta (which drifts at chance nodes) and P2's strategy
        IS saturation (which mean-reverts at chance nodes). OpenSpiel still
        gets a chance player it can sample, which is all a planner needs.
        """
        if state["phase"] != "operator":
            return []
        ctx = self._legality_context(state)
        moves: list[Move] = [Hold()]
        for angle, allocation in state["allocation"].items():
            if allocation > 0.0:
                moves.append(Scale(angle, 1.10))
                moves.append(Scale(angle, 1.20))
                moves.append(Kill(angle, "operator_decision"))
        moves.extend(self._candidate_publishes(state))
        return [self.codec.encode(m) for m in self.engine.legal_moves(moves, ctx)]

    def get_observations(self, state: State) -> dict[int, Observation]:
        """Each player's view. The operator's entry is degraded by the
        measured constants in ``blotto.game.priors``; the platform and the
        field see their own side in full. The hidden state never appears."""
        day = self._day(state)
        operator_obs = (
            state["log"][-1]
            if state["log"]
            else Observation(utm_content="", posted_at=_iso(day), observed_at=_iso(day))
        )
        true_obs = (
            state["true_log"][-1]
            if state["true_log"]
            else Observation(utm_content="", posted_at=_iso(day), observed_at=_iso(day))
        )
        return {OPERATOR: operator_obs, PLATFORM: true_obs, FIELD: true_obs}

    def get_rewards(self, state: State) -> dict[int, float]:
        """Contribution margin from settled observations, minus committed
        spend.

        Reward accrues only on SETTLED observations: a number inside the
        reporting lag is provisional, and provisional margin is not margin.
        Reach appears nowhere in this sum (``blotto.game.payoff`` says why,
        and asks future readers not to fix the omission)."""
        economics = self.config.economics
        margin = sum(
            reward(obs, economics, economics.fixed_cost_per_post)
            for obs in state["log"]
            if not obs.is_partial
        )
        operator = margin - state["spend_committed"]
        return {OPERATOR: operator, PLATFORM: 0.0, FIELD: 0.0}

    def chance_outcomes(self, state: State) -> list[tuple[ActionKey, float]]:
        if state["phase"] != "chance":
            return []
        outcomes: list[tuple[ActionKey, float]] = []
        for bucket, p_response in _RESPONSE_BUCKETS:
            for direction, _sign in _DRIFT_DIRECTIONS:
                outcomes.append(
                    (
                        ActionKey(f"chance|response={bucket}|drift={direction}"),
                        p_response * _DRIFT_PROBABILITIES[direction],
                    )
                )
        return outcomes

    # -- Operator moves ----------------------------------------------------------

    def _apply_operator(self, state: State, action: ActionKey) -> None:
        move = self.codec.decode(action)
        day = self._day(state)
        if isinstance(move, Scale):
            state["angle_first_day"].setdefault(move.angle, day)
            before = state["allocation"].get(move.angle, 0.0)
            after = before * move.factor
            state["allocation"][move.angle] = after
            state["allocation_history"].append(
                AllocationChange(
                    at=datetime.combine(_EPOCH + timedelta(days=day), time()),
                    angle=move.angle,
                    allocation=after,
                )
            )
            committed = max(0.0, after - before) * _SPEND_PER_ALLOCATION_UNIT
            state["spend_committed"] += committed
            state["angle_spend"][move.angle] = (
                state["angle_spend"].get(move.angle, 0.0) + committed
            )
        elif isinstance(move, Kill):
            state["allocation"][move.angle] = 0.0
        elif isinstance(move, Publish):
            self._apply_publish(state, move, day)
        # Hold: a slot spent on nothing, which costs nothing.
        state["phase"] = "chance"

    def _apply_publish(self, state: State, move: Publish, day: int) -> None:
        hidden = self._hidden(state)
        theta = hidden["theta"]
        archetype = ARCHETYPE_PRIORS[move.archetype]
        platform = PLATFORM_PRIORS[move.platform]

        hook = archetype.hook_rate.mid
        hold = archetype.hold_rate.mid
        engagement = platform.engagement_rate.mid
        saves_exp, shares_exp, comments_exp = (
            engagement * share for share in _ENGAGEMENT_SPLIT
        )
        # Ranking score per the game spec: expectations pulled from the
        # priors, dotted with the hidden theta, scaled by account standing,
        # decayed by angle congestion. dwell and follow_through weights exist
        # in theta and drift with it, but the spec's score is five terms, and
        # inventing two more is redesign rather than implementation.
        score = (
            theta["hook_rate"] * hook
            + theta["hold_rate"] * hold
            + theta["saves"] * saves_exp
            + theta["shares"] * shares_exp
            + theta["comments"] * comments_exp
        )

        fatigue_key = f"{move.archetype.value}|{move.angle}"
        uses = hidden["asset_fatigue"].get(fatigue_key, 0)
        # A proven hook keeps working for WINNING_HOOK_BURNOUT_DAYS and then
        # the audience pattern-matches it to an ad; repeated use of the same
        # (archetype, angle) pair rides down that same curve.
        burnout = WINNING_HOOK_BURNOUT_DAYS.mid
        fatigue_factor = 1.0 / (1.0 + uses / burnout)

        belief = hidden["belief"].setdefault(
            move.avatar, {"exposures": 0, "credence": 0.5, "fatigue": 0.0}
        )
        belief_reach_factor = 1.0 / (1.0 + belief["fatigue"])

        base_audience = (
            self.config.field_size
            * 800.0
            * platform.organic_reach_share.mid
            * hidden["standing"]
        )
        reach = (
            base_audience
            * score
            * (1.0 - hidden["saturation"].get(move.angle, 0.0))
            * fatigue_factor
            * belief_reach_factor
        )
        if move.cta_mode in _OFF_PLATFORM_CTAS:
            reach *= 1.0 - theta["external_link_penalty"]
        reach = max(0.0, reach)

        # Audience belief: each exposure increments, conversion probability
        # falls with exposure count, credence moves with separating power.
        belief["exposures"] += 1
        belief["credence"] = min(
            1.0, max(0.0, belief["credence"] + _CREDENCE_DELTA[move.claim_class])
        )
        belief["fatigue"] = min(1.0, belief["fatigue"] + 0.10)
        hidden["asset_fatigue"][fatigue_key] = uses + 1

        conversion_factor = (1.0 / (1.0 + belief["exposures"] * 0.08)) * (
            0.5 + belief["credence"]
        )
        conversions = reach * archetype.expected_cvr.mid * conversion_factor
        profile_visits = reach * archetype.outbound_ctr.mid
        link_clicks = profile_visits * platform.landing_page_cvr.mid

        cost = self.config.economics.fixed_cost_per_post
        state["angle_spend"][move.angle] = (
            state["angle_spend"].get(move.angle, 0.0) + cost
        )
        state["angle_first_day"].setdefault(move.angle, day)
        state["utm_angle"][move.utm_content] = move.angle
        state["published_today"][move.format.value] = (
            state["published_today"].get(move.format.value, 0) + 1
        )
        if move.cta_mode in ACQUISITION_CTAS:
            state["acq_today"] += 1
        else:
            state["svc_today"] += 1

        # Reporting lag in days, drawn deterministically from the utm: the
        # same post settles at the same time in every replay.
        lag_low = int(REPORTING_LAG_HOURS.low // 24)
        lag_high = max(lag_low, int(REPORTING_LAG_HOURS.high // 24))
        span = lag_high - lag_low + 1
        lag = lag_low + int(_stable_uniform(move.utm_content, "lag") * span)
        settles_day = day + lag

        # Per-post jitter, likewise deterministic from the utm, so two posts
        # of the same shape on the same day still report different numbers.
        jitter = 0.9 + 0.2 * _stable_uniform(move.utm_content, "metric_jitter")
        saturation = hidden["saturation"].get(move.angle, 0.0)
        state["pending"].append(
            {
                "utm": move.utm_content,
                "angle": move.angle,
                "platform": move.platform.value,
                "claim": move.claim_class.value,
                "posted_day": day,
                "settles_day": settles_day,
                "resolved": False,
                "base": {
                    "reach": reach * jitter,
                    "hook_rate": hook
                    * jitter
                    * (1.0 - 0.5 * saturation),
                    "hold_rate": hold * jitter,
                    "saves": reach * saves_exp * jitter,
                    "shares": reach * shares_exp * jitter,
                    "comments": reach * comments_exp * jitter,
                    "profile_visits": profile_visits * jitter,
                    "link_clicks": link_clicks * jitter,
                    "conversions": conversions * jitter,
                },
            }
        )

    # -- Chance resolution ---------------------------------------------------------

    def _apply_chance(self, state: State, action: ActionKey) -> None:
        try:
            bucket = action.split("|")[1].split("=")[1]
            direction = action.split("|")[2].split("=")[1]
        except IndexError as exc:
            raise ValueError(f"not a chance outcome key: {action!r}") from exc
        if bucket not in _RESPONSE_MULTIPLIER or direction not in dict(_DRIFT_DIRECTIONS):
            raise ValueError(f"not a chance outcome key: {action!r}")
        hidden = self._hidden(state)
        day = hidden["step"]

        for entry in state["pending"]:
            if entry["resolved"] or entry["posted_day"] != day:
                continue
            self._resolve_pending(state, entry, _RESPONSE_MULTIPLIER[bucket])
        self._drift_theta(hidden, direction)
        self._advance_day(state, hidden, bucket, day)

    def _resolve_pending(
        self, state: State, entry: dict[str, Any], multiplier: float
    ) -> None:
        base = entry["base"]
        day = entry["posted_day"]
        reach = base["reach"] * multiplier
        true_conversions_count = base["conversions"] * multiplier
        utm = entry["utm"]

        # Degradation draws, deterministic per utm, from the measured bands:
        # coverage = 1 - client-side loss x (1 - server-side recovery);
        # incrementality = 1 - non-incremental share.
        loss = CLIENT_SIDE_EVENT_LOSS.low + _stable_uniform(utm, "loss") * (
            CLIENT_SIDE_EVENT_LOSS.high - CLIENT_SIDE_EVENT_LOSS.low
        )
        recovery = SERVER_SIDE_RECOVERY.low + _stable_uniform(utm, "recovery") * (
            SERVER_SIDE_RECOVERY.high - SERVER_SIDE_RECOVERY.low
        )
        coverage = max(0.0, 1.0 - loss * (1.0 - recovery))
        non_incremental = NON_INCREMENTAL_SHARE.low + _stable_uniform(
            utm, "non_incremental"
        ) * (NON_INCREMENTAL_SHARE.high - NON_INCREMENTAL_SHARE.low)
        dark_social = DARK_SOCIAL_B2B_SHARE.low + _stable_uniform(utm, "dark") * (
            DARK_SOCIAL_B2B_SHARE.high - DARK_SOCIAL_B2B_SHARE.low
        )

        settles = entry["settles_day"]
        attributed = true_conversions_count * coverage

        common = dict(
            utm_content=utm,
            posted_at=_iso(day),
            observed_at=_iso(day),
            impressions=int(reach * 1.5),
            reach=int(reach),
            hook_rate=min(1.0, base["hook_rate"] * multiplier),
            hold_rate=min(1.0, base["hold_rate"] * multiplier),
            saves=int(base["saves"] * multiplier),
            shares=int(base["shares"] * multiplier),
            comments=int(base["comments"] * multiplier),
            profile_visits=int(base["profile_visits"] * multiplier),
            link_clicks=int(base["link_clicks"] * multiplier),
            dark_social_estimate=round(dark_social, 4),
        )
        # The operator's degraded view: partial while inside the lag, seeing
        # only the covered share of conversions.
        state["log"].append(
            Observation(
                **common,
                attributed_conversions=int(round(attributed)),
                attribution_coverage=round(coverage, 4),
                incrementality=round(1.0 - non_incremental, 4),
                is_partial=settles > self._day(state),
            )
        )
        # The platform-side truth: full coverage, true conversions. Only the
        # platform and field players ever receive this entry.
        state["true_log"].append(
            Observation(
                **common,
                attributed_conversions=int(round(true_conversions_count)),
                attribution_coverage=1.0,
                incrementality=1.0,
                is_partial=False,
            )
        )

        # Compliance strike: a high-risk claim that lands badly draws platform
        # review. Standing falls and recovers only slowly, so one strike taxes
        # the rest of the episode's reach.
        if entry["claim"] in {c.value for c in _STRIKE_RISK_CLAIMS} and multiplier < 1.0:
            hidden = self._hidden(state)
            hidden["standing"] = max(0.0, hidden["standing"] - _STRIKE_PENALTY)
            strikes = state["strikes"]
            strikes[entry["platform"]] = strikes.get(entry["platform"], 0) + 1
        entry["resolved"] = True

    def _drift_theta(self, hidden: dict[str, Any], direction: str) -> None:
        """The platform's bounded random walk, at a chance node only."""
        step = dict(_DRIFT_DIRECTIONS)[direction]
        rate = self.config.drift_rate
        theta = hidden["theta"]
        for key in _THETA_WEIGHT_KEYS:
            theta[key] = min(3.0, max(0.2, theta[key] + step * rate))
        theta["external_link_penalty"] = min(
            1.0, max(0.0, theta["external_link_penalty"] + step * rate * 0.5)
        )

    def _advance_day(
        self, state: State, hidden: dict[str, Any], bucket: str, day: int
    ) -> None:
        new_day = day + 1
        hidden["step"] = new_day
        hidden["standing"] = min(1.0, hidden["standing"] + _STANDING_RECOVERY_PER_DAY)
        # The field's congestion strategy, as mean reversion: attention flows
        # to angles that are paying and away from ones that are not.
        target = 0.30 + _SATURATION_ADJUST[bucket]
        for angle in hidden["saturation"]:
            current = hidden["saturation"][angle]
            hidden["saturation"][angle] = min(
                1.0, max(0.0, current + (target - current) * 0.2)
            )
        for belief in hidden["belief"].values():
            belief["fatigue"] = max(0.0, belief["fatigue"] - 0.02)

        # Settle: posts whose reporting lag has cleared flip to final. The
        # pending entry is dropped at the same moment, so "in the log and not
        # partial" and "no longer pending" can never disagree.
        settled_utms = {
            entry["utm"] for entry in state["pending"] if entry["settles_day"] <= new_day
        }
        if settled_utms:
            state["log"] = [
                replace(obs, is_partial=False, observed_at=_iso(new_day))
                if obs.utm_content in settled_utms and obs.is_partial
                else obs
                for obs in state["log"]
            ]
        state["pending"] = [
            entry for entry in state["pending"] if entry["settles_day"] > new_day
        ]
        state["published_today"] = {}
        state["acq_today"] = 0
        state["svc_today"] = 0
        state["phase"] = "terminal" if new_day >= self.config.horizon else "operator"

    # -- Legality plumbing -----------------------------------------------------------

    def _candidate_publishes(self, state: State) -> list[Publish]:
        return list(
            enumerate_publishes(
                platforms=(Platform.TIKTOK, Platform.X, Platform.LINKEDIN),
                formats=(Format.CAROUSEL, Format.TEXT_THREAD),
                archetypes=tuple(Archetype),
                vectors=(
                    EmotionalVector.ANGER_INJUSTICE,
                    EmotionalVector.EXHAUSTION_RELIEF,
                    EmotionalVector.ASPIRATION_STATUS,
                ),
                tiers=(SemanticTier.TRIBAL_IDENTITY, SemanticTier.HYPER_SPECIFIC_ENEMY),
                hooks=(HookFamily.PATTERN_INTERRUPT, HookFamily.CURIOSITY_GAP),
                avatars=self.config.avatars,
                cta_modes=(CtaMode.START_TRIAL, CtaMode.REPLY_KEYWORD),
                claim_classes=(
                    ClaimClass.NO_CLAIM,
                    ClaimClass.MECHANISM,
                    ClaimClass.FIRST_PARTY_PROOF,
                ),
                angles=self.config.angles,
                limit=24,
                seed=self.config.seed,
            )
        )

    def _legality_context(self, state: State) -> LegalityContext:
        day = self._day(state)
        current = _EPOCH + timedelta(days=day)
        settled = [obs for obs in state["log"] if not obs.is_partial]

        ctx = LegalityContext(current_date=current)
        for window_days, target in (
            (7, ctx.settled_conversions_7d),
            (14, ctx.settled_conversions_14d),
        ):
            cutoff = current - timedelta(days=window_days)
            for obs in settled:
                if datetime.fromisoformat(obs.posted_at).date() >= cutoff:
                    angle = state["utm_angle"].get(obs.utm_content, "")
                    if angle:
                        target[angle] = (
                            target.get(angle, 0) + obs.attributed_conversions
                        )
        coverages: dict[str, list[float]] = {}
        for obs in settled:
            angle = state["utm_angle"].get(obs.utm_content, "")
            if not angle:
                continue
            ctx.angle_reach[angle] = ctx.angle_reach.get(angle, 0) + obs.reach
            ctx.angle_total_conversions[angle] = (
                ctx.angle_total_conversions.get(angle, 0) + obs.attributed_conversions
            )
            coverages.setdefault(angle, []).append(obs.attribution_coverage)
        # Per-angle coverage collapses to its mean; unknown angles keep the
        # conservative default of 0.0 -- unjudgeable, not clean.
        ctx.angle_attribution_coverage = {
            angle: sum(values) / len(values) for angle, values in coverages.items()
        }
        ctx.angle_spend = dict(state["angle_spend"])
        ctx.allocation = dict(state["allocation"])
        ctx.allocation_history = list(state["allocation_history"])
        ctx.angle_age_days = {
            angle: day - first for angle, first in state["angle_first_day"].items()
        }
        # Substantiation is modelled as settled evidence at the creative
        # judgement threshold: an angle with that many settled conversions
        # has receipts on file.
        threshold = self.engine.policy.creative_judgement_min_conversions
        ctx.substantiated_claims = {
            angle
            for angle, count in ctx.settled_conversions_7d.items()
            if count >= threshold
        }
        ctx.published_today_by_format = {
            Format(name): count for name, count in state["published_today"].items()
        }
        ctx.acquisition_posts_today = state["acq_today"]
        ctx.service_posts_today = state["svc_today"]
        ctx.account_standing = self._hidden(state)["standing"]
        ctx.platform_strikes = {
            Platform(name): count for name, count in state["strikes"].items()
        }
        return ctx

    # -- Trajectories -----------------------------------------------------------------

    def generate_trajectory(
        self,
        policy: Callable[[CodeWorldModel, State], ActionKey],
        steps: int,
        rng: random.Random,
    ) -> Trajectory:
        """Play one episode under ``policy`` and record what the operator saw.

        Observations attach to their moves after the episode ends, because a
        publish's numbers do not exist until the reporting lag clears -- a
        trajectory is recorded history, and history is written backwards.
        """
        state = self.initial_state()
        moves: list[Move] = []
        chance: list[ActionKey] = []
        while self.get_current_player(state) != TERMINAL_PLAYER and len(moves) < steps:
            player = self.get_current_player(state)
            if player == CHANCE_PLAYER:
                outcomes = self.chance_outcomes(state)
                action = rng.choices(
                    [key for key, _ in outcomes],
                    weights=[prob for _, prob in outcomes],
                    k=1,
                )[0]
                # Recorded, not discarded. A transition is deterministic given
                # the chance action, so a test that re-draws is measuring the
                # dice rather than the model. See Trajectory.chance.
                chance.append(action)
                state = self.apply_action(state, action)
                continue
            legal = self.get_legal_actions(state)
            if not legal:
                break
            action = policy(self, state)
            if action not in legal:
                raise ValueError(
                    f"policy returned illegal action {action!r}; legality is a "
                    "precondition on planning, not a penalty"
                )
            move = self.codec.decode(action)
            if isinstance(move, Publish):
                # Re-stamp with a per-step unique id. The legal-action set
                # offers a fixed catalogue, so a policy that picks the same
                # entry twice would publish two distinct posts under one utm --
                # and utm is the join key between a move and its observation.
                # The two posts then collapse to one row, the later metrics
                # overwrite the earlier, and a transition test compares a
                # model's prediction for post A against the numbers post B
                # eventually produced. That looked like a 3x modelling error
                # and was a bookkeeping collision.
                move = replace(move, utm_content=utm_content_id(move, salt=str(len(moves))))
                action = self.codec.encode(move)
            moves.append(move)
            state = self.apply_action(state, action)

        by_utm = {obs.utm_content: obs for obs in state["log"]}
        trajectory = Trajectory(account="reference", chance=chance)
        for move in moves:
            obs: Observation | None = None
            if isinstance(move, Publish):
                obs = by_utm.get(move.utm_content)
            trajectory.steps.append(Step(move=move, observation=obs))
        return trajectory
