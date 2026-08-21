"""Core types for the distribution game.

This module is the fixed contract every other package builds against. It is
deliberately small. Nothing here does work; everything here defines what work
is allowed to look like.

Design note, and it matters
---------------------------
The world model's *internal* state is ``dict[str, Any]`` -- not a typed
dataclass. This is not laziness. In Lehrach et al. (2025), the LLM synthesises
its own latent state representation, and the game rules plus the required API
act as the only regulariser:

    "Instead of a bottleneck, or a regularization term, the game rules and the
    required OpenSpiel API (used in the unit tests) introduced in the context
    of the LLM act as regularizers to prevent trivial latent spaces from being
    discovered."  -- Code World Models for General Game Playing, section 4.4

If we impose our own schema on the synthesised state, we destroy exactly the
degree of freedom that makes closed-deck synthesis work. The paper's own
Hand of war result -- where the closed-deck agent *beat* the open-deck agent --
is attributed to "the freedom to synthesize simpler state spaces."

So: ``HiddenState`` below describes what we believe is really going on, and is
used by the reference model and by evaluation. The synthesised CWM is free to
disagree, and often should.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NewType

__all__ = [
    "CHANCE_PLAYER",
    "TERMINAL_PLAYER",
    "OPERATOR",
    "PLATFORM",
    "FIELD",
    "State",
    "ActionKey",
    "Platform",
    "Format",
    "Archetype",
    "EmotionalVector",
    "SemanticTier",
    "HookFamily",
    "CtaMode",
    "ClaimClass",
    "MoveKind",
    "Publish",
    "Scale",
    "Kill",
    "Hold",
    "Move",
    "RankingWeights",
    "AudienceBelief",
    "HiddenState",
    "Observation",
    "Trajectory",
    "Step",
    "EvidenceGate",
]


# ---------------------------------------------------------------------------
# Player indices. Follows the OpenSpiel convention used by the paper: chance is
# -1 and terminal is -4, so a synthesised model written against OpenSpiel
# examples will land on the right integers without being told.
# ---------------------------------------------------------------------------

CHANCE_PLAYER = -1
TERMINAL_PLAYER = -4

OPERATOR = 0
"""Us. The only player whose policy we control."""

PLATFORM = 1
"""The ranking algorithm. Not adversarial -- *misaligned*. It maximises session
time and retention; we maximise paying users. It moves first and commits, which
makes it a Stackelberg leader and us the follower."""

FIELD = 2
"""The aggregate of every other creator competing for the same attention. Plays
a congestion strategy over the angle space: the more of them on an angle, the
less that angle pays."""


State = NewType("State", dict[str, Any])
"""Opaque world-model state. Schema is the synthesised model's business."""

ActionKey = NewType("ActionKey", str)
"""Stable string encoding of a Move.

The paper defines ``Action = str`` and we keep that at the CWM boundary: the
synthesised Python manipulates strings, never our dataclasses. ``ActionCodec``
in ``blotto.game.action_space`` is the only thing allowed to cross that line.
"""


# ---------------------------------------------------------------------------
# Action space dimensions.
#
# These enums are not invented. Each carries an empirical prior drawn from the
# DTC Engineer / GTM Engineer corpora, wired up in ``blotto.game.priors``.
# Where a member has a benchmark attached, the docstring cites its source file
# so that no number in this repository is unattributable.
# ---------------------------------------------------------------------------


class Platform(str, Enum):
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    THREADS = "threads"
    X = "x"
    LINKEDIN = "linkedin"
    YOUTUBE_SHORTS = "youtube_shorts"
    REDDIT = "reddit"


class Format(str, Enum):
    CAROUSEL = "carousel"
    SHORT_VIDEO = "short_video"
    TEXT_THREAD = "text_thread"
    SINGLE_IMAGE = "single_image"
    LONG_FORM = "long_form"


class Archetype(str, Enum):
    """Narrative archetypes. The benchmark-carrying taxonomy.

    Source: Storytelling Engineer / 24_Benchmark_Tables_CTR_and_CVR_by_
    Narrative_Archetype.md -- each member has measured hook rate, hold rate,
    outbound CTR and expected CVR bands.
    """

    UGC_REVIEW = "ugc_review"
    FOUNDER_TRAUMA = "founder_trauma"
    ANTI_HERO_RANT = "anti_hero_rant"
    AUTONOMOUS_VOXEL = "autonomous_voxel"


class EmotionalVector(str, Enum):
    """Source: Storytelling Engineer / 01_The_Semantic_Data_Layer_of_Persuasion.md

    Carries CAC / churn / LTV priors. The corpus's central finding is that the
    cheapest-CAC vector is a false positive: Aspiration acquires at $30 CAC with
    45% M3 churn and $250 LTV, while Exhaustion acquires at $65 CAC with 8% M3
    churn and $1,200 LTV. A reward function that optimises CAC picks the wrong
    one. See ``blotto.game.payoff``.
    """

    ANGER_INJUSTICE = "anger_injustice"
    ASPIRATION_STATUS = "aspiration_status"
    FEAR_LOSS = "fear_loss"
    EXHAUSTION_RELIEF = "exhaustion_relief"
    ANALYTICAL_PROOF = "analytical_proof"


class SemanticTier(str, Enum):
    """Source: Storytelling Engineer / 01_The_Semantic_Data_Layer_of_Persuasion.md

    Generic slop costs $3.50+ CPC at 0.8% CVR; tribal identity costs $0.45 CPC
    at 6.8% CVR. This is the single widest measured spread in either corpus.
    """

    GENERIC_SLOP = "generic_slop"
    FEATURE_HEAVY = "feature_heavy"
    HYPER_SPECIFIC_ENEMY = "hyper_specific_enemy"
    TRIBAL_IDENTITY = "tribal_identity"


class HookFamily(str, Enum):
    """Hook mechanics. Union of the DTC hook frameworks and the mechanics named
    as non-negotiable in the operator's own CLAUDE.md."""

    PRONOUN_INVERSION = "pronoun_inversion"
    NEGATIVE_PRESCRIPTIVISM = "negative_prescriptivism"
    TACTICAL_SINGULARITY = "tactical_singularity"
    OBJECTION_MIRRORING = "objection_mirroring"
    RADICAL_TRANSPARENCY = "radical_transparency"
    PATTERN_INTERRUPT = "pattern_interrupt"
    CURIOSITY_GAP = "curiosity_gap"
    IDENTITY_CONFIRMATION = "identity_confirmation"
    PROBLEM_SOLUTION_LEAD = "problem_solution_lead"


class CtaMode(str, Enum):
    START_TRIAL = "start_trial"
    BOOK_DEMO = "book_demo"
    INSTALL_APP = "install_app"
    JOIN_WAITLIST = "join_waitlist"
    REPLY_KEYWORD = "reply_keyword"
    FOLLOW_FOR_SERIES = "follow_for_series"


class ClaimClass(str, Enum):
    """What kind of assertion the content makes.

    Drives two separate systems. Compliance: ``get_legal_actions`` refuses
    claims the operator cannot substantiate. Signalling: a claim's separating
    power in ``blotto.solvers.signalling`` depends on how expensive it is for a
    low-quality sender to imitate.
    """

    NO_CLAIM = "no_claim"
    MECHANISM = "mechanism"
    """Names how it works, never what it cures. The compliance-safe form."""
    VERIFIABLE_METRIC = "verifiable_metric"
    """A number the operator can produce receipts for."""
    FIRST_PARTY_PROOF = "first_party_proof"
    """Own-data demonstration. Expensive to fake -- high separating power."""
    THIRD_PARTY_TESTIMONIAL = "third_party_testimonial"
    COMPARATIVE = "comparative"
    """Names a competitor. High CTR, high platform-rejection risk."""
    OUTCOME_PROMISE = "outcome_promise"
    """Requires substantiation. Illegal without it."""


# ---------------------------------------------------------------------------
# Moves.
#
# The operator has two genuinely different kinds of decision, and collapsing
# them is the modelling error this whole repository exists to fix. Publishing is
# cheap and reversible. Scaling commits real budget behind a belief, and the
# corpora are unanimous that scaling on thin evidence is how accounts die.
#
# Making Scale a first-class move is what lets ``get_legal_actions`` refuse it
# until the evidence gate opens. The planner then cannot choose it -- not
# "is penalised for choosing it", cannot choose it. That is the paper's
# verifiability claim applied to money.
# ---------------------------------------------------------------------------


class MoveKind(str, Enum):
    PUBLISH = "publish"
    SCALE = "scale"
    KILL = "kill"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class Publish:
    """Ship one content item into one slot."""

    platform: Platform
    format: Format
    archetype: Archetype
    vector: EmotionalVector
    semantic_tier: SemanticTier
    hook: HookFamily
    avatar: str
    cta_mode: CtaMode
    claim_class: ClaimClass
    angle: str
    """The contested resource. Congestion is computed per angle, not per post."""
    utm_content: str
    """Tracking id. This is the join key between a move and its observation, and
    the reason an untracked post is not a legal move."""

    kind: MoveKind = MoveKind.PUBLISH


@dataclass(frozen=True, slots=True)
class Scale:
    """Commit more budget or volume behind an angle already in flight."""

    angle: str
    factor: float
    """Multiplier on current allocation. Velocity limits are enforced in
    ``blotto.game.legality``, not here."""

    kind: MoveKind = MoveKind.SCALE


@dataclass(frozen=True, slots=True)
class Kill:
    """Retire an angle."""

    angle: str
    reason: str
    kind: MoveKind = MoveKind.KILL


@dataclass(frozen=True, slots=True)
class Hold:
    """Spend a slot on nothing.

    Legal and sometimes correct. An operator with no unsaturated angle and no
    substantiated claim is better off silent than shipping generic slop at
    $3.50 CPC. The planner must be able to represent that.
    """

    kind: MoveKind = MoveKind.HOLD


Move = Publish | Scale | Kill | Hold


# ---------------------------------------------------------------------------
# Hidden state. Never observed by the operator. This is the closed-deck setting
# of the paper: "the agent can only ever access its own observations and
# actions."
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RankingWeights:
    """The platform's current objective, which we never see and which drifts.

    Recovering a posterior over this is the job of ``resample_state`` in
    ``blotto.cwm.inference``. Best-responding to the estimate is the job of
    ``blotto.solvers.stackelberg``.
    """

    hook_rate: float = 1.0
    hold_rate: float = 1.0
    saves: float = 1.0
    shares: float = 1.0
    comments: float = 1.0
    dwell: float = 1.0
    follow_through: float = 1.0
    external_link_penalty: float = 0.0
    drift_rate: float = 0.0
    """Per-step magnitude of random walk. Non-zero drift is why angle selection
    uses EXP3 rather than a stochastic bandit -- see ``blotto.solvers.exp3``."""


@dataclass(slots=True)
class AudienceBelief:
    """Per-avatar state. Attention is not a renewable resource within a cohort."""

    exposures: int = 0
    credence: float = 0.5
    """How much this segment believes the operator's central claim, [0, 1]."""
    fatigue: float = 0.0


@dataclass(slots=True)
class HiddenState:
    """Ground truth as the reference model understands it.

    Used by the hand-written reference model, by trajectory generation, and by
    evaluation. A synthesised CWM is under no obligation to use this shape.
    """

    theta: RankingWeights = field(default_factory=RankingWeights)
    standing: float = 0.5
    """Account quality score in [0, 1]. Degrades on compliance strikes."""
    saturation: dict[str, float] = field(default_factory=dict)
    """Angle -> share of the field currently running it, [0, 1]."""
    belief: dict[str, AudienceBelief] = field(default_factory=dict)
    asset_fatigue: dict[str, float] = field(default_factory=dict)
    step: int = 0


# ---------------------------------------------------------------------------
# Observations. What the operator actually gets: late, lossy, and partly
# fictional. Every degradation below is a measured figure from the corpora, not
# a modelling convenience.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Observation:
    utm_content: str
    posted_at: str
    observed_at: str

    impressions: int = 0
    reach: int = 0
    hook_rate: float = 0.0
    """Watch past 2s. Below 25% the algorithm abandons the asset
    (DTC 13_The_Hook_Library_and_Visual_Pattern_Interrupts.md)."""
    hold_rate: float = 0.0
    saves: int = 0
    shares: int = 0
    comments: int = 0
    profile_visits: int = 0
    link_clicks: int = 0

    attributed_conversions: int = 0
    attribution_coverage: float = 1.0
    """Fraction of true conversions this observation can see. Ad blockers, ITP
    and Privacy Sandbox block 25-40% of events client-side; server-side recovery
    returns 20-30% (GTM 09_Attribution_and_Analytics.md). Coverage below the
    operator's floor makes an angle unjudgeable, not merely noisy."""
    incrementality: float = 1.0
    """Fraction of attributed conversions that would not have happened anyway.
    Holdout tests put this at 0.6-0.8 (GTM 09). Reward uses the product of
    coverage and incrementality, never the raw dashboard number."""

    is_partial: bool = False
    """True while inside the 24-72h reporting lag (DTC 21_Media_Buying_Risk_
    Management_and_Drawdowns.md). A partial observation may be read but must
    not open an evidence gate."""

    dark_social_estimate: float = 0.0
    """Unattributable sharing. In B2B this is 60-80% of all content sharing
    (GTM 03_Channel_Playbooks_2026.md) and surfaces as direct traffic."""


@dataclass(frozen=True, slots=True)
class Step:
    """One (move, resulting observation) pair. The unit of a trajectory."""

    move: Move
    observation: Observation | None
    """None while the move is in flight and no data has landed yet."""


@dataclass(slots=True)
class Trajectory:
    """A sequence of the operator's own moves and observations.

    This is the *only* training signal available in the closed-deck setting, and
    it is what unit tests are generated from in ``blotto.cwm.tests_from_traj``.
    """

    steps: list[Step] = field(default_factory=list)
    account: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Evidence gates.
#
# The single most consequential type in this file. A gate is a precondition on
# acting, not a penalty for acting. Both corpora independently arrive at the
# same conclusion -- that the dominant failure mode in distribution is not bad
# creative but premature scaling -- and both state it numerically:
#
#   "Never declare a winning emotional arc under 50 conversion events in a
#    7-day rolling window"; calling it at 5 conversions sees CPA "frequently
#    explode to $150" on scale-up.
#       -- DTC / Storytelling Engineer 17_A_B_Testing_Story_Arcs...md
#
#   "95% confidence, 80% power, 300+ conversions per variant, minimum 14 days,
#    one variable per test... no peeking before minimum sample size."
#       -- GTM 04_The_100x_Engineer_Mindset.md, 08_Landing_Pages_and_CRO.md
#
# The two thresholds differ because the decisions differ: 50/7d licenses a
# creative-level judgement, 300/14d licenses a spend commitment. Both are
# expressed here and both are enforced.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceGate:
    """A precondition that must be satisfied before a class of move is legal."""

    name: str
    min_conversions: int
    window_days: int
    min_reach: int = 0
    min_attribution_coverage: float = 0.0
    requires_settled: bool = True
    """If True, observations still inside the reporting lag do not count toward
    the threshold. Peeking at partial data is the failure this prevents."""
    source: str = ""
    """Citation. A gate without a source is not permitted to exist."""
