"""Every empirical number the system is allowed to use.

HARD REQUIREMENT, and the reason this module exists in this shape: a prior
without a source does not exist. Each number below is a measured figure from a
specific corpus file, and the dataclass carrying it names that file. If a value
cannot be attributed, it is not entered here -- it is either omitted or set to
``None`` with a comment saying the corpus does not measure it. An unattributed
benchmark in a planning system is worse than a missing one: it optimises
confidently against a number nobody can defend.

Two conventions make the source rule enforceable:

* Every range is a ``Band``, and every ``Band`` carries its own ``source``.
* A point estimate is encoded as a degenerate band (``low == high``) rather
  than a bare float, so that it too is forced to carry a citation. The
  semantic-tier CPC column already works this way in the corpus -- only
  generic slop is reported as a range -- so the convention is inherited, not
  invented.

Nothing in this module computes anything. It is a table, checked at import
time by ``scripts/selfcheck.py`` and at test time by ``tests/test_game.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from blotto.game.types import (
    Archetype,
    EmotionalVector,
    Platform,
    SemanticTier,
)

__all__ = [
    "Band",
    "ArchetypePrior",
    "SemanticPrior",
    "VectorPrior",
    "PlatformPrior",
    "ARCHETYPE_PRIORS",
    "SEMANTIC_TIER_PRIORS",
    "VECTOR_PRIORS",
    "PLATFORM_PRIORS",
    "CLIENT_SIDE_EVENT_LOSS",
    "SERVER_SIDE_RECOVERY",
    "NON_INCREMENTAL_SHARE",
    "DARK_SOCIAL_B2B_SHARE",
    "REPORTING_LAG_HOURS",
    "HOOK_RATE_ABANDON_THRESHOLD",
    "HOLD_RATE_DECAY_TRIGGER",
    "CTR_FATIGUE_TRIGGER",
    "WINNING_HOOK_BURNOUT_DAYS",
]


@dataclass(frozen=True, slots=True)
class Band:
    """A measured range. ``mid`` exists because planners need a point estimate
    to seed a prior with, and taking the mean of the measured band is the least
    committed choice available."""

    low: float
    high: float
    source: str

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2.0


@dataclass(frozen=True, slots=True)
class ArchetypePrior:
    """Per-archetype attention and conversion benchmarks.

    Source: Storytelling Engineer/
    24_Benchmark_Tables_CTR_and_CVR_by_Narrative_Archetype.md
    """

    hook_rate: Band
    hold_rate: Band
    outbound_ctr: Band
    expected_cvr: Band
    source: str = (
        "Storytelling Engineer/"
        "24_Benchmark_Tables_CTR_and_CVR_by_Narrative_Archetype.md"
    )


@dataclass(frozen=True, slots=True)
class SemanticPrior:
    """Per-tier paid economics.

    Source: Storytelling Engineer/01_The_Semantic_Data_Layer_of_Persuasion.md
    """

    cpc_usd: Band
    cvr: float
    source: str = (
        "Storytelling Engineer/01_The_Semantic_Data_Layer_of_Persuasion.md"
    )


@dataclass(frozen=True, slots=True)
class VectorPrior:
    """Per-emotional-vector customer economics.

    Source: Storytelling Engineer/01_The_Semantic_Data_Layer_of_Persuasion.md
    """

    cac_usd: Band
    m3_churn: float
    ltv_usd: Band
    source: str = (
        "Storytelling Engineer/01_The_Semantic_Data_Layer_of_Persuasion.md"
    )


@dataclass(frozen=True, slots=True)
class PlatformPrior:
    """Per-platform operating constraints.

    Engagement and reach: GTM Engineer/Encyclopedia/
    02_Metrics_and_Baselines_2026.md. Creative lifespan and weekly volume: GTM
    Engineer/Encyclopedia/11_Paid_Channels_Deep_Dive.md -- the volume floor is
    what makes "post consistently" a scheduling constraint rather than advice.
    Landing-page CVR: GTM Engineer/Encyclopedia/08_Landing_Pages_and_CRO.md.
    """

    engagement_rate: Band
    organic_reach_share: Band
    creative_lifespan_days: Band
    min_new_creatives_per_week: Band
    landing_page_cvr: Band
    source: str = "GTM Engineer/Encyclopedia/02_Metrics_and_Baselines_2026.md"


_ARCHETYPE_SOURCE = (
    "Storytelling Engineer/"
    "24_Benchmark_Tables_CTR_and_CVR_by_Narrative_Archetype.md"
)
_SEMANTIC_SOURCE = (
    "Storytelling Engineer/01_The_Semantic_Data_Layer_of_Persuasion.md"
)


ARCHETYPE_PRIORS: dict[Archetype, ArchetypePrior] = {
    Archetype.UGC_REVIEW: ArchetypePrior(
        hook_rate=Band(0.22, 0.28, _ARCHETYPE_SOURCE),
        hold_rate=Band(0.18, 0.22, _ARCHETYPE_SOURCE),
        outbound_ctr=Band(0.008, 0.012, _ARCHETYPE_SOURCE),
        expected_cvr=Band(0.015, 0.020, _ARCHETYPE_SOURCE),
    ),
    Archetype.FOUNDER_TRAUMA: ArchetypePrior(
        hook_rate=Band(0.30, 0.35, _ARCHETYPE_SOURCE),
        hold_rate=Band(0.35, 0.42, _ARCHETYPE_SOURCE),
        outbound_ctr=Band(0.015, 0.025, _ARCHETYPE_SOURCE),
        expected_cvr=Band(0.035, 0.050, _ARCHETYPE_SOURCE),
    ),
    Archetype.ANTI_HERO_RANT: ArchetypePrior(
        hook_rate=Band(0.40, 0.48, _ARCHETYPE_SOURCE),
        hold_rate=Band(0.25, 0.30, _ARCHETYPE_SOURCE),
        outbound_ctr=Band(0.025, 0.045, _ARCHETYPE_SOURCE),
        expected_cvr=Band(0.025, 0.040, _ARCHETYPE_SOURCE),
    ),
    Archetype.AUTONOMOUS_VOXEL: ArchetypePrior(
        hook_rate=Band(0.35, 0.45, _ARCHETYPE_SOURCE),
        hold_rate=Band(0.20, 0.25, _ARCHETYPE_SOURCE),
        outbound_ctr=Band(0.018, 0.028, _ARCHETYPE_SOURCE),
        expected_cvr=Band(0.020, 0.035, _ARCHETYPE_SOURCE),
    ),
}


SEMANTIC_TIER_PRIORS: dict[SemanticTier, SemanticPrior] = {
    SemanticTier.GENERIC_SLOP: SemanticPrior(
        cpc_usd=Band(3.50, 5.00, _SEMANTIC_SOURCE),
        cvr=0.008,
    ),
    SemanticTier.FEATURE_HEAVY: SemanticPrior(
        cpc_usd=Band(1.80, 1.80, _SEMANTIC_SOURCE),
        cvr=0.015,
    ),
    SemanticTier.HYPER_SPECIFIC_ENEMY: SemanticPrior(
        cpc_usd=Band(0.85, 0.85, _SEMANTIC_SOURCE),
        cvr=0.042,
    ),
    SemanticTier.TRIBAL_IDENTITY: SemanticPrior(
        cpc_usd=Band(0.45, 0.45, _SEMANTIC_SOURCE),
        cvr=0.068,
    ),
}


# The inversion this table encodes, stated once where it does its damage:
# ASPIRATION_STATUS is the *cheaper* acquisition ($30 CAC) and the *worse*
# business (45% month-3 churn, $250 LTV); EXHAUSTION_RELIEF acquires at more
# than twice the cost ($65) and returns five times the lifetime value
# ($1,200 at 8% churn). Any objective that minimises CAC -- or maximises
# conversions per dollar, which is the same thing -- therefore selects the
# wrong vector and compounds the error every day it runs. This is why
# ``blotto.game.payoff`` rewards contribution margin and never CAC.
#
# Only two vectors are measured in the corpus. The rest are ``None`` rather
# than interpolated: a plausible-looking guess placed here would be
# indistinguishable from a measurement downstream, and the whole point of the
# source rule is that nothing in this file is allowed to be folklore.
VECTOR_PRIORS: dict[EmotionalVector, VectorPrior | None] = {
    EmotionalVector.ASPIRATION_STATUS: VectorPrior(
        cac_usd=Band(30.0, 30.0, _SEMANTIC_SOURCE),
        m3_churn=0.45,
        ltv_usd=Band(250.0, 250.0, _SEMANTIC_SOURCE),
    ),
    EmotionalVector.EXHAUSTION_RELIEF: VectorPrior(
        cac_usd=Band(65.0, 65.0, _SEMANTIC_SOURCE),
        m3_churn=0.08,
        ltv_usd=Band(1200.0, 1200.0, _SEMANTIC_SOURCE),
    ),
    # Unmeasured in the corpus.
    EmotionalVector.ANGER_INJUSTICE: None,
    # Unmeasured in the corpus.
    EmotionalVector.FEAR_LOSS: None,
    # Unmeasured in the corpus.
    EmotionalVector.ANALYTICAL_PROOF: None,
}


_PLATFORM_METRICS_SOURCE = (
    "GTM Engineer/Encyclopedia/02_Metrics_and_Baselines_2026.md"
)
_PLATFORM_CREATIVE_SOURCE = (
    "GTM Engineer/Encyclopedia/11_Paid_Channels_Deep_Dive.md"
)
_PLATFORM_LANDING_SOURCE = (
    "GTM Engineer/Encyclopedia/08_Landing_Pages_and_CRO.md"
)


def _platform_prior(
    engagement: tuple[float, float],
    reach: tuple[float, float],
    lifespan: tuple[float, float],
    creatives: tuple[float, float],
    lp_cvr: tuple[float, float],
) -> PlatformPrior:
    """Assemble a PlatformPrior with per-field citations, so that a Band
    extracted from its table still knows where it came from."""

    return PlatformPrior(
        engagement_rate=Band(*engagement, _PLATFORM_METRICS_SOURCE),
        organic_reach_share=Band(*reach, _PLATFORM_METRICS_SOURCE),
        creative_lifespan_days=Band(*lifespan, _PLATFORM_CREATIVE_SOURCE),
        min_new_creatives_per_week=Band(*creatives, _PLATFORM_CREATIVE_SOURCE),
        landing_page_cvr=Band(*lp_cvr, _PLATFORM_LANDING_SOURCE),
    )


PLATFORM_PRIORS: dict[Platform, PlatformPrior] = {
    Platform.X: _platform_prior(
        (0.01, 0.03), (0.05, 0.15), (7, 21), (5, 10), (0.03, 0.06)
    ),
    Platform.LINKEDIN: _platform_prior(
        (0.02, 0.05), (0.08, 0.20), (21, 42), (1, 2), (0.02, 0.05)
    ),
    Platform.TIKTOK: _platform_prior(
        (0.03, 0.08), (0.15, 0.40), (5, 14), (10, 15), (0.005, 0.02)
    ),
    Platform.INSTAGRAM: _platform_prior(
        (0.01, 0.03), (0.08, 0.15), (7, 21), (5, 10), (0.03, 0.06)
    ),
    Platform.YOUTUBE_SHORTS: _platform_prior(
        (0.02, 0.05), (0.08, 0.20), (30, 90), (1, 2), (0.03, 0.06)
    ),
    Platform.REDDIT: _platform_prior(
        (0.01, 0.03), (0.05, 0.15), (14, 30), (1, 2), (0.03, 0.06)
    ),
    Platform.THREADS: _platform_prior(
        (0.01, 0.03), (0.05, 0.15), (7, 21), (5, 10), (0.03, 0.06)
    ),
}


# ---------------------------------------------------------------------------
# Observation degradation. Why these live with the game priors rather than in
# an analytics layer: they parameterise the imperfect-information structure of
# the game itself. A planner that assumes it sees the world clearly is not
# planning for this game.
# ---------------------------------------------------------------------------

_ATTRIBUTION_SOURCE = "GTM Engineer/Encyclopedia/09_Attribution_and_Analytics.md"

CLIENT_SIDE_EVENT_LOSS = Band(0.25, 0.40, _ATTRIBUTION_SOURCE)
"""Ad blockers, ITP and Privacy Sandbox block this share of client-side
events before they are ever reported."""

SERVER_SIDE_RECOVERY = Band(0.20, 0.30, _ATTRIBUTION_SOURCE)
"""Fraction of lost client-side events a server-side tag wins back."""

NON_INCREMENTAL_SHARE = Band(0.20, 0.40, _ATTRIBUTION_SOURCE)
"""Share of attributed conversions that would have happened anyway. The
complement of ``Observation.incrementality``."""

DARK_SOCIAL_B2B_SHARE = Band(
    0.60,
    0.80,
    "GTM Engineer/Encyclopedia/03_Channel_Playbooks_2026.md",
)
"""Unattributable sharing in B2B. Surfaces as direct traffic, which is why
direct traffic is a lagging signal of creative performance, not a channel."""

REPORTING_LAG_HOURS = Band(
    24,
    72,
    "DTC Engineer/21_Media_Buying_Risk_Management_and_Drawdowns.md",
)
"""The window inside which an observation is ``is_partial`` and must not open
an evidence gate. Peeking inside the lag is the specific failure the gates
exist to prevent."""


# ---------------------------------------------------------------------------
# Decay constants. The thresholds at which the platform stops distributing an
# asset, expressed as the measured quantity rather than a score, so the same
# number can be used by the world model (to trigger decay) and by the operator
# (to understand why reach collapsed).
# ---------------------------------------------------------------------------

HOOK_RATE_ABANDON_THRESHOLD = Band(
    0.25,
    0.25,
    "DTC Engineer/13_The_Hook_Library_and_Visual_Pattern_Interrupts.md",
)
"""Below this 2-second watch-through the platform stops serving the asset."""

HOLD_RATE_DECAY_TRIGGER = Band(
    0.15,
    0.15,
    "Storytelling Engineer/"
    "18_Structuring_the_Story_Team_Roles_and_KPI_Mapping.md",
)
"""Fractional drop from baseline hold rate at which the algorithm demotes the
asset. Expressed as a fraction because the baseline is per-account."""

CTR_FATIGUE_TRIGGER = Band(
    0.30,
    0.30,
    "GTM Engineer/Encyclopedia/11_Paid_Channels_Deep_Dive.md",
)
"""Fractional drop from peak CTR at which creative fatigue is declared."""

WINNING_HOOK_BURNOUT_DAYS = Band(
    3,
    5,
    "Storytelling Engineer/"
    "04_Hook_Architecture_The_First_3_Seconds_of_Attention.md",
)
"""Days a proven hook keeps working before the audience pattern-matches it to
an ad and stops watching. The reason a winning angle still needs fresh hooks."""
