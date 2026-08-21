"""The reward function: contribution margin from paying users.

It is not engagement, not reach, and not attributed conversions taken at face
value. ``CodeWorldModel.get_rewards`` promises exactly this, and the reason is
the inversion documented in ``blotto.game.priors.VECTOR_PRIORS``: the
cheapest-CAC emotional vector is the worse business, so any objective that
leans on acquisition cost selects confidently against lifetime value. Reward
has to be denominated in the thing the operator actually keeps.

Deliberate omission, stated so no one "fixes" it: reach does not appear in
the reward, and there is no ``vanity_penalty`` helper to make its absence
look handled. Reach is an input the world model uses to model distribution;
it is not a benefit. Adding it back -- even as a small negative term --
re-introduces the objective the whole architecture exists to escape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from blotto.game.types import Observation

__all__ = [
    "Economics",
    "PayoffHealth",
    "ltv",
    "true_conversions",
    "reward",
    "ltv_cac_ratio",
    "cac_payback_months",
    "health",
]

_EPS = 1e-9
"""Guards divisions so a degenerate input yields a very large number rather
than an exception -- a planner comparing two huge ratios is still behaving
correctly, while a crashed planner is not."""


@dataclass(frozen=True, slots=True)
class Economics:
    """The unit economics every reward is computed against.

    ``allowable_cac_share`` defaults to 0.30 per GTM Engineer/Encyclopedia/
    06_Pricing_and_Packaging.md -- the ceiling on CAC as a share of LTV. The
    other fields are per-business inputs, not corpus constants, and carry no
    citation for that reason.
    """

    arpu_monthly: float
    gross_margin: float
    monthly_churn: float
    cogs_share: float
    fixed_cost_per_post: float
    allowable_cac_share: float = 0.30


def ltv(economics: Economics) -> float:
    """Lifetime contribution of one paying user: ARPU x margin / churn.

    A zero churn rate means nobody ever leaves, which makes lifetime value
    infinite, not undefined -- the guard returns ``math.inf`` so downstream
    ratios stay orderable.
    """
    if economics.monthly_churn <= 0.0:
        return math.inf
    return (
        economics.arpu_monthly * economics.gross_margin / economics.monthly_churn
    )


def true_conversions(observation: Observation) -> float:
    """De-biased conversion count: the number the dashboard would show if it
    could see everything and take credit for nothing it did not cause.

    Two corrections pull in OPPOSITE directions, and an implementation that
    applies only one of them is biased in a specific, damaging way:

    * Dividing by ``attribution_coverage`` grosses UP for conversions the
      trackers never saw. Ignoring this under-counts and systematically
      underrates channels with technical audiences (ad-blocker users vanish
      first).
    * Multiplying by ``incrementality`` discounts DOWN for conversions that
      would have happened anyway. Ignoring this over-counts and rewards
      harvesting demand that was already there -- the failure that makes
      retargeting look like magic.

    Gross up for what you could not see, discount for what you would have got
    anyway. Only the product of the two is an estimate of what the move
    *caused*, and only that belongs in a reward.
    """
    coverage = max(observation.attribution_coverage, _EPS)
    return observation.attributed_conversions / coverage * (
        observation.incrementality
    )


def reward(observation: Observation, economics: Economics, cost: float) -> float:
    """Contribution margin attributable to one observation, minus its cost.

    ``cost`` is passed in rather than read from ``Economics`` because spend
    varies per move (a Scale costs more than a Hold) while unit economics do
    not. ``fixed_cost_per_post`` belongs inside the caller's ``cost`` for
    publishes.
    """
    return true_conversions(observation) * ltv(economics) - cost


def ltv_cac_ratio(economics: Economics, cac_usd: float) -> float:
    """Lifetime value over acquisition cost. The primary ratio, precisely
    because it ranks the expensive-and-durable vector above the cheap-and-
    evaporating one (see ``VECTOR_PRIORS``)."""
    if cac_usd <= 0.0:
        return math.inf
    return ltv(economics) / cac_usd


def cac_payback_months(cac_usd: float, economics: Economics) -> float:
    """Months of contribution needed to recover CAC.

    Payback is the cash-flow view of the same economics: two businesses with
    identical LTV:CAC die differently when one recovers CAC in a month and
    the other in a year, because the second must finance the gap.
    """
    monthly_contribution = economics.arpu_monthly * economics.gross_margin
    if monthly_contribution <= 0.0:
        return math.inf
    return cac_usd / monthly_contribution


class PayoffHealth(Enum):
    """Classification of an LTV:CAC ratio against the corpus bands.

    Source: GTM Engineer/Encyclopedia/02_Metrics_and_Baselines_2026.md.
    The bands 2.0-3.0 and 3.0-5.0 are the crossings between the named states
    rather than independently named corpus figures; ``HEALTHY`` labels the
    gap between marginal and golden so every finite ratio classifies.
    """

    STOP = "stop"
    """Below 1.0: losing money on every acquisition. Stop spend."""

    MARGINAL = "marginal"
    """1.0 to 2.0: survives only if nothing goes wrong."""

    HEALTHY = "healthy"
    """2.0 to 3.0: sound, unremarkable."""

    GOLDEN = "golden"
    """3.0 to 5.0: the corpus's benchmark for a scalable engine."""

    UNDERSPENDING = "underspending"
    """Above 5.0: the constraint is no longer economics but supply -- you are
    leaving growth un-bought, which is its own failure mode."""


def health(ratio: float) -> PayoffHealth:
    """Classify an LTV:CAC ratio.

    Infinity (zero churn, or zero CAC) classifies as UNDERSPENDING, which is
    the correct reading: a ratio with no denominator is not a number to
    admire but a signal that growth is unconstrained by economics.
    """
    if ratio < 1.0:
        return PayoffHealth.STOP
    if ratio < 2.0:
        return PayoffHealth.MARGINAL
    if ratio < 3.0:
        return PayoffHealth.HEALTHY
    if ratio <= 5.0:
        return PayoffHealth.GOLDEN
    return PayoffHealth.UNDERSPENDING
